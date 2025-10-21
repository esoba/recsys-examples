"""Test te column/row parallel linear"""
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.transformer.module import MegatronModule
from configs.hstu_config import HSTUConfig
from megatron.core import parallel_state
import torch
import torch.nn.functional as F
import torch.nn as nn
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling, Float8CurrentScaling, Float8BlockScaling
import commons.utils.initialize as init
from megatron.core.transformer.module import Float16Module
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.distributed import finalize_model_grads

global_recipe = Float8CurrentScaling(fp8_format = Format.HYBRID)
class TestMegatronSplit(MegatronModule):
    def __init__(self, config: HSTUConfig):
        self.config = config
        super().__init__(config=config)
        self._tp_size = parallel_state.get_tensor_model_parallel_world_size()
        print(f"tp_size = {self._tp_size}")

        self._linear_uvqk = TEColumnParallelLinear(
            input_size=128,
            output_size=128*4*4,
            init_method=config.init_method,
            config=config,
            bias=True,
            gather_output=False,
            skip_bias_add=False,  # note: TEColumnParallelLinear does not support bias fusion!
            is_expert=False,
        )

        self._linear_proj = TERowParallelLinear(
            input_size=128*4,  # num_heads * head_dim, NOT output of column parallel
            output_size=128,
            init_method=config.init_method,
            config=config,
            input_is_parallel=True,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
        )

    def forward(self, x: torch.Tensor):
        x, _ = self._linear_uvqk(x)
        x = F.silu(x)
        x = x.view(-1, 4//self._tp_size, 128*4)
        u, v, q, k = torch.split(x, [128, 128, 128, 128], dim=-1)
        x = F.scaled_dot_product_attention(q, k, v, is_causal=self.config.is_causal)
        # Reshape attention output from [batch*seq, num_heads_per_partition, head_dim]
        # to [batch*seq, num_heads_per_partition * head_dim] for row parallel linear
        x = x.reshape(-1, (4//self._tp_size) * 128)
        # print(x, flush=True)
        # print(x.shape, flush=True)
        x, _ = self._linear_proj(x)
        return x


def simple_train(config):
    model = TestMegatronSplit(config)
    model = Float16Module(config, model)
    model.train()
    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=True,
        overlap_grad_reduce=False,
        use_distributed_optimizer=False,
        check_for_nan_in_grad=False,
        bucket_size=True,
    )
    model = DDP(config, ddp_config, model)
    #print(next(model.parameters()).dtype, flush=True)
    dense_optimizer_config = OptimizerConfig(
        optimizer="adam",
        lr=1e-3,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
        params_dtype=torch.float16 if config.fp16 else torch.bfloat16,
        bf16=False,
        fp16=True,
        weight_decay=0.0,
    )
    optimizer = get_megatron_optimizer(dense_optimizer_config, [model])
    for i in range(100):
        optimizer.zero_grad()
        x = torch.randn(16, 1024, 128).cuda()
        label = torch.randn(16384, 128).cuda()
        with te.fp8_autocast(enabled=True, fp8_recipe=global_recipe):
            y = model(x)
            loss = torch.nn.functional.mse_loss(y.float(), label.float(), reduction="sum").cuda()
        loss.backward()
        finalize_model_grads([model], None)
        optimizer.step()

if __name__ == "__main__":
    init.initialize_distributed()
    init.initialize_model_parallel(
        tensor_model_parallel_size=2
    )
    config = HSTUConfig(  # type: ignore
        hidden_size=128,
        kv_channels=128,
        num_attention_heads=4,
        hidden_dropout=0.2,
        layernorm_epsilon=1e-5,
        num_layers=1,
        bf16=True,
        tensor_model_parallel_size=parallel_state.get_tensor_model_parallel_world_size(),
        pipeline_model_parallel_size=parallel_state.get_pipeline_model_parallel_world_size(),
        context_parallel_size=parallel_state.get_pipeline_model_parallel_world_size(),
        fp16=False,
        is_causal=True,
        target_group_size=1,
        learnable_input_layernorm=True,
        residual=True,
        async_wgrad=False,
        async_wgrad_stream=None,
        async_wgrad_event=None,
        recompute_input_layernorm=False,
        recompute_input_silu=False,
        add_uvqk_bias=True,
        is_inference=False,
        fuse_norm_mul_dropout=True,
        hstu_attn_quantization_mode=0,
        fp8="hybrid",
        fp8_recipe="tensorwise",
        # params_dtype = torch.bfloat16,  # From Megatron config
    )
    simple_train(config)