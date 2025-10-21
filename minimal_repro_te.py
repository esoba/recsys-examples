"""Test te column/row parallel linear"""
import os
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer import TransformerConfig
from megatron.core import parallel_state
import torch
import torch.nn.functional as F
import torch.nn as nn
import transformer_engine.pytorch as te

def initialize_distributed():
    if torch.distributed.is_initialized():
        return
    torch.set_printoptions(precision=6, sci_mode=False)
    rank = int(os.environ["LOCAL_RANK"])
    device: torch.device = torch.device(f"cuda:{rank}")
    backend = "nccl"
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(backend=backend)


def initialize_model_parallel(tensor_model_parallel_size=1):
    if parallel_state.model_parallel_is_initialized():
        return
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size,
    )
    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])

class TestMegatronSplit(MegatronModule):
    def __init__(self, config: TransformerConfig):
        self.config = config
        super().__init__(config=config)
        self._tp_size = parallel_state.get_tensor_model_parallel_world_size()

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
        self._to_category = torch.nn.Linear(128, 10)

    def forward(self, x: torch.Tensor):
        x, _ = self._linear_uvqk(x)
        x = F.silu(x)
        x = x.view(-1, 4//self._tp_size, 128*4)
        u, v, q, k = torch.split(x, [128, 128, 128, 128], dim=-1)
        x = F.scaled_dot_product_attention(q, k, v, is_causal=self.config.is_causal)
        # Reshape attention output from [batch*seq, num_heads_per_partition, head_dim]
        # to [batch*seq, num_heads_per_partition * head_dim] for row parallel linear
        x = x.reshape(-1, (4//self._tp_size) * 128)
        print(x, flush=True)
        print(x.shape, flush=True)
        x, _ = self._linear_proj(x)
        x = self._to_category(x)
        return x


def simple_train(config):
    model = TestMegatronSplit(config)
    print(model.dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(100):
        x = torch.randn(16, 1024, 128, dtype=config.params_dtype).cuda()
        label = torch.randn(16384, 10, dtype=config.params_dtype).cuda()
        with te.fp8_autocast():
            y = model(x)
        loss = torch.nn.functional.mse_loss(y, label, reduction="sum").cuda()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    initialize_distributed()
    initialize_model_parallel(
        tensor_model_parallel_size=1
    )
    config = TransformerConfig(  # type: ignore
        hidden_size=128,
        kv_channels=128,
        num_attention_heads=4,
        hidden_dropout=0.2,
        layernorm_epsilon=1e-5,
        fp8="hybrid",
        fp8_recipe="tensorwise",
        params_dtype = torch.bfloat16,  # From Megatron config
    )
    simple_train(config)