# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Dict, Tuple

import torch
from configs import HSTUConfig, RankingConfig
from dataset.utils import RankingBatch
from distributed.dmp_to_tp import dmp_batch_to_tp, jt_dict_grad_scaling_and_allgather
from megatron.core import parallel_state
from model.base_model import BaseModel
from modules.embedding import ShardedEmbedding
from modules.hstu_block import HSTUBlock
from modules.metrics import get_multi_event_metric_module
from modules.mlp import MLP
from modules.multi_task_loss_module import MultiTaskLossModule
from torchrec.sparse.jagged_tensor import JaggedTensor
from commons.utils.logger import print_rank_0


def inspect_sharded_embedding_tables(embedding_collection: ShardedEmbedding, table_name_filter: str = None, tracking_state: dict = None) -> dict:
    """
    Helper function to inspect all embedding tables in a ShardedEmbedding collection.
    Works with both regular embeddings and dynamic embeddings.
    
    Args:
        embedding_collection: The ShardedEmbedding instance to inspect
        table_name_filter: Optional filter to only show tables containing this string
        tracking_state: Optional dict to track IDs across iterations {table_name: {'all_ids': set(), 'nan_ids': set()}}
    
    Returns:
        Updated tracking_state dict
    """
    if tracking_state is None:
        tracking_state = {}
    print_rank_0("=" * 80)
    print_rank_0("EMBEDDING COLLECTION INSPECTION")
    print_rank_0("=" * 80)
    
    # Check model-parallel embeddings
    if embedding_collection._model_parallel_embedding_collection is not None:
        print_rank_0("\n[MODEL-PARALLEL EMBEDDINGS]")
        mp_collection = embedding_collection._model_parallel_embedding_collection
        
        # Try to get dynamic embedding modules
        try:
            from dynamicemb.dump_load import get_dynamic_emb_module
            dynamic_modules = get_dynamic_emb_module(mp_collection)
            
            if len(dynamic_modules) > 0:
                print_rank_0(f"Found {len(dynamic_modules)} dynamic embedding module(s)")
                for module_idx, dyn_module in enumerate(dynamic_modules):
                    print_rank_0(f"\n  Dynamic Module {module_idx}:")
                    for table_idx, (table_name, table) in enumerate(zip(dyn_module.table_names, dyn_module.tables)):
                        if table_name_filter and table_name_filter not in table_name:
                            continue
                        print_rank_0(f"    Table '{table_name}':")
                        print_rank_0(f"      Type: Dynamic Embedding (KeyValueTable)")
                        capacity = table.capacity()
                        used_size = table.size()
                        emb_dim = table.embedding_dim()
                        opt_dim = table.optim_state_dim()
                        
                        print_rank_0(f"      Capacity: {capacity}")
                        print_rank_0(f"      Size (used): {used_size}")
                        print_rank_0(f"      Embedding dim: {emb_dim}")
                        print_rank_0(f"      Optimizer state dim: {opt_dim}")
                        print_rank_0(f"      Effective shape: [{used_size}, {emb_dim}] (sparse)")
                        
                        # Check ALL embeddings for NaNs (sparse hash table requires full scan)
                        if used_size > 0:
                            from dynamicemb.dump_load import export_keys_values
                            device = torch.device(f"cuda:{torch.cuda.current_device()}")
                            
                            try:
                                total_checked = 0
                                total_nan_count = 0
                                all_embedding_ids = []
                                ids_with_nan = []
                                emb_min = float('inf')
                                emb_max = float('-inf')
                                has_any_nan = False
                                
                                # Scan through entire hash table to find all stored embeddings
                                for keys, embeddings, opt_states, scores in export_keys_values(table, device, batch_size=65536):
                                    batch_size_actual = embeddings.shape[0]
                                    total_checked += batch_size_actual
                                    
                                    # Track all embedding IDs
                                    all_embedding_ids.extend(keys.tolist())
                                    
                                    # Check for NaNs and get the actual movie IDs with NaN
                                    if torch.isnan(embeddings).any():
                                        has_any_nan = True
                                        nan_mask = embeddings.isnan().any(dim=1)  # Which embeddings have NaN
                                        nan_per_embedding = embeddings.isnan().sum(dim=1)  # Count NaNs per embedding
                                        total_nan_count += torch.isnan(embeddings).sum().item()
                                        ids_with_nan_in_batch = keys[nan_mask].tolist()  # Get actual movie IDs
                                        ids_with_nan.extend(ids_with_nan_in_batch)
                                        
                                        # Store NaN pattern for detailed analysis
                                        if 'nan_patterns' not in tracking_state:
                                            tracking_state['nan_patterns'] = {}
                                        if table_name not in tracking_state['nan_patterns']:
                                            # For each ID with NaN, store how many dimensions have NaN
                                            nan_pattern = {
                                                int(keys[i].item()): int(nan_per_embedding[i].item()) 
                                                for i in range(len(keys)) if nan_mask[i]
                                            }
                                            tracking_state['nan_patterns'][table_name] = nan_pattern
                                    else:
                                        emb_min = min(emb_min, embeddings.min().item())
                                        emb_max = max(emb_max, embeddings.max().item())
                                
                                print_rank_0(f"      Scanned {total_checked}/{used_size} embeddings")
                                print_rank_0(f"      Has NaN: {has_any_nan}")
                                if has_any_nan:
                                    print_rank_0(f"      Total NaN count: {total_nan_count}")
                                    total_elements = total_checked * emb_dim
                                    print_rank_0(f"      NaN percentage: {100.0 * total_nan_count / total_elements:.2f}%")
                                    print_rank_0(f"      Number of embeddings with NaN: {len(ids_with_nan)}")
                                    print_rank_0(f"      IDs with NaN: {ids_with_nan}")
                                    
                                    # Print NaN pattern analysis
                                    if 'nan_patterns' in tracking_state and table_name in tracking_state['nan_patterns']:
                                        nan_pattern = tracking_state['nan_patterns'][table_name]
                                        fully_nan = sum(1 for count in nan_pattern.values() if count == emb_dim)
                                        partially_nan = len(nan_pattern) - fully_nan
                                        print_rank_0(f"      NaN Pattern:")
                                        print_rank_0(f"        Fully NaN embeddings: {fully_nan}/{len(nan_pattern)} (all {emb_dim} dims)")
                                        print_rank_0(f"        Partially NaN embeddings: {partially_nan}/{len(nan_pattern)}")
                                        if partially_nan > 0:
                                            # Show examples of partial NaN
                                            partial_examples = [(id_, count) for id_, count in list(nan_pattern.items())[:5] if count < emb_dim]
                                            if partial_examples:
                                                print_rank_0(f"        Partial NaN examples (ID: NaN_count): {partial_examples}")
                                    
                                    # Store for comparison across iterations
                                    if table_name not in tracking_state:
                                        tracking_state[table_name] = {'all_ids': set(), 'nan_ids': set()}
                                    
                                    prev_all_ids = tracking_state[table_name]['all_ids']
                                    prev_nan_ids = tracking_state[table_name]['nan_ids']
                                    current_all_ids = set(all_embedding_ids)
                                    current_nan_ids = set(ids_with_nan)
                                    
                                    new_ids = current_all_ids - prev_all_ids
                                    new_ids_with_nan = current_nan_ids & new_ids
                                    old_ids_with_nan = current_nan_ids - new_ids
                                    
                                    print_rank_0(f"      New embeddings this iteration: {len(new_ids)}")
                                    print_rank_0(f"      New embeddings with NaN: {len(new_ids_with_nan)} (IDs: {list(new_ids_with_nan)[:10]})")
                                    print_rank_0(f"      Old embeddings corrupted: {len(old_ids_with_nan)} (IDs: {list(old_ids_with_nan)[:10]})")
                                    
                                    # Update tracking
                                    tracking_state[table_name]['all_ids'] = current_all_ids
                                    tracking_state[table_name]['nan_ids'] = current_nan_ids
                                else:
                                    print_rank_0(f"      Min/Max: {emb_min:.4f} / {emb_max:.4f}")
                                    
                                    # Track IDs even when no NaN for future comparison
                                    if table_name not in tracking_state:
                                        tracking_state[table_name] = {'all_ids': set(), 'nan_ids': set()}
                                    tracking_state[table_name]['all_ids'] = set(all_embedding_ids)
                            except Exception as e:
                                print_rank_0(f"      Error scanning embeddings: {e}")
        except ImportError:
            print_rank_0("  Dynamic embeddings module not available")
        
        # Check regular (non-dynamic) embeddings via state_dict
        print_rank_0("\n  Regular embeddings in state_dict:")
        for name, tensor in mp_collection.state_dict().items():
            if table_name_filter and table_name_filter not in name:
                continue
            print_rank_0(f"    '{name}':")
            
            if hasattr(tensor, "local_shards"):
                print_rank_0(f"      Type: ShardedTensor (model-parallel)")
                for shard_idx, shard in enumerate(tensor.local_shards()):
                    shard_tensor = shard.tensor
                    print_rank_0(f"      Shard {shard_idx}:")
                    print_rank_0(f"        Shape: {shard_tensor.shape}")
                    print_rank_0(f"        Offsets: {shard.metadata.shard_offsets}")
                    print_rank_0(f"        Sizes: {shard.metadata.shard_sizes}")
                    print_rank_0(f"        Has NaN: {torch.isnan(shard_tensor).any()}")
                    if torch.isnan(shard_tensor).any():
                        print_rank_0(f"        NaN count: {torch.isnan(shard_tensor).sum()}")
            else:
                print_rank_0(f"      Type: Regular Tensor")
                print_rank_0(f"      Shape: {tensor.shape}")
                print_rank_0(f"      Has NaN: {torch.isnan(tensor).any()}")
                if torch.isnan(tensor).any():
                    print_rank_0(f"      NaN count: {torch.isnan(tensor).sum()}")
    
    # Check data-parallel embeddings
    if embedding_collection._data_parallel_embedding_collection is not None:
        print_rank_0("\n[DATA-PARALLEL EMBEDDINGS]")
        dp_collection = embedding_collection._data_parallel_embedding_collection
        
        if hasattr(dp_collection, 'embedding_weights'):
            for table_name, weight_tensor in dp_collection.embedding_weights.items():
                if table_name_filter and table_name_filter not in table_name:
                    continue
                print_rank_0(f"  Table '{table_name}':")
                print_rank_0(f"    Shape: {weight_tensor.shape}")
                print_rank_0(f"    Has NaN: {torch.isnan(weight_tensor).any()}")
                if torch.isnan(weight_tensor).any():
                    print_rank_0(f"    NaN count: {torch.isnan(weight_tensor).sum()}")
    
    print_rank_0("=" * 80)
    return tracking_state


class RankingGR(BaseModel):
    """
    A class representing the ranking model. Inherits from BaseModel. A ranking model consists of
    a sparse architecture and a dense architecture. A ranking model is able to process multiple labels
    and thus has multiple logit dimensions. Each label is associated with a loss function (e.g. BCE, CE).

    Args:
        hstu_config (HSTUConfig): The HSTU configuration.
        task_config (RankingConfig): The ranking task configuration.
    """

    def __init__(
        self,
        hstu_config: HSTUConfig,
        task_config: RankingConfig,
    ):
        super().__init__()
        self._tp_size = parallel_state.get_tensor_model_parallel_world_size()
        self._device = torch.device("cuda", torch.cuda.current_device())
        self._hstu_config = hstu_config
        self._task_config = task_config

        self._embedding_collection = ShardedEmbedding(task_config.embedding_configs)

        self._hstu_block = HSTUBlock(hstu_config)
        self._mlp = MLP(
            hstu_config.hidden_size,
            task_config.prediction_head_arch,
            task_config.prediction_head_act_type,
            task_config.prediction_head_bias,
            device=self._device,
        )

        # TODO, make reduction configurable
        self._loss_module = MultiTaskLossModule(
            num_classes=task_config.prediction_head_arch[-1],
            num_tasks=task_config.num_tasks,
            reduction="none",
        )
        self._metric_module = get_multi_event_metric_module(
            num_classes=task_config.prediction_head_arch[-1],
            num_tasks=task_config.num_tasks,
            metric_types=task_config.eval_metrics,
            comm_pg=parallel_state.get_data_parallel_group(with_context_parallel=True),
        )

    def bfloat16(self):
        """
        Convert the model to use bfloat16 precision. Only affects the dense module.

        Returns:
            RankingGR: The model with bfloat16 precision.
        """
        self._hstu_block.bfloat16()
        self._mlp.bfloat16()
        return self

    def half(self):
        """
        Convert the model to use half precision. Only affects the dense module.

        Returns:
            RankingGR: The model with half precision.
        """
        self._hstu_block.half()
        self._mlp.half()
        return self

    def get_logit_and_labels(
        self, batch: RankingBatch
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the logits and labels for the batch.

        Args:
            batch (RankingBatch): The batch of ranking data.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The logits and labels.
        """
        # DMP embedding

        if not hasattr(self, '_iter_count'):
            self._iter_count = 0
            self._embedding_tracking = {}  # Track embedding IDs across iterations
        self._iter_count += 1
        print_rank_0(f"=== Iteration {self._iter_count} ===")

        # On first iteration, inspect all embedding tables to see the structure
        if self._iter_count == 1:
            print_rank_0("=== ITERATION 1 - Full embedding structure ===")
            self._embedding_tracking = inspect_sharded_embedding_tables(
                self._embedding_collection, tracking_state=self._embedding_tracking
            )

        if self._iter_count == 63:
            print_rank_0("=== ITERATION 63 - Checking embeddings ===")
            # Use helper function to inspect all embedding tables
            self._embedding_tracking = inspect_sharded_embedding_tables(
                self._embedding_collection, table_name_filter="movie_id", tracking_state=self._embedding_tracking
            )
        
        if self._iter_count == 64:
            print_rank_0("=== ITERATION 64 - Checking embeddings (NaNs expected here) ===")
            # Check embeddings at start of iteration 64 to see if NaN already present
            self._embedding_tracking = inspect_sharded_embedding_tables(
                self._embedding_collection, table_name_filter="movie_id", tracking_state=self._embedding_tracking
            )

        print_rank_0("=== Checking movie_id indices ===")
        movie_id_indices = batch.features['movie_id']  
        if hasattr(movie_id_indices, 'values'):
            indices = movie_id_indices.values() if callable(movie_id_indices.values) else movie_id_indices.values
            print_rank_0(f"movie_id indices shape: {indices.shape}")
            if indices.numel() > 0:
                print_rank_0(f"movie_id indices min/max: {indices.min()} / {indices.max()}")
                print_rank_0(f"movie_id unique indices: {indices.unique()[:20]}")  # First 20 unique
            else:
                print_rank_0("movie_id indices is EMPTY on this rank!")

        print_rank_0("=== Checking input to embedding collection ===")
        for key in batch.features.keys():
            indices = batch.features[key].values()
            if indices.numel() > 0 and torch.isnan(indices).any():
                print_rank_0(f"Feature '{key}' indices has NaNs!")
                print_rank_0(f"  Shape: {indices.shape}")
                print_rank_0(f"  Min/Max: {indices.min()} / {indices.max()}")
                print_rank_0(f"  Unique count: {indices.unique().numel()}")
                print_rank_0(f"  First 10 indices: {indices.flatten()[:10]}")
                print_rank_0(f"  NaN count: {torch.isnan(indices).sum()}")

        embeddings: Dict[str, JaggedTensor] = self._embedding_collection(batch.features)

        print_rank_0("=== Checking embedding weights ===")
        for name, param in self._embedding_collection.named_parameters():
            if torch.isnan(param).any():
                print_rank_0(f"EMBEDDING WEIGHT '{name}' HAS NaN!")
                print_rank_0(f"  Shape: {param.shape}")
                print_rank_0(f"  NaN count: {torch.isnan(param).sum()}")
        
        has_any_nan = any(
            torch.isnan(jt.values() if callable(jt.values) else jt.values).any() 
            for jt in embeddings.values()
        )
        print_rank_0(f"Any embeddings pre maybe detachhave NaNs: {has_any_nan}")
        
        # maybe freeze embedding for debugging
        embeddings = self._embedding_collection._maybe_detach(embeddings)
        # For model-parallel embedding, torchrec does gradient division by (tp_size * dp_size). However, we only need to divide by dp size. In such case, we need to scale the gradient by tp_size.
        # But simultaneously, the DP embedding might be scaled by tp_size unintentionally. On the other hand, the DDP will divide the DP embedding gradient by dp_size (allreduce avg).
        # We need to perform allreduce sum across tp ranks after/before the DDP allreduce avg.
        has_any_nan = any(
            torch.isnan(jt.values() if callable(jt.values) else jt.values).any() 
            for jt in embeddings.values()
        )
        print_rank_0(f"Any embeddings have NaNs: {has_any_nan}")
        grad_scaling_factor = self._tp_size
        embeddings = jt_dict_grad_scaling_and_allgather(
            embeddings,
            grad_scaling_factor,
            parallel_state.get_tensor_model_parallel_group(),
        )
        for key, jt in embeddings.items():
            tensor = jt.values() if callable(jt.values) else jt.values
            if torch.isnan(tensor).any():
                print_rank_0(f"After allgather, embeddings['{key}'] has NaNs!")
                #print_rank_0(f"Batch features \n {batch.features}")
                print_rank_0(f"NaN count: {torch.isnan(tensor).sum()}")
        batch = dmp_batch_to_tp(batch)
        # hidden_states is a JaggedData
        hidden_states_jagged, seqlen_after_preprocessor = self._hstu_block(
            embeddings=embeddings,
            batch=batch,
        )
        hidden_states = hidden_states_jagged.values
        logits = self._mlp(hidden_states)
        return logits, seqlen_after_preprocessor, batch.labels

    def forward(  # type: ignore[override]
        self,
        batch: RankingBatch,
    ) -> Tuple[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        """
        Perform the forward pass of the model.

        Args:
            batch (RankingBatch): The batch of ranking data.

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]: The losses and a tuple of losses, logits, and labels.
        """
        (
            jagged_item_logit,
            seqlen_after_preprocessor,
            labels,
        ) = self.get_logit_and_labels(batch)
        losses = self._loss_module(jagged_item_logit.float(), labels)
        return losses, (
            losses.detach(),
            jagged_item_logit.detach(),
            labels.detach(),
            seqlen_after_preprocessor.detach(),  # used to compute achieved flops/s
        )
