from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Optional

import torch
from torch import nn
from torch.optim import Optimizer

from esp_ppq.IR import BaseGraph, Operation
from esp_ppq.executor.torch import TorchExecutor


class BaseReplacementStrategy(ABC):
    """
    Abstract base class for subgraph replacement strategies.

    Subclasses define which ops to replace and how to build replacement
    nn.Modules.  The engine (AdaptiveSubgraphOptimizationPass) handles
    tracing, splicing, distillation, rollback, and quality evaluation.
    """

    def __init__(self, **kwargs):
        self.params = kwargs

    @abstractmethod
    def select_target(self, op: Operation, block: Any, graph: BaseGraph) -> bool:
        """
        Should this op be replaced?
        Has full visibility into the block and graph for context-dependent decisions.
        """
        pass

    @abstractmethod
    def build_replacement(self, op: Operation, block: Any, graph: BaseGraph) -> Tuple[nn.Module, list]:
        """
        Build the replacement nn.Module for the given op.
        Returns (module, input_shape) — the engine handles tracing and splicing.
        """
        pass

    @abstractmethod
    def calculate_samples(self, block, block_ops: List[Operation], dataloader_size: int) -> int:
        """
        Return the number of dataloader batches to cache for this block's
        distillation. Receives the block, its operations, and the total
        dataloader size so strategies can make informed decisions.
        Return dataloader_size to use all available data.
        """
        pass

    @abstractmethod
    def evaluate_validation(self, 
                            executor: TorchExecutor, 
                            block_ops: List[Operation], 
                            input_wire: str, 
                            output_wire: str, 
                            val_inputs: List[torch.Tensor], 
                            val_outputs: List[torch.Tensor],
                            executing_device: str) -> bool:
        """
        Evaluate all validation batches and return True if the block should
        be accepted.
        """
        pass

    @abstractmethod
    def get_criterion(self) -> nn.Module:
        pass

    @abstractmethod
    def get_scheduler(self, optimizer: Optimizer, steps: int) -> Optional[Any]:
        pass

    @property
    @abstractmethod
    def learning_rate(self) -> float:
        pass

    @property
    @abstractmethod
    def weight_decay(self) -> float:
        pass

    @property
    @abstractmethod
    def steps(self) -> int:
        """Total number of gradient update steps (flat, like TQT)."""
        pass

    @property
    @abstractmethod
    def patience(self) -> int:
        pass

    def build_replacement_subgraph(self, op: Operation, block: Any, graph: BaseGraph,
                                   module_to_subgraph_fn) -> Tuple[BaseGraph, list]:
        """Build a PPQ subgraph to replace the target op.

        Default: delegates to build_replacement() → nn.Module → ONNX → PPQ.
        Override this directly to build PPQ ops natively (no ONNX round-trip).

        Returns:
            (sub_graph, input_shape) — sub_graph has .inputs and .outputs set.
        """
        module, input_shape = self.build_replacement(op, block, graph)
        return module_to_subgraph_fn(module, input_shape=input_shape), input_shape

    # ──────────────────────────────────────────────────────────────────
    # OPTIONAL HOOK: per-step callback
    # ──────────────────────────────────────────────────────────────────
    def on_step_end(self, trainable_params: List[torch.Tensor], step: int, **kwargs):
        """Called after each optimizer step during distillation.

        Override this to apply per-step constraints (e.g. masking pruned
        channels, enforcing sparsity, clamping weights). Default: no-op.

        Args:
            trainable_params: list of weight tensors being optimized
            step: current training step index
        """
        pass

    # ──────────────────────────────────────────────────────────────────
    # IMPORTANT: Block Extension for Activation Replacement Strategies
    # ──────────────────────────────────────────────────────────────────
    # When a strategy replaces an activation op (e.g. SiLU → HardSiLU),
    # the predecessor op (e.g. Conv) must be in the SAME block so that
    # distillation can adjust its weights to compensate for the output
    # distribution shift caused by the activation change.
    #
    # If requires_predecessor is True, the engine will extend each block
    # by one op when the block's endpoint feeds directly into a target op.
    # This prevents target ops from being stranded at block boundaries
    # where their predecessor is in the previous (already-processed) block.
    #
    # Strategies that replace computing ops (e.g. Conv → DW+PW) do NOT
    # need this — the replaced op IS the computing op, so no predecessor
    # compensation is needed. Set to False for those strategies.
    # ──────────────────────────────────────────────────────────────────
    @property
    def requires_predecessor(self) -> bool:
        return False

    @property
    def compensate_on_reject(self) -> bool:
        """If True, fine-tune the original block after rejection to correct
        upstream drift from previously accepted replacements."""
        return False

