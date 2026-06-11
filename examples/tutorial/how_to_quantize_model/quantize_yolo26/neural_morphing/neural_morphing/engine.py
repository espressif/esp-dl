import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple

from esp_ppq.IR import BaseGraph, Operation, Variable
from esp_ppq.quantization.optim import QuantizationOptimizationPass
from esp_ppq.executor.torch import TorchExecutor
from esp_ppq.quantization.algorithm.training import BlockBuilder
from esp_ppq.api import load_torch_model
from .interface import BaseReplacementStrategy


@dataclass
class EngineConfig:
    """Parsed and validated arguments for the optimize() loop."""
    target_op_types:  list
    # Maximum topological depth distance from the block's start op (sp) to its
    # end op (ep). This is NOT an operation count — it measures the longest path
    # length in the graph.  E.g. block_size=2 allows depth 0→1→2, which yields
    # up to 3 ops in a linear chain (and more if the block extension adds a
    # successor via requires_predecessor).
    block_size:       int
    dataloader:       list
    val_dataloader:   list
    executing_device: str
    caching_device:   str
    strategy:         BaseReplacementStrategy
    plots_dir:        Optional[str] = None


class AdaptiveSubgraphOptimizationPass(QuantizationOptimizationPass):
    """
    Replaces target operations with equivalent subgraphs and distills
    their weights block-by-block using a teacher-student scheme.

    The strategy (BaseReplacementStrategy) controls what gets replaced
    and how the replacement module is built.  The engine handles everything
    else: tracing, splicing, distillation, rollback, and quality evaluation.
    """

    def __init__(self, name: str = 'Adaptive Subgraph Optimization Pass'):
        super().__init__(name=name)
        
    def _parse_arguments(self, kwargs: dict) -> EngineConfig:
        strategy = kwargs.get("strategy")
        if not isinstance(strategy, BaseReplacementStrategy):
            raise ValueError("[Engine Error] No BaseReplacementStrategy provided!")

        return EngineConfig(
            target_op_types  = kwargs.get("target_op_types", []),
            block_size       = kwargs.get("block_size", 10),
            dataloader       = kwargs.get("dataloader", []),
            val_dataloader   = kwargs.get("val_dataloader", []),
            executing_device = kwargs.get("executing_device", "cuda"),
            caching_device   = kwargs.get("caching_device", "cpu"),
            strategy         = strategy,
            plots_dir        = kwargs.get("plots_dir", None),
        )

    # ─────────────────────────────────────────────────────────────────────
    # GENERAL PRIMITIVES (strategy-independent)
    # ─────────────────────────────────────────────────────────────────────

    def _module_to_subgraph(self, module: nn.Module, **kwargs) -> BaseGraph:
        """
        Trace a PyTorch nn.Module into a PPQ BaseGraph suitable for splice().

        The traced graph has its entrance and exit boundary variables marked
        in sub_graph.inputs / sub_graph.outputs so that splice() can wire
        them into the main graph.
        """
        input_shape = kwargs.get("input_shape")
        if input_shape is None:
            raise ValueError("[Engine Error] _module_to_subgraph() requires 'input_shape'.")

        # Force CPU for tracing — we only need the topology, not the device.
        # The engine handles device placement during distillation.
        module = module.cpu()
        dummy_input = torch.randn(input_shape)

        # Temporarily disable BN fusion/replacement to avoid crashes on
        # subgraph templates that have isolated (unfused) BatchNorm layers.
        # IMPORTANT: We must set flags on esp_ppq.api.interface directly,
        # because format_graph() reads them from its own module namespace.
        import esp_ppq.api.interface as ppq_interface
        _orig_fuse_bn    = ppq_interface.FORMATTER_FUSE_BN
        _orig_replace_bn = ppq_interface.FORMATTER_REPLACE_BN_TO_CONV
        ppq_interface.FORMATTER_FUSE_BN            = False
        ppq_interface.FORMATTER_REPLACE_BN_TO_CONV = False
        try:
            sub_graph = load_torch_model(model=module, sample=dummy_input, device='cpu')
        finally:
            ppq_interface.FORMATTER_FUSE_BN            = _orig_fuse_bn
            ppq_interface.FORMATTER_REPLACE_BN_TO_CONV = _orig_replace_bn

        # Identify entrance (no source op) and exit (no dest ops) variables
        entrance_var, exit_var = None, None
        for var in sub_graph.variables.values():
            if var.source_op is None and not var.is_parameter:
                entrance_var = var
                break
        for var in sub_graph.variables.values():
            if len(var.dest_ops) == 0:
                exit_var = var
                break

        if entrance_var is None or exit_var is None:
            raise RuntimeError("[Engine Error] Failed to identify topological boundaries of the subgraph.")

        sub_graph.inputs[entrance_var.name] = entrance_var
        sub_graph.outputs[exit_var.name] = exit_var
        return sub_graph

    def splice(self, main_graph: BaseGraph, target_op: Operation, sub_graph: BaseGraph, block: Any) -> None:
        """
        Replace a single operation in the graph with an equivalent subgraph.

        Handles namespace prefixing, graph wiring, rollback-record stamping,
        block boundary sync, and topological re-sort.

        NOTE: This method writes to PPQ private fields (_name, _operations,
        _variables, _source_op) because PPQ's public append_operation() and
        append_variable() have circular validation:
          - append_operation() requires its input variables to already be registered
          - append_variable() requires its dest_ops to already be registered
        Direct dict writes are the only way to perform surgical insertion.
        """
        prefix = target_op.name
        print(f"       Replacing [{prefix}]...")

        # --- STEP 0: RENAME & REGISTER SUBGRAPH OPS ---
        all_sub_ops  = list(sub_graph.operations.values())
        all_sub_vars = list(sub_graph.variables.values())

        # Identify boundary activation variables (not parameters) — these are
        # handled by the wiring step and must NOT be re-registered.
        sub_entrance_var = list(sub_graph.inputs.values())[0]
        sub_exit_var     = list(sub_graph.outputs.values())[0]

        # Prefix all ops and internal variables with the target op name
        for op in all_sub_ops:
            op._name = f"{prefix}_{op._name}"
            op.attributes['_parent_op'] = prefix  # rollback origin stamp

        for var in all_sub_vars:
            if var is sub_entrance_var or var is sub_exit_var:
                continue  # boundary wires handled by wiring step
            var._name = f"{prefix}_{var._name}"

        # Inject into the main graph's internal dicts (bypasses circular validation)
        for op in all_sub_ops:
            if op._name not in main_graph._operations:
                main_graph._operations[op._name] = op

        for var in all_sub_vars:
            if var is sub_entrance_var or var is sub_exit_var:
                continue
            if var._name not in main_graph._variables:
                main_graph._variables[var._name] = var

        # --- STEP 1: REWIRE GRAPH CONNECTIONS ---
        # Entrance: wire input_var to ALL ops in the subgraph that consumed
        # the subgraph's entrance variable. This handles fan-out subgraphs
        # (e.g. HardSwish: x feeds both Add and Mul).
        input_var = target_op.inputs[0]
        entrance_consumers = list(sub_entrance_var.dest_ops)  # ops that read the input
        first_consumer = entrance_consumers[0]

        # Replace target_op with the first consumer in input_var.dest_ops
        input_var.dest_ops[input_var.dest_ops.index(target_op)] = first_consumer
        # Add remaining consumers
        for consumer in entrance_consumers[1:]:
            if consumer not in input_var.dest_ops:
                input_var._dest_ops.append(consumer)

        # Point each consumer's input slot from sub_entrance_var → input_var
        for consumer in entrance_consumers:
            for i, v in enumerate(consumer.inputs):
                if v is sub_entrance_var:
                    consumer.inputs[i] = input_var
        
        # Exit: link sub_exit's output to target_op's downstream consumers
        sub_exit     = all_sub_ops[-1]
        sub_exit_var = sub_exit.outputs[0]
        target_exit_var = target_op.outputs[0]

        # Check if the target's output was a graph output (before it gets removed)
        was_graph_output = target_exit_var.name in main_graph.outputs

        # create_link_with_var(A, B) asserts B.source_op is None — clear it first
        target_exit_var._source_op = None

        # Guard: if sub_exit_var's ONNX name collides with an existing variable
        # (e.g. conv1x1's output is also named '4'), rename to avoid corruption
        if sub_exit_var._name in main_graph._variables:
            sub_exit_var._name = f"{prefix}_exit"

        # Register sub_exit_var so graph.variables[name] returns THIS object
        main_graph._variables[sub_exit_var._name] = sub_exit_var

        # create_link_with_var transfers all downstream references from
        # target_exit_var to sub_exit_var, then removes target_exit_var.
        main_graph.create_link_with_var(A=sub_exit_var, B=target_exit_var)

        # Re-register as graph output if the original was one
        if was_graph_output:
            main_graph.outputs[sub_exit_var.name] = sub_exit_var

        # --- STEP 2: WRITE ROLLBACK METADATA ---
        # Every replacement op gets the complete undo record so rollback
        # is deterministic and needs zero searching.
        entrance_wire_name = input_var.name
        exit_wire_name     = sub_exit_var._name
        for op in all_sub_ops:
            op.attributes['_parent_entrance_wire'] = entrance_wire_name
            op.attributes['_parent_exit_wire']     = exit_wire_name

        # --- STEP 3: REMOVE ORIGINAL OP ---
        main_graph.remove_operation(target_op)

        # --- STEP 4: UPDATE BLOCK BOUNDARIES ---
        idx = block.rps.index(target_op)
        block.rps.pop(idx)
        new_ops = list(sub_graph.operations.values())
        for i, op in enumerate(new_ops):
            block.rps.insert(idx + i, op)
            
        if block.sp == target_op: block.sp = first_consumer
        if block.ep == target_op: block.ep = sub_exit

        # --- STEP 5: RE-SORT GRAPH ---
        main_graph.topological_sort()
        print(f"       Replaced. Block updated.")

    def rollback(self, graph: BaseGraph, fp32_graph_clone: BaseGraph, block: Any) -> None:
        """
        Undo all replacements in a block using the FP32 clone and the
        genealogy stamps (_parent_op, _parent_entrance_wire, _parent_exit_wire)
        written by splice().

        This is fully general — it works for any replacement topology
        (DW+PW, inverted bottleneck, etc.) because it only reads the stamps.
        """
        offspring_by_parent = {}
        for op in list(block.rps):
            parent_name = op.attributes.get('_parent_op')
            if parent_name is not None:
                offspring_by_parent.setdefault(parent_name, []).append(op)

        if not offspring_by_parent:
            return

        for parent_name, offspring_ops in offspring_by_parent.items():
            if parent_name not in fp32_graph_clone.operations:
                continue

            first              = offspring_ops[0]
            entrance_wire_name = first.attributes['_parent_entrance_wire']
            exit_wire_name     = first.attributes['_parent_exit_wire']
            fp32_op            = fp32_graph_clone.operations[parent_name]

            # Snapshot wires before any modification
            entrance_var = graph.variables[entrance_wire_name]
            exit_var_now = graph.variables[exit_wire_name]

            # Build restored_op from FP32 snapshot
            restored_op = Operation(
                name=fp32_op.name,
                op_type=fp32_op.type,
                attributes={k: v for k, v in fp32_op.attributes.items()
                            if not k.startswith('_')}
            )

            # Restore ALL parameter inputs (weight, bias, etc.) from FP32 clone.
            # inputs[0] is the activation wire — handled separately.
            # inputs[1:] are parameters (weight, optional bias, etc.).
            restored_params = []
            for fp32_var in fp32_op.inputs[1:]:
                var_copy = Variable(
                    name=fp32_var.name,
                    value=fp32_var.value.clone(),
                    is_parameter=True
                )
                var_copy._dest_ops = [restored_op]
                restored_params.append(var_copy)

            restored_op._input_vars = [entrance_var] + restored_params
            # Reuse exit_var as restored_op's output — downstream ops already read from it
            restored_op._output_vars = [exit_var_now]

            # Re-route: the exit wire is now produced by the restored op
            exit_var_now._source_op = restored_op

            # Swap entrance_var.dest_ops: remove ALL offspring refs, add restored_op
            offspring_set_local = set(offspring_ops)
            entrance_var._dest_ops = [
                ref for ref in entrance_var._dest_ops if ref not in offspring_set_local
            ]
            entrance_var._dest_ops.append(restored_op)

            # Purge offspring: weight vars, internal activation wires, the ops
            for op in offspring_ops:
                for var in list(op._input_vars):
                    if var.is_parameter and var._name in graph._variables:
                        del graph._variables[var._name]
                for var in list(op._output_vars):
                    if var is not exit_var_now and var._name in graph._variables:
                        del graph._variables[var._name]
                if op._name in graph._operations:
                    del graph._operations[op._name]
                if op in block.rps:
                    block.rps.remove(op)

            # Inject restored_op and all its parameter variables into graph
            graph._operations[restored_op._name] = restored_op
            for param_var in restored_params:
                graph._variables[param_var._name] = param_var

            # Sync block boundaries
            block.rps.append(restored_op)
            offspring_set = set(offspring_ops)
            if block.sp in offspring_set: block.sp = restored_op
            if block.ep in offspring_set: block.ep = restored_op

            # Restore predecessor weights if they were trained during distillation.
            # This happens when requires_predecessor=True and the replacement ops
            # are parameterless (e.g. HardSwish). The predecessor's weights were
            # modified by the optimizer and must be reverted from the FP32 clone.
            predecessor = entrance_var.source_op
            if predecessor is not None and predecessor._name in fp32_graph_clone.operations:
                fp32_pred = fp32_graph_clone.operations[predecessor._name]
                for orig_var, fp32_var in zip(predecessor.inputs, fp32_pred.inputs):
                    if orig_var.is_parameter and fp32_var.is_parameter:
                        orig_var.value = fp32_var.value.clone()

        graph.topological_sort()

    # ─────────────────────────────────────────────────────────────────────
    # INTERNAL HELPERS
    # ─────────────────────────────────────────────────────────────────────

    def _compensate_rejected_block(
        self,
        graph:             BaseGraph,
        block:             Any,
        block_idx:         int,
        total_blocks:      int,
        strategy:          'BaseReplacementStrategy',
        dataloader:        list,
        val_dataloader:    list,
        student_executor:  TorchExecutor,
        teacher_executor:  TorchExecutor,
        fp32_graph_clone:  BaseGraph,
        executing_device:  str,
        caching_device:    str,
    ) -> None:
        """Fine-tune a rejected block's original ops to correct upstream drift.

        After rollback, the block has original ops + original weights.
        The block's INPUT may be perturbed from upstream replacements.
        This pass trains the original op's params (+ predecessor if required)
        to match the teacher output, correcting accumulated drift.
        """
        print(f"  ---> Compensation: fine-tuning original block...")

        required_samples = strategy.calculate_samples(
            block, block.rps, len(dataloader))

        # --- Cache collection (train + val) ---
        comp_inputs, comp_outputs, comp_in_wire, comp_out_wire = \
            self._collect_block_cache(
                block, f"Caching Comp Train: {block_idx}/{total_blocks}",
                required_samples, dataloader,
                student_executor, teacher_executor, fp32_graph_clone,
                executing_device, caching_device,
            )

        comp_val_inputs, comp_val_outputs = [], []
        if val_dataloader:
            comp_val_inputs, comp_val_outputs, _, _ = \
                self._collect_block_cache(
                    block, f"Caching Comp Val: {block_idx}/{total_blocks}",
                    len(val_dataloader), val_dataloader,
                    student_executor, teacher_executor, fp32_graph_clone,
                    executing_device, caching_device,
                )

        # --- Find trainable params (mirrors Phase C logic) ---
        comp_params = []
        target_ops = [op for op in block.rps
                      if strategy.select_target(op, block, graph)]

        for op in target_ops:
            for var in op.inputs:
                if (var.is_parameter and var.value is not None
                        and var.value.is_floating_point()):
                    if not var.value.requires_grad:
                        var.value.requires_grad_()
                    if not any(var.value is p for p in comp_params):
                        comp_params.append(var.value)

        if strategy.requires_predecessor:
            for op in target_ops:
                predecessor = op.inputs[0].source_op
                if predecessor is not None and predecessor in block.rps:
                    for var in predecessor.inputs:
                        if (var.is_parameter and var.value is not None
                                and var.value.is_floating_point()):
                            if not var.value.requires_grad:
                                var.value.requires_grad_()
                            if not any(var.value is p for p in comp_params):
                                comp_params.append(var.value)

        if not comp_params:
            print(f"  ---> No trainable params — skipped.")
            del comp_inputs, comp_outputs, comp_val_inputs, comp_val_outputs
            return

        # --- Training loop (identical structure to Phase C) ---
        criterion = strategy.get_criterion()
        comp_lr = strategy.learning_rate * 0.25  # 4x gentler than main distillation
        optimizer = torch.optim.AdamW(
            comp_params, lr=comp_lr,
            weight_decay=strategy.weight_decay)
        # Simple cosine decay — no warm restarts for compensation
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=strategy.steps, eta_min=1e-6)
        dataset_len = len(comp_inputs)
        eval_every = max(1, dataset_len)

        best_val_loss = float('inf')
        patience_counter = 0
        starting_loss, final_loss = None, None
        last_val_loss = None

        step_pbar = tqdm(range(strategy.steps),
                         desc=f"Compensating {block_idx}/{total_blocks}",
                         leave=True, dynamic_ncols=True)

        for step in step_pbar:
            idx = step % dataset_len
            optimizer.zero_grad()
            feed_dict = {comp_in_wire: comp_inputs[idx].to(executing_device)}
            preds = student_executor.partial_graph_forward(
                operations=block.rps, feed_dict=feed_dict,
                output_names=[comp_out_wire])
            loss = criterion(preds[0], comp_outputs[idx].to(executing_device))
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

            train_loss = loss.item()
            if starting_loss is None:
                starting_loss = train_loss
            final_loss = train_loss

            # Periodic validation
            if comp_val_inputs and (step + 1) % eval_every == 0:
                val_loss_sum = 0.0
                with torch.no_grad():
                    for val_in, val_out in zip(comp_val_inputs, comp_val_outputs):
                        feed_dict = {comp_in_wire: val_in.to(executing_device)}
                        val_preds = student_executor.partial_graph_forward(
                            operations=block.rps, feed_dict=feed_dict,
                            output_names=[comp_out_wire])
                        val_loss_sum += criterion(
                            val_preds[0], val_out.to(executing_device)).item()

                last_val_loss = val_loss_sum / max(len(comp_val_inputs), 1)

                if round(last_val_loss, 5) < round(best_val_loss, 5):
                    best_val_loss = last_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= strategy.patience:
                        print(f"       [Early Stopping] Triggered at Step {step}. Halting.")
                        break

            # Progress bar postfix
            postfix = {"T_Loss": f"{train_loss:.4f}"}
            if last_val_loss is not None:
                postfix["V_Loss"] = f"{last_val_loss:.4f}"
                postfix["P"] = f"{patience_counter}/{strategy.patience}"
            step_pbar.set_postfix(postfix)

        step_pbar.close()

        # --- Final metrics ---
        eval_inputs  = comp_val_inputs  if comp_val_inputs  else comp_inputs
        eval_outputs = comp_val_outputs if comp_val_outputs else comp_outputs

        cos_sims, scale_ratios = [], []
        with torch.no_grad():
            for val_in, val_out in zip(eval_inputs, eval_outputs):
                feed_dict = {comp_in_wire: val_in.to(executing_device)}
                Y_comp = student_executor.partial_graph_forward(
                    operations=block.rps, feed_dict=feed_dict,
                    output_names=[comp_out_wire])[0]
                val_out_dev = val_out.to(executing_device)
                cos_sims.append(
                    F.cosine_similarity(Y_comp, val_out_dev, dim=1).mean().item())
                scale_ratios.append(
                    (torch.norm(Y_comp, p=2) / torch.norm(val_out_dev, p=2)).item())

        mean_cos   = sum(cos_sims)     / max(len(cos_sims), 1)
        mean_scale = sum(scale_ratios) / max(len(scale_ratios), 1)

        print(f"  ---> Compensation complete. "
              f"Loss: {starting_loss:.4f} -> {final_loss:.4f} | "
              f"cos={mean_cos:.4f} scale={mean_scale:.4f}")

        # Re-sync executors
        student_executor._executing_order = graph.topological_sort()
        student_executor.tracing_operation_meta(
            inputs=dataloader[0].to(executing_device))

        del comp_inputs, comp_outputs, comp_val_inputs, comp_val_outputs

    def _save_loss_plot(
        self,
        block:             Any,
        block_idx:         int,
        train_loss_history: List[float],
        val_loss_history:   List[float],
        is_accepted:       bool,
        plots_dir:         str,
        plot_label:        str = None,
    ) -> None:
        """
        Render and save a train/val loss curve PNG for one distillation block.
        File name is derived from plot_label (captured before rollback).
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        op_label  = plot_label or f'block_{block_idx}'
        safe_name = op_label.replace('/', '_').replace(' ', '').strip('_')
        status    = 'ACCEPTED' if is_accepted else 'REJECTED'
        title_color = '#2ecc71' if is_accepted else '#e74c3c'

        fig, ax = plt.subplots(figsize=(10, 5))
        steps_axis = list(range(1, len(train_loss_history) + 1))

        ax.plot(steps_axis, train_loss_history,
                label='Train Loss', color='#3498db', linewidth=1.5, alpha=0.7)
        if val_loss_history:
            # Val checkpoints are spaced evenly across the step range
            n_val = len(val_loss_history)
            val_step_size = max(1, len(train_loss_history) // max(1, n_val))
            val_steps = [val_step_size * (i + 1) for i in range(n_val)]
            ax.plot(val_steps, val_loss_history,
                    label='Val Loss', color='#e67e22', linewidth=2,
                    linestyle='--', marker='o', markersize=4)

        ax.set_title(
            f'Distillation Loss  |  {op_label}  |  [{status}]',
            fontsize=13, fontweight='bold', color=title_color,
        )
        ax.set_xlabel('Step', fontsize=11)
        ax.set_ylabel('Loss',  fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        plot_path = os.path.join(plots_dir, f'{safe_name}.png')
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
        print(f"       [Plot] Saved \u2192 {os.path.relpath(plot_path)}")

    def _initialize_executors(self, graph: BaseGraph, fp32_graph_clone: BaseGraph, dataloader: list, executing_device: str):
        print(f"  ---> Initializing executors on [{executing_device.upper()}]...")
        if not dataloader or len(dataloader) == 0:
            raise ValueError("[Engine Error] Calibration dataloader is empty!")
            
        dummy_input = dataloader[0].to(executing_device)
        student_executor = TorchExecutor(graph=graph, device=executing_device)
        student_executor.tracing_operation_meta(inputs=dummy_input)
        
        teacher_executor = TorchExecutor(graph=fp32_graph_clone, device=executing_device)
        teacher_executor.tracing_operation_meta(inputs=dummy_input)
        
        return student_executor, teacher_executor

    def _generate_blocks(self, graph: BaseGraph, student_executor: TorchExecutor,
                          target_op_types: list, block_size: int,
                          strategy: BaseReplacementStrategy):
        print(f"  ---> Building topology blocks...")
        blocks = []
        visited_ops = set()
        block_builder = BlockBuilder(graph=graph, topo_order=student_executor._executing_order)

        for op in graph.operations.values():
            if op in visited_ops:
                continue
            if op.is_computing_op or (op.type in target_op_types):
                block = block_builder.build(op, limit=block_size)
                for internal_op in block.rps:
                    visited_ops.add(internal_op)

                # Block extension: if the strategy requires the predecessor
                # to be in the block, check if the endpoint feeds a target op.
                # If so, extend the block by one to include it.
                if strategy.requires_predecessor:
                    for successor in block.ep.outputs[0].dest_ops:
                        if successor not in visited_ops and strategy.select_target(successor, block, graph):
                            successor.attributes['_is_target'] = True  # cache result for Phase A
                            block.rps.append(successor)
                            block.ep = successor
                            visited_ops.add(successor)

                blocks.append(block)
        return blocks

    def _collect_block_cache(
        self, block, cache_name: str, required_samples: int,
        dataloader: list,
        student_executor: TorchExecutor,
        teacher_executor: TorchExecutor,
        teacher_graph:    BaseGraph,
        executing_device: str,
        caching_device:   str,
    ):
        input_wire_name   = block.sp.inputs[0].name
        student_out_name  = block.ep.outputs[0].name

        # Teacher wire: resolve original wire from FP32 clone via _parent_op stamp
        ep_parent_name = block.ep.attributes.get('_parent_op', None)
        if ep_parent_name and ep_parent_name in teacher_graph.operations:
            teacher_ep_op   = teacher_graph.operations[ep_parent_name]
            teacher_out_name = teacher_ep_op.outputs[0].name
        elif block.ep.name in teacher_graph.operations:
            # Rolled-back op: no _parent_op, but same name as original
            teacher_ep_op   = teacher_graph.operations[block.ep.name]
            teacher_out_name = teacher_ep_op.outputs[0].name
        else:
            teacher_out_name = student_out_name

        student_input_cache  = []
        teacher_output_cache = []

        pbar = tqdm(
            dataloader, desc=cache_name, leave=True,
            dynamic_ncols=True, total=max(1, required_samples),
        )

        samples_collected = 0
        for batch in pbar:
            if samples_collected >= required_samples:
                break

            batch = batch.to(executing_device)
            teacher_batch = teacher_executor.forward(
                [batch], output_names=[teacher_out_name]
            )[0].to(caching_device)
            student_batch = student_executor.forward(
                [batch], output_names=[input_wire_name]
            )[0].to(caching_device)

            teacher_output_cache.append(teacher_batch)
            student_input_cache.append(student_batch)
            samples_collected += 1

        pbar.close()
        return student_input_cache, teacher_output_cache, input_wire_name, student_out_name

    # ─────────────────────────────────────────────────────────────────────
    # BLOCK LOGGING
    # ─────────────────────────────────────────────────────────────────────

    def _write_block_log(self, block_log, log_dir, strategy):
        """Write detailed per-block accept/reject log to auto-indexed file."""
        os.makedirs(log_dir, exist_ok=True)
        prefix = "morph_log"
        existing = [f for f in os.listdir(log_dir) if f.startswith(prefix) and f.endswith('.txt')]
        run_idx = len(existing) + 1
        log_path = os.path.join(log_dir, f"{prefix}_{run_idx:02d}.txt")

        with open(log_path, 'w') as lf:
            lf.write(f"Neural Morphing Run #{run_idx:02d}\n")
            lf.write(f"Strategy: {type(strategy).__name__}\n")
            lf.write(f"  lr={strategy.learning_rate}, steps={strategy.steps}, "
                     f"patience={strategy.patience}\n")
            if hasattr(strategy, '_min_cosine'):
                lf.write(f"  min_cosine={strategy._min_cosine}, "
                         f"scale_bounds={strategy._scale_bounds}\n")
            lf.write("=" * 80 + "\n\n")

            accepted_count = 0
            rejected_count = 0
            for entry in block_log:
                status = "ACCEPTED" if entry['accepted'] else "REJECTED"
                if entry['accepted']:
                    accepted_count += 1
                else:
                    rejected_count += 1

                lf.write(f"Block {entry['block_idx']}/{entry['total_blocks']}: {entry['block_range']}\n")
                lf.write(f"  Status:       {status}\n")
                lf.write(f"  Replaced ops: {entry['replaced_ops']}\n")
                lf.write(f"  Loss:         {entry.get('starting_loss', 'N/A')} -> {entry.get('final_loss', 'N/A')}\n")
                lf.write(f"  Duration:     {entry.get('duration_s', 0):.1f}s\n")
                if 'mean_cosine' in entry:
                    lf.write(f"  Cosine:       {entry['mean_cosine']:.6f}  "
                             f"(req >= {entry['min_cosine_req']})  "
                             f"{'PASS' if entry['cosine_pass'] else 'FAIL'}\n")
                    lf.write(f"  Scale:        {entry['mean_scale']:.6f}  "
                             f"(req {entry['scale_bounds']})  "
                             f"{'PASS' if entry['scale_pass'] else 'FAIL'}\n")
                lf.write("\n")

            lf.write("=" * 80 + "\n")
            lf.write(f"Summary: {accepted_count} accepted, {rejected_count} rejected\n")

        print(f"  ---> Block log saved -> {log_path}")
        self._last_log_path = log_path

    # ─────────────────────────────────────────────────────────────────────
    # MAIN LOOP
    # ─────────────────────────────────────────────────────────────────────

    def optimize(self, graph: BaseGraph, **kwargs) -> None:


        cfg = self._parse_arguments(kwargs)

        # Local aliases for frequently-used config fields (readability)
        strategy         = cfg.strategy
        dataloader       = cfg.dataloader
        val_dataloader   = cfg.val_dataloader
        executing_device = cfg.executing_device
        caching_device   = cfg.caching_device
        plots_dir        = cfg.plots_dir

        if plots_dir:
            os.makedirs(plots_dir, exist_ok=True)

        print(f"  ---> Cloning graph (teacher snapshot)...")
        fp32_graph_clone = graph.copy(copy_value=True) 
        
        student_executor, teacher_executor = self._initialize_executors(graph, fp32_graph_clone, dataloader, executing_device)
        blocks = self._generate_blocks(graph, student_executor, cfg.target_op_types, cfg.block_size, strategy)
        total_blocks = len(blocks)
        
        print(f"  ---> Starting block-wise distillation on [{executing_device.upper()}]...")
        block_log = []
        for block_idx, block in enumerate(blocks, 1):
            op_names = [op.name for op in block.rps]
            print(f"\n  ┌── Block {block_idx}/{total_blocks}: {block}")
            for name in op_names:
                print(f"  │   {name}")
            print(f"  └──")
            
            # --- PHASE A: SUBGRAPH REPLACEMENT ---
            # Engine iterates ops; strategy only provides select + build.
            replaced = False
            for op in list(block.rps):  # list() copy — block.rps mutates during splice
                is_target = op.attributes.pop('_is_target', False) or strategy.select_target(op, block, graph)
                if is_target:
                    sub_graph, input_shape = strategy.build_replacement_subgraph(
                        op, block, graph, self._module_to_subgraph)
                    self.splice(graph, op, sub_graph, block)
                    replaced = True
            
            if not replaced:
                continue
            
            # splice() added/removed ops — the cached execution order is stale
            student_executor._executing_order = graph.topological_sort()
            student_executor.tracing_operation_meta(inputs=dataloader[0].to(executing_device))

            # --- PHASE B: DATA CACHING ---
            required_train_samples = strategy.calculate_samples(block, block.rps, len(dataloader))
            train_input_cache, train_output_cache, input_wire_name, output_wire_name = self._collect_block_cache(
                block, f"Caching Train: {block_idx}/{total_blocks}", required_train_samples,
                dataloader, student_executor, teacher_executor, fp32_graph_clone,
                executing_device, caching_device,
            )
            
            val_input_cache, val_output_cache = [], []
            if val_dataloader:
                val_input_cache, val_output_cache, _, _ = self._collect_block_cache(
                    block, f"Caching Val: {block_idx}/{total_blocks}", len(val_dataloader),
                    val_dataloader, student_executor, teacher_executor, fp32_graph_clone,
                    executing_device, caching_device,
                )
            
            # --- PHASE C: DISTILLATION TRAINING (step-based, like TQT) ---
            t_start = time.time()
            step_pbar = tqdm(range(strategy.steps), desc=f"Tuning Block {block_idx}/{total_blocks}", leave=True, dynamic_ncols=True)
            criterion = strategy.get_criterion()
            
            trainable_params = []
            for op in block.rps:
                # Only train replacement ops (those with _parent_op attribute).
                # Non-replacement ops (proj, relu...) that share the block must NOT
                # have their weights modified. If the block is rejected and rolled
                # back, only replacement-target weights are restored from the FP32
                # clone. Any other modified weights become permanently corrupted —
                # causing a silent accuracy drop after rollback.
                if '_parent_op' not in op.attributes:
                    continue
                for var in op.inputs:
                    if var.is_parameter and var.value is not None and var.value.is_floating_point():
                        if not var.value.requires_grad: var.value.requires_grad_()
                        trainable_params.append(var.value)
                # Also collect any private params (e.g. _scale_param on HardSiluPie8)
                for p in getattr(op, '_extra_params', []):
                    if not p.requires_grad: p.requires_grad_()
                    trainable_params.append(p)

            # For strategies that require the predecessor in the block
            # (e.g. activation replacement), also train the predecessor's
            # weights so it can compensate for the output distribution shift.
            # This runs unconditionally — even if replacement ops already
            # contributed their own params, the predecessor's weights still
            # need to adapt.
            if strategy.requires_predecessor:
                predecessor_params_added = set()
                for op in block.rps:
                    if '_parent_op' in op.attributes:
                        predecessor = op.inputs[0].source_op
                        if predecessor is not None and predecessor in block.rps and id(predecessor) not in predecessor_params_added:
                            predecessor_params_added.add(id(predecessor))
                            for var in predecessor.inputs:
                                if var.is_parameter and var.value is not None:
                                    if not var.value.requires_grad: var.value.requires_grad_()
                                    if not any(var.value is p for p in trainable_params):
                                        trainable_params.append(var.value)
            
            if not trainable_params:
                step_pbar.close()
                continue
                
            optimizer = torch.optim.AdamW(trainable_params, lr=strategy.learning_rate, weight_decay=strategy.weight_decay) 
            scheduler = strategy.get_scheduler(optimizer, strategy.steps)
            
            best_val_loss = float('inf')
            patience_counter = 0
            starting_loss, final_loss = None, None
            last_val_loss = None
            train_loss_history: List[float] = []
            val_loss_history:   List[float] = []

            # Flat step loop: one batch per step, cycle through cache (like TQT)
            dataset_length = len(train_input_cache)
            eval_every = max(1, dataset_length)  # validate once per full data pass
            
            for step in step_pbar:
                idx = step % dataset_length
                student_input = train_input_cache[idx]
                teacher_output = train_output_cache[idx]

                optimizer.zero_grad()
                feed_dict = {input_wire_name: student_input.to(executing_device)}
                quantized_outputs = student_executor.partial_graph_forward(
                    operations=block.rps, feed_dict=feed_dict, output_names=[output_wire_name]
                )
                loss = criterion(quantized_outputs[0], teacher_output.to(executing_device))
                loss.backward()
                optimizer.step()
                if scheduler: scheduler.step()
                strategy.on_step_end(trainable_params, step)

                train_loss = loss.item()
                train_loss_history.append(train_loss)
                if starting_loss is None: starting_loss = train_loss
                final_loss = train_loss
                
                # Periodic validation (once per full data pass)
                if val_input_cache and (step + 1) % eval_every == 0:
                    val_loss_sum = 0.0
                    with torch.no_grad():
                        for val_in, val_out in zip(val_input_cache, val_output_cache):
                            feed_dict = {input_wire_name: val_in.to(executing_device)}
                            val_preds = student_executor.partial_graph_forward(
                                operations=block.rps, feed_dict=feed_dict, output_names=[output_wire_name]
                            )
                            val_loss_sum += criterion(val_preds[0], val_out.to(executing_device)).item()
                            
                    last_val_loss = val_loss_sum / max(len(val_input_cache), 1)
                    val_loss_history.append(last_val_loss)
                    
                    # Round to 5 decimals so noise-level fluctuations don't reset patience
                    rounded_val = round(last_val_loss, 7)
                    if rounded_val < round(best_val_loss, 7):
                        best_val_loss = last_val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= strategy.patience:
                            print(f"       [Early Stopping] Triggered at Step {step}. Halting.")
                            break

                # Always show last known val loss in progress bar
                postfix = {"T_Loss": f"{train_loss:.4f}"}
                if last_val_loss is not None:
                    postfix["V_Loss"] = f"{last_val_loss:.4f}"
                    postfix["P"] = f"{patience_counter}/{strategy.patience}"
                step_pbar.set_postfix(postfix)
            
            step_pbar.close()
            duration_s = time.time() - t_start
            
            # --- PHASE D: QUALITY GATE & ROLLBACK ---
            eval_inputs  = val_input_cache  if val_input_cache  else train_input_cache
            eval_outputs = val_output_cache if val_output_cache else train_output_cache

            eval_result = strategy.evaluate_validation(
                student_executor, block.rps, input_wire_name, output_wire_name,
                eval_inputs, eval_outputs, executing_device,
            )

            # Support both old (bool) and new (bool, dict) return formats
            if isinstance(eval_result, tuple):
                is_accepted, metrics = eval_result
            else:
                is_accepted, metrics = eval_result, {}

            # Build block log entry
            replaced_ops = [op.name for op in block.rps if '_parent_op' in op.attributes]
            # Capture plot label NOW (before potential rollback deletes _parent_op)
            parent_names = sorted(set(
                op.attributes['_parent_op']
                for op in block.rps
                if '_parent_op' in op.attributes
            ))
            plot_label = ' + '.join(parent_names) if parent_names else f'block_{block_idx}'
            block_entry = {
                'block_idx':     block_idx,
                'total_blocks':  total_blocks,
                'block_range':   str(block),
                'replaced_ops':  replaced_ops,
                'accepted':      is_accepted,
                'starting_loss': starting_loss,
                'final_loss':    final_loss,
                'duration_s':    duration_s,
                **metrics,
            }
            block_log.append(block_entry)

            if not is_accepted:
                reason_parts = []
                if metrics.get('cosine_pass') is False:
                    reason_parts.append(f"cosine={metrics['mean_cosine']:.4f} < {metrics['min_cosine_req']}")
                if metrics.get('scale_pass') is False:
                    reason_parts.append(f"scale={metrics['mean_scale']:.4f} out of {metrics['scale_bounds']}")
                reason = ", ".join(reason_parts) if reason_parts else "quality gate failed"
                print(f"  ---> Block REJECTED ({reason}) — rolling back.")
                self.rollback(graph, fp32_graph_clone, block)
                # Rollback mutated the graph — re-sync before next block
                student_executor._executing_order = graph.topological_sort()
                student_executor.tracing_operation_meta(inputs=dataloader[0].to(executing_device))

                # --- PHASE D2: DRIFT COMPENSATION (optional) ---
                if strategy.compensate_on_reject:
                    self._compensate_rejected_block(
                        graph, block, block_idx, total_blocks, strategy,
                        dataloader, val_dataloader,
                        student_executor, teacher_executor,
                        fp32_graph_clone, executing_device, caching_device,
                    )
            else:
                cos_str = f"cos={metrics.get('mean_cosine', 0):.4f}" if metrics else ""
                scale_str = f"scale={metrics.get('mean_scale', 0):.4f}" if metrics else ""
                print(f"  ---> Block ACCEPTED. Loss: {starting_loss:.4f} -> {final_loss:.4f} | {cos_str} {scale_str}")

            # --- PHASE E: LOSS PLOT ---
            if plots_dir and train_loss_history:
                self._save_loss_plot(
                    block, block_idx,
                    train_loss_history, val_loss_history,
                    is_accepted, plots_dir, plot_label,
                )

            del train_loss_history, val_loss_history
            del train_input_cache, train_output_cache, val_input_cache, val_output_cache

        print("  ---> Subgraph optimization complete.")

        # Write block log if log_dir is specified
        log_dir = cfg.plots_dir  # reuse plots_dir for log output
        if log_dir and block_log:
            self._write_block_log(block_log, log_dir, strategy)

