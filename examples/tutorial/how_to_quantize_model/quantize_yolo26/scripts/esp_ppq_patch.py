# ==============================================================================
# ESP-PPQ RUNTIME PATCHES
# ==============================================================================
import torch
import onnx
from onnx import helper, numpy_helper
from typing import List
# Fix Import Path: TorchBackendContext is in executor.op.torch.base
from esp_ppq.executor.op.torch.base import (
    TorchBackendContext, 
    ASSERT_NUM_OF_INPUT, 
    VALUE_TO_EXECUTING_DEVICE
)
from esp_ppq.IR import Operation
from esp_ppq.parser.onnx_parser import OnnxParser
from esp_ppq.executor.op import DEFAULT_BACKEND_TABLE
from esp_ppq.log import NaiveLogger
logger = NaiveLogger.get_logger('PPQ')

def apply_esp_ppq_patches():
    """
    Applies runtime monkey patches to ESP-PPQ to fix compatibility issues
    with newer versions of ONNX (>1.13) and PyTorch (multidimensional indexing).
    """
    print("Applying ESP-PPQ Runtime Patches...")
    # --------------------------------------------------------------------------
    # 1. Patch OnnxParser.refine_graph
    # --------------------------------------------------------------------------
    def patched_refine_graph(self, graph):
        for op in graph.operations.values():
            for key, value in op.attributes.items():
                if isinstance(value, bytes):
                    value = value.decode('utf-8')
                if op.type == 'Constant' or op.type == 'ConstantOfShape':
                    value = numpy_helper.to_array(value).copy()
                if op.type == 'Cast' and key == 'to':
                    # Fix for ONNX > 1.13
                    major, minor = map(int, onnx.__version__.split('.')[:2])    # <--- PATCH
                    if (major, minor) >= (1, 13):                               # <--- PATCH
                        from onnx import mapping                                # <--- PATCH
                        value = mapping.TENSOR_TYPE_TO_NP_TYPE[value]           # <--- PATCH
                    else:                                                       # <--- PATCH
                        value = helper.tensor_dtype_to_np_dtype[value]
                op.attributes[key] = value
        graph_initializers = []
        for input_var in graph.inputs.values():
            if input_var.value is not None:
                graph_initializers.append(input_var.name)
        for non_input_var in graph_initializers:
            graph.inputs.pop(non_input_var)
        return graph
    # Apply Patch 1
    OnnxParser.refine_graph = patched_refine_graph
    print("  [x] Patched OnnxParser.refine_graph")
    # --------------------------------------------------------------------------
    # 2. Patch Slice_forward
    # --------------------------------------------------------------------------
    def patched_Slice_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
        ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=3, max_num_of_input=5)
        data, starts, ends = values[:3]
        axes = values[3] if len(values) > 3 else torch.tensor([int(_) for idx, _ in enumerate(starts.tolist())])
        steps = values[4] if len(values) > 4 else torch.ones_like(starts)
        if axes is not None:
            axes = axes.tolist()
        starts, ends, steps = starts.tolist(), ends.tolist(), steps.tolist()
        slices, flip_dims = {}, []
        for start, end, axis, step in zip(starts, ends, axes, steps):
            if step < 0:
                flip_dims.append(axis)
                start, end, step = -start - 1, -end - 1, -step
            slices[axis] = slice(int(start), int(end), int(step))
        pos_axes_slices = list(slices.get(a, slice(None, None)) for a in range(max(axes) + 1))
        neg_axes_slices = list(slices.get(a, slice(None, None)) for a in range(min(axes), 0))
        if neg_axes_slices:
            neg_axes_slices = [Ellipsis] + neg_axes_slices
        if flip_dims:
            data = torch.flip(data, dims=flip_dims)
        if pos_axes_slices:
            # Fix: Tuple conversion ensures compatibility with newer PyTorch
            data = data[tuple(pos_axes_slices)]                                 # <--- PATCH
        if neg_axes_slices:
            data = data[neg_axes_slices]
        return data
    # Apply Patch 2
    DEFAULT_BACKEND_TABLE['Slice'] = patched_Slice_forward
    print("  [x] Patched Backend: Slice")
    # --------------------------------------------------------------------------
    # 3. Patch Gather_forward
    # --------------------------------------------------------------------------
    def patched_Gather_forward(op: Operation, values: List[torch.Tensor], ctx: TorchBackendContext = None, **kwargs) -> torch.Tensor:
        values = VALUE_TO_EXECUTING_DEVICE(op=op, ctx=ctx, values=values)
        input_data, indices = values
        indices = indices.long()
        axis = op.attributes.get('axis', 0)
        if op.type == 'Gather':
            array_idx = [indices if axis == i else slice(dim) for i, dim in enumerate(input_data.shape)]
            # Fix: Tuple conversion ensures compatibility with newer PyTorch
            output = input_data[tuple(array_idx)]                               # <--- PATCH
        elif op.type == 'GatherElements':
            output = torch.gather(input_data, axis, indices)
        else:
            logger.warning('Not Gather op, return input as output')
            output = values
        return output
    # Apply Patch 3
    DEFAULT_BACKEND_TABLE['Gather'] = patched_Gather_forward
    print("  [x] Patched Backend: Gather")
    print("ESP-PPQ Runtime Patches Applied Successfully.")
