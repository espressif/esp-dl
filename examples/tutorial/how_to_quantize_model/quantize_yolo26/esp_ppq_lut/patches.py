
import torch
from esp_ppq.IR import Variable, Operation
from esp_ppq.IR.quantize import QuantableOperation
from esp_ppq.executor.base import OPERATION_FORWARD_TABLE
from esp_ppq.executor.op.torch.default import DEFAULT_BACKEND_TABLE, ASSERT_NUM_OF_INPUT
from esp_ppq.parser.espdl.espdl_typedef import ExporterPatternInfo
from esp_ppq.parser.espdl.export_patterns import AddLUTPattern
from esp_ppq import PPQLinearQuant_toInt
from esp_ppq.utils.round import ppq_tensor_round
import torch.nn.functional as F
from typing import List

def Sigmoid_forward(op: Operation, values: List[torch.Tensor], **kwargs) -> torch.Tensor:
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    [input_value] = values
    return torch.sigmoid(input_value)

def Tanh_forward(op: Operation, values: List[torch.Tensor], **kwargs) -> torch.Tensor:
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    [input_value] = values
    return torch.tanh(input_value)

def Relu_forward(op: Operation, values: List[torch.Tensor], **kwargs) -> torch.Tensor:
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    [input_value] = values
    return F.relu(input_value)

def patch_esp_ppq_library(verbose=False):
    """
    Registers activation forwarders to enable LUT simulation.
    Note: We no longer monkey-patch the exporter's AddLUTPattern directly 
    to avoid side-effects during final deployment.
    """
    # 2. Patch DEFAULT_BACKEND_TABLE to bypass UnaryEltwise type-checking
    # This ensures LUT simulation can always find the correct math truth
    DEFAULT_BACKEND_TABLE['Sigmoid'] = Sigmoid_forward
    DEFAULT_BACKEND_TABLE['Tanh']    = Tanh_forward
    DEFAULT_BACKEND_TABLE['Relu']    = Relu_forward
    
    if verbose:
        print("[ESP-PPQ-LUT] Activation forwarders registered for simulation.")
