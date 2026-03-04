
import torch
import torch.nn as nn
from enum import Enum
from typing import Any
from esp_ppq.utils.round import ppq_tensor_round
from esp_ppq.core import RoundingPolicy, TargetPlatform
from esp_ppq.executor.base import OPERATION_FORWARD_TABLE
from esp_ppq.executor.op.torch.default import DEFAULT_BACKEND_TABLE
from esp_ppq.parser.espdl.espdl_typedef import ACTIVATION_OP_SET, PASSIVE_LAYOUT_OP_SET

class SimulationMode(Enum):
    SIMULATION = 1
    IDEAL_MATH = 2

def set_simulation_mode(mode: SimulationMode):
    """Convenience helper to set the global emulator mode."""
    GlobalMode.set(mode)

class GlobalMode:
    """Manages the current mode of the ESP-DL emulator."""
    _current_mode = SimulationMode.SIMULATION

    @classmethod
    def set(cls, mode: SimulationMode):
        cls._current_mode = mode

    @classmethod
    def get(cls) -> SimulationMode:
        return cls._current_mode


class HardwareEmulator(torch.autograd.Function):
    """
    Bit-Exact Integer LUT Emulator — mirrors ESP-DL dl_module_lut.hpp exactly.
    
    The hardware C code (dl_module_lut.hpp:72-81):
        int idx = input_ptr[i] + 32768;
        int len = idx % step;
        idx = idx / step;
        int x = table_ptr[idx];
        int y = table_ptr[idx + 1];
        output_ptr[i] = x + len * (y - x) / step;
    
    ALL arithmetic is pure integer (C int division truncates toward zero).
    This emulator reproduces that exactly using torch.int32 tensors.
    """
    # Cache: maps (op_name, step) -> int16 table tensor
    _table_cache = {}

    @staticmethod
    def _build_table(math_fn, op_context, in_scale, out_scale, step, rounding):
        """
        Build the INT16 LUT table exactly as the esp-ppq exporter does.
        Matches export_patterns.py calculate_lut():
            input = torch.arange(min, max+1, step=step) * in_scale
            output = operation_forward_func(op, [input])
            lut = ppq_tensor_round(output / out_scale, rounding).clamp().to(int16)
        """
        # For INT16: range is [-32768, 32767], table indices are [0, 65536/step]
        # The table has (65536 // step + 1) entries
        n_entries = 65536 // step + 1
        
        # Generate the input integer indices for table pivots
        # These are the INT16 values at each table entry
        table_input_int = torch.arange(0, n_entries, dtype=torch.float32) * step - 32768
        
        # Convert to real-world float values
        if isinstance(in_scale, torch.Tensor):
            s = in_scale.flatten()[0].item()
        else:
            s = float(in_scale)
        table_input_float = table_input_int * s
        
        # Compute the ideal math function output
        table_output_float = math_fn(op_context, [table_input_float])
        
        # Quantize to INT16 exactly as the exporter does
        if isinstance(out_scale, torch.Tensor):
            os = out_scale.flatten()[0].item()
        else:
            os = float(out_scale)
        table_int16 = ppq_tensor_round(table_output_float / os, rounding)
        table_int16 = torch.clamp(table_int16, -32768, 32767).to(torch.int32)
        
        return table_int16.flatten()

    @staticmethod
    def forward(ctx, input_tensor, math_fn, op_context, in_scale, out_scale, step, rounding):
        # Save for backward
        ctx.math_fn = math_fn
        ctx.op_context = op_context
        ctx.save_for_backward(input_tensor)

        # --- Step 1: Quantize input to INT16 ---
        if isinstance(in_scale, torch.Tensor) and in_scale.ndim > 0:
            in_scale_bc = in_scale.view(1, -1, 1, 1) if input_tensor.ndim == 4 else in_scale
        else:
            in_scale_bc = in_scale

        # Use float64 for the division to avoid float32 rounding amplification
        # when in_scale_bc is very small (same rationale as the FP64 Conv patch).
        # The result is immediately rounded to int32 so the pipeline stays float32.
        input_int = ppq_tensor_round(input_tensor.double() / in_scale_bc.double(), rounding)
        input_int = torch.clamp(input_int, -32768, 32767).to(torch.int32)

        # --- Step 2: Build or retrieve the LUT table ---
        cache_key = id(op_context)  # unique per operation instance
        if cache_key not in HardwareEmulator._table_cache:
            table = HardwareEmulator._build_table(
                math_fn, op_context, in_scale, out_scale, step, rounding
            )
            HardwareEmulator._table_cache[cache_key] = table
        table = HardwareEmulator._table_cache[cache_key]
        table = table.to(input_int.device)  # ensure same device as input

        # --- Step 3: Pure integer LUT interpolation (mirrors C code exactly) ---
        # C code: int idx = input_ptr[i] + 32768;
        idx = input_int + 32768  # shift to [0, 65535], int32

        # C code: int len = idx % step;
        remainder = idx % step  # int32 modulo

        # C code: idx = idx / step;
        #   C integer division truncates toward zero.
        #   For non-negative idx (always true here since idx ∈ [0,65535]),
        #   Python // is equivalent to C integer division.
        base_idx = idx // step  # int32 floor division (same as C for non-negative)

        # Clamp base_idx to valid table range
        max_idx = table.shape[0] - 2  # -2 because we read table[idx+1]
        base_idx = torch.clamp(base_idx, 0, max_idx)

        # C code: int x = table_ptr[idx];
        #          int y = table_ptr[idx + 1];
        # Flatten for indexing, then reshape back
        orig_shape = base_idx.shape
        base_flat = base_idx.flatten().long()
        x = table[base_flat].to(torch.int32)
        y = table[base_flat + 1].to(torch.int32)

        # C code: output_ptr[i] = x + len * (y - x) / step;
        #   This is ALL integer arithmetic in C.
        #   For non-negative numerator: C division truncates = Python // (floor division)
        #   For negative numerator: C truncates toward zero, Python floors.
        #   We must match C behavior: use (a // b) when a >= 0, else -((-a) // b)
        remainder_flat = remainder.flatten().to(torch.int32)
        delta = y - x  # int32
        numerator = remainder_flat * delta  # int32

        # C-style integer division (truncate toward zero)
        # For non-negative: n // step
        # For negative: -((-n) // step)
        interp = torch.where(
            numerator >= 0,
            numerator // step,
            -((-numerator) // step)
        )

        output_int = x + interp  # int32
        output_int = torch.clamp(output_int, -32768, 32767)
        output_int = output_int.view(orig_shape)

        # --- Step 4: Dequantize back to float for pipeline ---
        if isinstance(out_scale, torch.Tensor) and out_scale.ndim > 0:
            out_scale_bc = out_scale.view(1, -1, 1, 1) if input_tensor.ndim == 4 else out_scale
        else:
            out_scale_bc = out_scale

        return output_int.float() * out_scale_bc

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, = ctx.saved_tensors
        math_fn = ctx.math_fn
        op_context = ctx.op_context

        with torch.enable_grad():
            x = input_tensor.detach().requires_grad_(True)
            y = math_fn(op_context, [x])
            grad = torch.autograd.grad(y.sum(), x)[0]
        
        return grad_output * grad, None, None, None, None, None, None


def lut_forward_provider(op, values, ctx=None, **kwargs):
    """
    The 'Universal LUT Execution Engine'.
    Switches between Ideal Math (for export) and Bit-Exact Simulation (for everything else).
    """
    input_tensor = values[0]
    
    # 1. Strict Attribute Verification Loop
    mandatory_attrs = ['original_op_type', 'int16_lut_step']
    for attr in mandatory_attrs:
        if attr not in op.attributes:
            print(f"\033[91m[CRITICAL ERROR] Mandatory attribute '{attr}' missing for Operation: {op.name}\033[0m")
            raise AttributeError(f"Required attribute '{attr}' is missing. Check fusion pipeline.")
            
    original_type = op.attributes['original_op_type']
    step          = op.attributes['int16_lut_step']
    
    # 2. Find the Mathematical Ideal (The 'Truth')
    if original_type not in DEFAULT_BACKEND_TABLE:
        print(f"\033[91m[CRITICAL ERROR] Math implementation for '{original_type}' not found in DEFAULT_BACKEND_TABLE.\033[0m")
        raise KeyError(f"Mathematical ground truth for {original_type} is missing.")
    ideal_math_fn = DEFAULT_BACKEND_TABLE[original_type]
            
    if ideal_math_fn is None:
        raise KeyError(f"Could not find ideal math function for {original_type} in DEFAULT_BACKEND_TABLE.")

    # 3. Path A: IDEAL MODE (For Table Generation during Export)
    if GlobalMode.get() == SimulationMode.IDEAL_MATH:
        return ideal_math_fn(op, values)

    # 4. Path B: SIMULATION MODE (Default: For Validation, PTQ, and STE)
    in_scale = op.input_quant_config[0].scale
    out_scale = op.output_quant_config[0].scale
    rounding = op.input_quant_config[0].rounding

    return HardwareEmulator.apply(
        input_tensor, ideal_math_fn, op, 
        in_scale, out_scale, step, rounding
    )

def register_lut_op_handler(verbose=False):
    """
    Registers the LUT operation handler globally.
    This enables hardware-aware simulation for LUT operations.
    """
    ACTIVATION_OP_SET.add("LUT")
    PASSIVE_LAYOUT_OP_SET.add("LUT")
    
    # Inject our provider into ALL platforms in the global forward table
    for platform in OPERATION_FORWARD_TABLE:
        OPERATION_FORWARD_TABLE[platform]['LUT'] = lut_forward_provider
    
    if verbose:
        print("[ESPDL Emulator] LUT Operation Handler Registered Globally.")
