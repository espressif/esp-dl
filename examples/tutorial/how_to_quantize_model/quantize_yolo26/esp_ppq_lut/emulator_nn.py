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
    Bit-Exact Nearest-Neighbor INT16 LUT Emulator.

    Mirrors the ESP32-P4 PIE8 SIMD kernel (dl_esp32p4_s16_lut_pie8.S):
        esp.xorq     q0, q0, q7     // signed → unsigned: XOR 0x8000
        esp.vmul.u16 q0, q0, q5     // q0 = round_half_even(q0 / 2^SAR)
        // scalar table lookup per element

    No interpolation — direct nearest-neighbor table lookup.

    Bit-exactness guaranteed because:
      1. PIE uses HALF_EVEN rounding (PIE CFG register default)
      2. PyTorch torch.round() uses HALF_EVEN (IEEE 754 default)
      3. step is always power-of-2, so uint16/step is exact in float32
    """

    # Cache: maps op_id -> int16 table tensor
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
        n_entries = 65536 // step + 1

        # Generate the input integer indices for table pivots
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
    def forward(
        ctx, input_tensor, math_fn, op_context, in_scale, out_scale, step, rounding
    ):
        # Save for backward
        ctx.math_fn = math_fn
        ctx.op_context = op_context
        ctx.save_for_backward(input_tensor)

        # --- Step 1: Quantize input to INT16 ---
        if isinstance(in_scale, torch.Tensor) and in_scale.ndim > 0:
            in_scale_bc = (
                in_scale.view(1, -1, 1, 1) if input_tensor.ndim == 4 else in_scale
            )
        else:
            in_scale_bc = in_scale

        # Use float64 for the division to avoid float32 rounding amplification
        input_int = ppq_tensor_round(
            input_tensor.double() / in_scale_bc.double(), rounding
        )
        input_int = torch.clamp(input_int, -32768, 32767).to(torch.int32)

        # --- Step 2: Build or retrieve the LUT table ---
        cache_key = id(op_context)
        if cache_key not in HardwareEmulator._table_cache:
            table = HardwareEmulator._build_table(
                math_fn, op_context, in_scale, out_scale, step, rounding
            )
            HardwareEmulator._table_cache[cache_key] = table
        table = HardwareEmulator._table_cache[cache_key]
        table = table.to(input_int.device)

        # --- Step 3: Nearest-neighbor lookup (mirrors PIE8 kernel exactly) ---
        # PIE kernel: esp.xorq q0, q0, q7  →  signed_to_unsigned
        idx = input_int + 32768  # [0, 65535], matches XOR 0x8000

        # PIE kernel: esp.vmul.u16 q0, q0, q5  →  round_half_even(val / step)
        # step is power-of-2, so float32 division is exact (no rounding error)
        nn_idx = torch.round(idx.float() / step).to(torch.int32)

        # Clamp to valid table range
        nn_idx = torch.clamp(nn_idx, 0, table.shape[0] - 1)

        # Direct table lookup — no interpolation
        orig_shape = nn_idx.shape
        output_int = table[nn_idx.flatten().long()].view(orig_shape)
        output_int = torch.clamp(output_int, -32768, 32767)

        # --- Step 4: Dequantize back to float for pipeline ---
        if isinstance(out_scale, torch.Tensor) and out_scale.ndim > 0:
            out_scale_bc = (
                out_scale.view(1, -1, 1, 1) if input_tensor.ndim == 4 else out_scale
            )
        else:
            out_scale_bc = out_scale

        return output_int.float() * out_scale_bc

    @staticmethod
    def backward(ctx, grad_output):
        (input_tensor,) = ctx.saved_tensors
        math_fn = ctx.math_fn
        op_context = ctx.op_context

        with torch.enable_grad():
            x = input_tensor.detach().requires_grad_(True)
            y = math_fn(op_context, [x])
            grad = torch.autograd.grad(y.sum(), x)[0]

        return grad_output * grad, None, None, None, None, None, None


def lut_forward_provider(op, values, ctx=None, **kwargs):
    """
    The 'Universal LUT Execution Engine' — Nearest-Neighbor version.
    Switches between Ideal Math (for export) and Bit-Exact Simulation (for everything else).
    """
    input_tensor = values[0]

    # 1. Strict Attribute Verification Loop
    mandatory_attrs = ["original_op_type", "int16_lut_step"]
    for attr in mandatory_attrs:
        if attr not in op.attributes:
            print(
                f"\033[91m[CRITICAL ERROR] Mandatory attribute '{attr}' missing for Operation: {op.name}\033[0m"
            )
            raise AttributeError(
                f"Required attribute '{attr}' is missing. Check fusion pipeline."
            )

    original_type = op.attributes["original_op_type"]
    step = op.attributes["int16_lut_step"]

    # 2. Find the Mathematical Ideal (The 'Truth')
    if original_type not in DEFAULT_BACKEND_TABLE:
        print(
            f"\033[91m[CRITICAL ERROR] Math implementation for '{original_type}' not found in DEFAULT_BACKEND_TABLE.\033[0m"
        )
        raise KeyError(f"Mathematical ground truth for {original_type} is missing.")
    ideal_math_fn = DEFAULT_BACKEND_TABLE[original_type]

    if ideal_math_fn is None:
        raise KeyError(
            f"Could not find ideal math function for {original_type} in DEFAULT_BACKEND_TABLE."
        )

    # 3. Path A: IDEAL MODE (For Table Generation during Export)
    if GlobalMode.get() == SimulationMode.IDEAL_MATH:
        return ideal_math_fn(op, values)

    # 4. Path B: SIMULATION MODE (Default: For Validation, PTQ, and STE)
    in_scale = op.input_quant_config[0].scale
    out_scale = op.output_quant_config[0].scale
    rounding = op.input_quant_config[0].rounding

    return HardwareEmulator.apply(
        input_tensor, ideal_math_fn, op, in_scale, out_scale, step, rounding
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
        OPERATION_FORWARD_TABLE[platform]["LUT"] = lut_forward_provider

    if verbose:
        print("[ESPDL Emulator] LUT Operation Handler Registered Globally (Nearest-Neighbor).")
