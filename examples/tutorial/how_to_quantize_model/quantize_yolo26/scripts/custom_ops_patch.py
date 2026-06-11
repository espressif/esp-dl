"""
Custom Operations Patch for ESP-PPQ
====================================
Registers custom operations as NEW op types in PPQ without modifying
any existing esp-ppq source files.

Usage:
    import custom_ops_patch
    # Both patches auto-apply on import (see bottom of file)

After importing, the following custom ops are available:

1. TransposePIE  — SIMD-accelerated transpose for ESP32-P4
   - Same forward as existing Transpose (permute dimensions)
   - Auto-enabled via enable_pie_transpose_all() before export
   - ESP-DL side: register dl_module_transpose_pie.hpp in dl_module_creator.hpp

2. HardSiluPie8  — Fused HardSiLU with PIE SIMD kernel
   - Forward: y = x * clamp(x/8 + 0.5, 0, 1)
   - To use: change the op type from "Swish" to "HardSiluPie8"
     in the graph (e.g., via neural morphing strategy).
   - ESP-DL side: register dl_module_hard_silu_pie8.hpp in dl_module_creator.hpp

TransposePIE is passive (quantization pass-through, like Transpose).
HardSiluPie8 is active (computes, like Swish).
"""

import torch
import torch.nn.functional as F
from typing import List

_TRANSPOSE_PIE_APPLIED = False
_HARDSILU_APPLIED = False
_HARDSILU_TRAINING_MODE = False


def enable_hardsilu_training_mode():
    """Force HardSiluPie8 to use float STE path (for TQT gradient flow)."""
    global _HARDSILU_TRAINING_MODE
    _HARDSILU_TRAINING_MODE = True
    print("[custom_ops_patch] HardSiluPie8 training mode: ON")


def disable_hardsilu_training_mode():
    """Revert HardSiluPie8 to bit-exact integer emulation."""
    global _HARDSILU_TRAINING_MODE
    _HARDSILU_TRAINING_MODE = False
    print("[custom_ops_patch] HardSiluPie8 training mode: OFF")


# ===========================================================================
# Forward functions
# ===========================================================================
def _TransposePIE_forward(op, values: List[torch.Tensor], ctx=None, **kwargs):
    """Same forward as Transpose — permute dimensions."""
    from esp_ppq.executor.op.torch.default import (
        ASSERT_NUM_OF_INPUT,
        GET_ATTRIBUTE_FROM_OPERATION,
    )
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    [input_value] = values
    perm = GET_ATTRIBUTE_FROM_OPERATION(op=op, attribute='perm', compulsive=True)
    return input_value.permute(perm)


class _RoundSTE(torch.autograd.Function):
    """Round with straight-through estimator for gradient."""
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class _ClampSTE(torch.autograd.Function):
    """Clamp with straight-through estimator for gradient.
    Gradient always flows, even when the value is outside [min, max].
    Prevents the parameter from getting stuck if the optimizer overshoots."""
    @staticmethod
    def forward(ctx, x, min_val, max_val):
        return torch.clamp(x, min_val, max_val)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


def _HardSiluPie8_forward(op, values: List[torch.Tensor], ctx=None, **kwargs):
    """
    Float: y = x * clamp(x/8 + 0.5, 0, 1) * scale

    Training (no quant configs):  float STE path for gradient flow.
    Inference (quant configs ready): bit-exact integer emulation matching PIE8 assembly.
                             Mirrors dl_esp32p4_s8_hard_silu_pie8.S exactly:
                               xs   = x_q × scale_int          (exact, SAR=0)
                               gate = clamp(x_q + half, 0, max) (exact)
                               y    = xs × gate >> SAR_total    (single HALF_EVEN round)
                               y    = sat(y, -128, 127)
    """
    from esp_ppq.executor.op.torch.default import ASSERT_NUM_OF_INPUT
    ASSERT_NUM_OF_INPUT(op=op, values=values, min_num_of_input=1, max_num_of_input=1)
    [x] = values

    # ── Resolve scale_int (from frozen attribute or live scale_param) ──
    if 'scale_int' in op.attributes:
        scale_int = op.attributes['scale_int']
    elif 'scale_param' in op.attributes:
        sp = op.attributes['scale_param']
        scale_int = int(torch.round(torch.clamp(sp, 0.0, 1.0) * 256.0).item())
    else:
        assert False, f"HardSiluPie8 op '{op.name}': neither scale_param nor scale_int."



    # ── Check if quant configs are ready ──
    has_quant = (hasattr(op, 'input_quant_config')
                 and op.input_quant_config
                 and op.input_quant_config[0].scale is not None)

    # ── Training path: float STE (no quant configs yet) ──
    if not has_quant or _HARDSILU_TRAINING_MODE:
        if 'scale_param' in op.attributes:
            sp = op.attributes['scale_param']
            scale = _RoundSTE.apply(_ClampSTE.apply(sp, 0.0, 1.0) * 256.0) / 256.0
        else:
            scale = scale_int / 256.0
        return x * torch.clamp(x / 8.0 + 0.5, 0, 1) * scale

    # ── Inference path: bit-exact integer emulation ──
    import math
    in_s  = op.input_quant_config[0].scale.item()
    out_s = op.output_quant_config[0].scale.item()
    e1 = int(round(math.log2(in_s)))
    e2 = int(round(math.log2(out_s)))

    # Recover INT8 (exact — PPQ already quantized x, torch.round handles float imprecision)
    x_q = torch.round(x / in_s).to(torch.int32)

    # Hardware constants (mirrors dl_module_hard_silu_pie8.hpp lines 74-76)
    half_val  = 1 << (-e1 + 2)
    clamp_hi  = 1 << (-e1 + 3)
    sar_total = -2 * e1 + 3 + e2 + 8

    # Integer arithmetic (mirrors dl_esp32p4_s8_hard_silu_pie8.S lines 109-116)
    xs   = x_q * scale_int                             # exact, no rounding (SAR=0)
    gate = torch.clamp(x_q + half_val, 0, clamp_hi)    # exact integer clamp

    # Promote to INT64 to avoid overflow (max ~33M > 2^24 float32 mantissa)
    num  = xs.to(torch.int64) * gate.to(torch.int64)   # exact INT64 product

    # HALF_EVEN rounding for >>SAR_total (matches esp.vmul.s16 PIE CFG register)
    # float64 is safe: max |num| ≈ 2^25, float64 mantissa = 53 bits
    y = torch.round(num.double() / (2.0 ** sar_total))
    y = torch.clamp(y, -128, 127)                      # INT8 saturation

    # Dequantize — PPQ's requant wrapper becomes a no-op
    return (y * out_s).float()


# ===========================================================================
# Shared helpers (register a single op into all tables)
# ===========================================================================
def _register_forward(op_name, forward_fn):
    """Register a forward function in all dispatch tables."""
    from esp_ppq.executor.op.torch.default import DEFAULT_BACKEND_TABLE
    from esp_ppq.executor.op.torch.espdl import ESPDL_QUANT_BACKEND_TABLE
    from esp_ppq.executor.base import OPERATION_FORWARD_TABLE

    DEFAULT_BACKEND_TABLE[op_name] = forward_fn
    ESPDL_QUANT_BACKEND_TABLE[op_name] = forward_fn
    for table in OPERATION_FORWARD_TABLE.values():
        if op_name not in table:
            table[op_name] = forward_fn


def _register_socket(op_name):
    """Register socket in DEFAULT_SOCKET_TABLE."""
    from esp_ppq.IR.base.opdef import DEFAULT_SOCKET_TABLE, DEFAULT_SOCKET_CREATOR
    DEFAULT_SOCKET_TABLE[op_name] = DEFAULT_SOCKET_CREATOR


def _register_quantizer(op_name):
    """Monkey-patch BaseEspdlQuantizer.quant_operation_types to include op."""
    from esp_ppq.quantization.quantizer.EspdlQuantizer import BaseEspdlQuantizer

    _orig_prop = BaseEspdlQuantizer.quant_operation_types.fget

    @property
    def _patched(self):
        types = _orig_prop(self)
        types.add(op_name)
        return types

    BaseEspdlQuantizer.quant_operation_types = _patched


# ===========================================================================
# TransposePIE patch
# ===========================================================================
def apply_patch_transpose_pie():
    """
    Register TransposePIE op in PPQ.
    Safe to call multiple times — only applies once.
    """
    global _TRANSPOSE_PIE_APPLIED
    if _TRANSPOSE_PIE_APPLIED:
        return
    _TRANSPOSE_PIE_APPLIED = True

    # Forward table
    _register_forward('TransposePIE', _TransposePIE_forward)

    # Socket
    _register_socket('TransposePIE')

    # Passive operations (quant pass-through, like Transpose)
    from esp_ppq.core import common as _common
    _common.PASSIVE_OPERATIONS.add('TransposePIE')

    # Quantizer
    _register_quantizer('TransposePIE')

    # Layout: same handling as Transpose (restore origin layout)
    from esp_ppq.parser.espdl import espdl_typedef as _td
    _td.OTHER_OP_SET.add('TransposePIE')

    # Exporter: hook for auto-rename Transpose → TransposePIE
    _patch_exporter_op_type_remap()

    print("[custom_ops_patch] Applied: TransposePIE")


# ===========================================================================
# HardSiluPie8 patch
# ===========================================================================
_HARDSILU_EXPORTER_PATCHED = False

def _patch_exporter_hardsilu_freeze():
    """Monkey-patch export_graph to freeze scale_param → scale_int before serialization."""
    global _HARDSILU_EXPORTER_PATCHED
    if _HARDSILU_EXPORTER_PATCHED:
        return
    _HARDSILU_EXPORTER_PATCHED = True

    from esp_ppq.parser import espdl_exporter as _exp
    _prev_export = _exp.EspdlExporter.export_graph

    def _patched_export(self, graph, **kwargs):
        _freeze_hardsilu_scales(graph)
        return _prev_export(self, graph, **kwargs)

    _exp.EspdlExporter.export_graph = _patched_export


def apply_patch_hardsilu():
    """
    Register HardSiluPie8 op in PPQ.
    Safe to call multiple times — only applies once.
    """
    global _HARDSILU_APPLIED
    if _HARDSILU_APPLIED:
        return
    _HARDSILU_APPLIED = True

    # Forward table
    _register_forward('HardSiluPie8', _HardSiluPie8_forward)

    # Socket
    _register_socket('HardSiluPie8')

    # NOT passive — HardSiluPie8 computes, like Swish

    # Quantizer
    _register_quantizer('HardSiluPie8')

    # Layout: elementwise activation, inherits upstream perm
    from esp_ppq.parser.espdl import espdl_typedef as _td
    _td.PASSIVE_LAYOUT_OP_SET.add('HardSiluPie8')
    _td.ACTIVATION_OP_SET.add('HardSiluPie8')

    # Patch exporter to freeze scale_param → scale_int before serialization
    _patch_exporter_hardsilu_freeze()

    print("[custom_ops_patch] Applied: HardSiluPie8")


# ===========================================================================
# TransposePIE exporter hook (auto-rename at export time)
# ===========================================================================
_PIE_TRANSPOSE_ENABLED = False
_PIE_TRANSPOSE_INT8_ONLY = True
_DEBUG_TRANSPOSE_ENABLED = False


def _patch_exporter_op_type_remap():
    """
    Monkey-patch export_graph to:
      1. Auto-rename Transpose -> TransposePIE (if enabled)
      2. Print debug table of all Transpose ops (if enabled)

    Both run AFTER all patterns have injected Transpose ops into the
    export-internal graph copy, but BEFORE flatbuffer serialization.
    """
    from esp_ppq.parser import espdl_exporter as _exp

    _orig_export = _exp.EspdlExporter.export_graph

    def _patched_export(self, graph, **kwargs):
        # Step 1: debug print (before rename, so we see original types)
        if _DEBUG_TRANSPOSE_ENABLED:
            _print_transpose_table(graph)

        # Step 2: auto-rename Transpose → TransposePIE
        if _PIE_TRANSPOSE_ENABLED:
            from esp_ppq.core import TargetPlatform
            int8_platforms = {
                TargetPlatform.ESPDL_INT8,
                TargetPlatform.ESPDL_S3_INT8,
                TargetPlatform.ESPDL_C_INT8,
            }
            count = 0
            for op in graph.topological_sort():
                if op.type == 'Transpose':
                    if _PIE_TRANSPOSE_INT8_ONLY:
                        if hasattr(op, 'platform') and op.platform not in int8_platforms:
                            continue
                    op._type = 'TransposePIE'
                    count += 1
            print(f"[custom_ops_patch] Changed op type: {count} Transpose -> TransposePIE")

        return _orig_export(self, graph, **kwargs)

    _exp.EspdlExporter.export_graph = _patched_export


# ---------------------------------------------------------------------------
# TransposePIE: enable/disable auto-rename
# ---------------------------------------------------------------------------
def enable_pie_transpose_all(int8_only=True):
    """
    Enable automatic Transpose -> TransposePIE rename at export time.
    Call at script start BEFORE export. The actual rename happens inside
    export_graph after all patterns have injected their Transpose ops.

    Usage:
        import custom_ops_patch
        custom_ops_patch.enable_pie_transpose_all(int8_only=True)

        # ... quantize, calibrate, etc ...
        espdl_export(graph, ...)  # rename happens automatically here

    Args:
        int8_only: If True, only rename INT8-targeted Transpose ops (default: True)
    """
    global _PIE_TRANSPOSE_ENABLED, _PIE_TRANSPOSE_INT8_ONLY
    _PIE_TRANSPOSE_ENABLED = True
    _PIE_TRANSPOSE_INT8_ONLY = int8_only
    print(f"[custom_ops_patch] Enabled TransposePIE auto-rename (int8_only={int8_only})")


def disable_pie_transpose_all():
    """Disable automatic Transpose -> TransposePIE rename."""
    global _PIE_TRANSPOSE_ENABLED
    _PIE_TRANSPOSE_ENABLED = False
    print("[custom_ops_patch] Disabled TransposePIE auto-rename")


# ---------------------------------------------------------------------------
# TransposePIE: enable/disable debug table
# ---------------------------------------------------------------------------
def enable_debug_transpose_cases():
    """
    Enable debug printing of all Transpose/TransposePIE ops at export time.
    The table prints inside export_graph on the internal graph copy where
    the Transpose ops actually exist.

    Usage:
        import custom_ops_patch
        custom_ops_patch.enable_debug_transpose_cases()

        # ... quantize, calibrate, export ...
        espdl_export(graph, ...)  # debug table prints automatically here
    """
    global _DEBUG_TRANSPOSE_ENABLED
    _DEBUG_TRANSPOSE_ENABLED = True
    print("[custom_ops_patch] Enabled transpose debug table")


def disable_debug_transpose_cases():
    """Disable transpose debug table."""
    global _DEBUG_TRANSPOSE_ENABLED
    _DEBUG_TRANSPOSE_ENABLED = False
    print("[custom_ops_patch] Disabled transpose debug table")


def _print_transpose_table(graph):
    """Internal: print the debug table (called from inside _patched_export)."""
    pass  # Silenced for clean export output


def _simulate_dispatch(shape, perm):
    """Simulate the 6-step C++ dispatch algorithm in Python. Returns description string."""
    ndims = len(perm)
    if ndims < 2 or ndims > 8:
        return "scalar (ndims)"

    # Normalize
    for i in range(ndims):
        if perm[i] < 0:
            perm[i] += ndims

    # Peel leading batch
    batch = 1
    lo = 0
    while lo < ndims and perm[lo] == lo:
        batch *= shape[lo]
        lo += 1

    # Peel trailing fixed
    K = 1
    hi = ndims - 1
    while hi >= lo and perm[hi] == hi:
        K *= shape[hi]
        hi -= 1

    if lo > hi:
        return "identity (no-op)"

    # Group consecutive ascending
    act = perm[lo:hi+1]
    groups = []
    cur_start = act[0]
    cur_len = 1
    for k in range(1, len(act)):
        if act[k] == act[k-1] + 1:
            cur_len += 1
        else:
            groups.append((cur_start, cur_len))
            cur_start = act[k]
            cur_len = 1
    groups.append((cur_start, cur_len))

    if len(groups) != 2:
        return f"scalar ({len(groups)} groups)"

    # Compute sizes
    g0_start, g0_len = groups[0]
    g1_start, g1_len = groups[1]
    s0 = 1
    for k in range(g0_len):
        s0 *= shape[g0_start + k]
    s1 = 1
    for k in range(g1_len):
        s1 *= shape[g1_start + k]

    if g0_start < g1_start:
        N, M = s0, s1
    else:
        N, M = s1, s0

    if K == 1:
        if N % 8 == 0 and M % 16 == 0:
            return f"K1_zip  batch={batch} N={N} M={M} K=1"
        else:
            return f"scalar (K=1 align: N%8={N%8} M%16={M%16})"
    elif K >= 16 and K % 16 == 0:
        return f"K2_blk  batch={batch} N={N} M={M} K={K}"
    else:
        return f"scalar (K={K} not 16-aligned)"


# ===========================================================================
# HardSiluPie8 scale freeze (export-time)
# ===========================================================================
def _freeze_hardsilu_scales(graph):
    """
    Convert HardSiluPie8 scale_param (learnable tensor) → scale_int (int attribute).
    Called at export time, before FlatBuffer serialization.
    Skips ops that are already frozen (have scale_int, no scale_param).
    """
    count = 0
    for op in graph.topological_sort():
        if op.type != 'HardSiluPie8':
            continue
        if 'scale_param' not in op.attributes:
            assert 'scale_int' in op.attributes, \
                f"HardSiluPie8 op '{op.name}' has neither scale_param nor scale_int."
            continue  # already frozen
        s_int = int(round(torch.clamp(op.attributes['scale_param'], 0.0, 1.0).item() * 256))
        op.attributes['scale_int'] = s_int
        del op.attributes['scale_param']
        count += 1

    print(f"[custom_ops_patch] HardSiluPie8: froze {count} learned scales")


# ===========================================================================
# TiledConvBlock fusion patch
# ===========================================================================
_TILED_CONV_APPLIED = False
_TILED_CONV_ENABLED = False

# ESP32-P4 L2 cache: 256KB. Budget per output tile = L2/8 = 32KB.
_L2_TILE_BUDGET = 256 * 1024 // 8  # 32768 bytes

# Supported act types for fusion (must match C++ apply_act dispatch).
_FUSABLE_ACT_TYPES = {'HardSiluPie8', 'Swish', 'LUT'}


def apply_patch_tiled_conv_block():
    """
    Register TiledConvBlock fusion pass in PPQ.
    The pass runs inside export_graph, just before FlatBuffer serialization.
    It detects Conv → Act patterns and fuses them into a single TiledConvBlock node.

    No forward/socket/quantizer registration needed — TiledConvBlock never runs
    in PPQ's executor. It only exists in the exported .espdl model.

    Safe to call multiple times — only applies once.
    """
    global _TILED_CONV_APPLIED
    if _TILED_CONV_APPLIED:
        return
    _TILED_CONV_APPLIED = True

    # Add TiledConvBlock to the layout sets so the exporter doesn't complain
    from esp_ppq.parser.espdl import espdl_typedef as _td
    _td.CONV_LAYOUT_OP_SET.add('TiledConvBlock')

    # Monkey-patch export_graph to inject the fusion pass
    _patch_exporter_tiled_conv()

    print("[custom_ops_patch] Applied: TiledConvBlock fusion")


def _patch_exporter_tiled_conv():
    """
    Monkey-patch export_graph to run Conv+Act → TiledConvBlock fusion
    AFTER prepare_graph (so shapes are NHWC and LUTs exist), then update
    the ExporterPatternInfo singleton's var_exponents to reflect the new
    output quantization config.

    Timeline:
      prepare_graph → QuantVariableToIntPattern bakes exponents
      export_graph  → OUR FUSION runs here, then we fix the exponents
    """
    from esp_ppq.parser import espdl_exporter as _exp

    _prev_export = _exp.EspdlExporter.export_graph

    def _patched_export(self, graph, **kwargs):
        if _TILED_CONV_ENABLED:
            _fuse_conv_act_to_tiled_block(graph)
        return _prev_export(self, graph, **kwargs)

    _exp.EspdlExporter.export_graph = _patched_export


def _calculate_tile_h(H_out, W_out, C_out, elem_size):
    """
    Compute tile_h so that the output tile = tile_h × W_out × C_out × elem_size ≤ L2 budget.
    Returns tile_h clamped to [1, H_out].
    """
    row_bytes = W_out * C_out * elem_size
    if row_bytes == 0:
        return H_out
    tile_h = _L2_TILE_BUDGET // row_bytes
    return max(1, min(tile_h, H_out))


# =========================================================================
# Channel tiling: compute c_tile from L2 budget
# =========================================================================
def _calculate_c_tile(kH, kW, C_in, C_out, elem_size):
    """
    Compute c_tile (output channels per tile) so the filter working set
    fits in L2.  Budget: kH × kW × C_in × c_tile × elem_size ≤ _L2_TILE_BUDGET.
    c_tile is rounded down to vector_width (16 for s8, 8 for s16).
    Returns C_out if the full filter already fits (no channel tiling needed).
    """
    vector_width = 16 if elem_size == 1 else 8
    filter_per_ch = kH * kW * C_in * elem_size
    if filter_per_ch == 0:
        return C_out
    c_tile = _L2_TILE_BUDGET // filter_per_ch
    c_tile = (c_tile // vector_width) * vector_width   # align down
    c_tile = max(vector_width, min(c_tile, C_out))      # clamp
    return c_tile
# =========================================================================


def _detect_fusion_chain(graph, conv_op):
    """
    Walk downstream from a Conv op and collect fusable ops into an ordered chain.

    Returns [Conv] (Conv-only) or [Conv, Act] if a fusable activation follows.
    Returns None if Conv has multiple consumers (can't fuse).
    """
    chain = [conv_op]

    # Look for optional Act after Conv
    downs = graph.get_downstream_operations(conv_op)
    if len(downs) != 1:
        return None  # Conv has multiple consumers — can't fuse
    next_op = downs[0]
    if next_op.type in _FUSABLE_ACT_TYPES:
        if len(next_op.outputs[0].dest_ops) <= 1:
            chain.append(next_op)
        # If Act has multiple consumers, still fuse Conv-only

    return chain


def _collect_chain_attributes(chain, tile_h, info, elem_size):
    """
    Iterate the chain and collect attributes by role.
    General for any chain length.

    Returns a dict of attributes to set on the TiledConvBlock op.
    """
    attrs = {'tile_h': tile_h}
    act_idx = 0

    # ── Channel tiling ──────────────────────────────────
    # Filter is already HWIO [kH, kW, C_in, C_out] at fusion time
    # (ResetParamLayoutPattern runs before our fusion in the export chain).
    # After export, layout is (N/16)HWC16 → groups of 16 C_out are contiguous.
    filter_var = chain[0].inputs[1]
    fshape = filter_var.value.shape
    kH = fshape[0]
    kW = fshape[1] if len(fshape) >= 2 else 1
    C_in = fshape[2] if len(fshape) >= 3 else 1
    C_out = fshape[3] if len(fshape) >= 4 else fshape[0]
    c_tile = _calculate_c_tile(kH, kW, C_in, C_out, elem_size)
    if c_tile < C_out:
        attrs['c_tile'] = c_tile

    # Conv output exponent — needed for both Conv-only and Conv+Act chains
    conv_out_var = chain[0].outputs[0]
    conv_out_exponents = info.get_var_exponents(conv_out_var.name)
    if conv_out_exponents:
        conv_exp = conv_out_exponents[0] if isinstance(conv_out_exponents, list) else conv_out_exponents
    else:
        conv_exp = 0

    # Default: Conv-only (no activation)
    attrs['act1_input_exponent'] = conv_exp

    for i, op in enumerate(chain):
        if i == 0:
            # Primary Conv: no extra attributes needed (Conv attrs are already on the op)
            continue

        if op.type in _FUSABLE_ACT_TYPES:
            act_idx += 1
            attrs['act_type'] = op.type

            # Override with the actual conv→act intermediate exponent
            attrs['act1_input_exponent'] = conv_exp

            # Copy activation-specific attributes
            if op.type == 'HardSiluPie8':
                assert 'scale_int' in op.attributes, \
                    f"HardSiluPie8 op '{op.name}' missing scale_int. " \
                    f"_freeze_hardsilu_scales must run first."
                attrs['scale_int'] = op.attributes['scale_int']
            if 'lut' in op.attributes:
                attrs['lut'] = op.attributes['lut']

    return attrs


def _fuse_chain(graph, chain, attrs):
    """
    Fuse a chain [Conv, Act, ...] into a single TiledConvBlock op.

    Strategy:
      1. Mutate Conv's type to TiledConvBlock
      2. Set collected attributes
      3. Transfer last op's output quantization config to Conv's output config
      4. Update ExporterPatternInfo singleton with correct output exponent
         (because QuantVariableToIntPattern already ran in prepare_graph)
      5. Remove all non-Conv ops from the graph
    """
    from esp_ppq.IR.quantize import QuantableOperation
    from esp_ppq.parser.espdl.espdl_typedef import ExporterPatternInfo
    from esp_ppq.parser.espdl.export_patterns import QuantVariableToIntPattern

    conv_op = chain[0]
    last_op = chain[-1]

    # ── 1. Mutate type + set attributes ─────────────────
    conv_op._type = 'TiledConvBlock'
    for k, v in attrs.items():
        conv_op.attributes[k] = v

    # ── 2. Transfer output quantization config ──────────
    # After fusion, TiledConvBlock's output wire = last op's output wire.
    # The exponent export pattern reads from op.config_with_variable,
    # which zips config.output_quantization_config with op.outputs.
    # We must ensure conv_op's output config matches the last op's output.
    if (isinstance(conv_op, QuantableOperation)
            and isinstance(last_op, QuantableOperation)):
        last_out_cfg = last_op.config.output_quantization_config
        conv_out_cfg = conv_op.config.output_quantization_config
        if last_out_cfg and len(last_out_cfg) > 0 and conv_out_cfg and len(conv_out_cfg) > 0:
            conv_out_cfg[-1] = last_out_cfg[0]

            # ── 3. Fix the ExporterPatternInfo exponents ────
            # QuantVariableToIntPattern already ran in prepare_graph and
            # baked Conv's ORIGINAL output exponent into the singleton.
            # We must overwrite it with Act's output exponent.
            out_var = conv_op.outputs[0]
            new_exp = QuantVariableToIntPattern.calculate_exponent(last_out_cfg[0])
            if new_exp is not None:
                info = ExporterPatternInfo()
                old_exp = info.get_var_exponents(out_var.name)
                info.add_var_exponents(out_var.name, new_exp)


    # ── 4. Remove fused ops (all except Conv) ───────────
    for op in reversed(chain[1:]):
        graph.remove_operation(op, keep_coherence=True)


def _fuse_conv_act_to_tiled_block(graph):
    """
    Walk the graph and fuse Conv → Act patterns into TiledConvBlock nodes.

    Constraints (v1):
      - Conv must be standard (group == 1), not depthwise
      - Conv kernel must NOT be 1×1
      - Conv must have exactly one downstream consumer
      - That consumer must be a fusable activation op
      - quant_type must be INT8 or INT16

    The fusion uses a chain-based architecture:
      1. _detect_fusion_chain: walk downstream, collect [Conv, Act]
      2. _collect_chain_attributes: gather tile_h, act_type, scale_int, etc.
      3. _fuse_chain: mutate Conv→TiledConvBlock, transfer quant config, remove Act
    """
    from esp_ppq.parser.espdl.espdl_typedef import EspQuantType, ExporterPatternInfo

    fused_count = 0
    # Collect chains first, then fuse (avoid modifying during iteration)
    chains = []

    for op in graph.topological_sort():
        if op.type != 'Conv':
            continue

        # ── Constraint checks ──────────────────────────
        group = op.attributes.get('group', 1)
        if group != 1:
            continue  # skip depthwise

        kernel_shape = op.attributes.get('kernel_shape', [])


        quant_type = op.attributes.get('quant_type', None)
        if quant_type not in (EspQuantType.S8, EspQuantType.S16):
            continue

        chain = _detect_fusion_chain(graph, op)
        if chain is not None:
            chains.append(chain)

    # ── Apply fusions ──────────────────────────────────
    info = ExporterPatternInfo()

    for chain in chains:
        conv_op = chain[0]
        last_op = chain[-1]

        # Determine output shape for tile_h calculation
        out_var = conv_op.outputs[0]
        out_shape = list(out_var.shape) if out_var.shape else []

        if len(out_shape) == 4:
            # Shapes in the PPQ graph are still NCHW at this point.
            # The ExporterPatternInfo stores a permute (e.g. [0,2,3,1])
            # that is applied at serialization time to get NHWC.
            # Apply the same permute to get the real NHWC shape.
            perm = info.get_var_permute(out_var.name)
            if perm and len(perm) == 4:
                nhwc_shape = [out_shape[p] for p in perm]
            else:
                # No permute recorded → assume already NHWC
                nhwc_shape = out_shape
            N, H_out, W_out, C_out = nhwc_shape[0], nhwc_shape[1], nhwc_shape[2], nhwc_shape[3]
        else:

            continue

        quant_type = conv_op.attributes.get('quant_type', None)
        elem_size = 1 if quant_type == EspQuantType.S8 else 2

        tile_h = _calculate_tile_h(H_out, W_out, C_out, elem_size)

        # Compute c_tile for skip check
        c_tile_val = attrs.get('c_tile', C_out) if 'attrs' in dir() else C_out

        # Skip only if BOTH spatial and channel dimensions fit without tiling
        if tile_h >= H_out:
            # Need to peek at c_tile before full attrs collection
            filter_var = conv_op.inputs[1]
            fshape = filter_var.value.shape
            kH_c = fshape[0]
            kW_c = fshape[1] if len(fshape) >= 2 else 1
            C_in_c = fshape[2] if len(fshape) >= 3 else 1
            c_tile_check = _calculate_c_tile(kH_c, kW_c, C_in_c, C_out, elem_size)
            if c_tile_check >= C_out:

                continue

        # Collect attributes from chain
        attrs = _collect_chain_attributes(chain, tile_h, info, elem_size)

        # Fuse the chain
        _fuse_chain(graph, chain, attrs)

        fused_count += 1
        chain_desc = " + ".join(f"{op.name}({op.type})" for op in chain[1:])
        c_tile_val = attrs.get('c_tile', C_out)


    print(f"[custom_ops_patch] TiledConvBlock: fused {fused_count} Conv+Act pairs")


# ---------------------------------------------------------------------------
# TiledConvBlock: enable/disable fusion
# ---------------------------------------------------------------------------
def enable_tiled_conv_fusion():
    """
    Enable automatic Conv+Act -> TiledConvBlock fusion at export time.
    Call BEFORE export. The actual fusion happens inside export_graph.

    Usage:
        import custom_ops_patch
        custom_ops_patch.enable_tiled_conv_fusion()

        # ... quantize, calibrate, etc ...
        espdl_export(graph, ...)  # fusion happens automatically here
    """
    global _TILED_CONV_ENABLED
    _TILED_CONV_ENABLED = True
    print("[custom_ops_patch] Enabled TiledConvBlock fusion")


def disable_tiled_conv_fusion():
    """Disable TiledConvBlock fusion."""
    global _TILED_CONV_ENABLED
    _TILED_CONV_ENABLED = False
    print("[custom_ops_patch] Disabled TiledConvBlock fusion")


# ---------------------------------------------------------------------------
# Auto-execute on import
# ---------------------------------------------------------------------------

# ── TransposePIE ──
apply_patch_transpose_pie()
enable_debug_transpose_cases()
enable_pie_transpose_all(int8_only=True)

# ── TiledConvBlock ──
apply_patch_tiled_conv_block()
enable_tiled_conv_fusion()

# ── HardSiLU ──
apply_patch_hardsilu()
 

