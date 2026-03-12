# esp-ppq Modification Checklist

This document helps determine whether and how to modify esp-ppq when adding a new operator.

## Decision Flowchart

Every new operator requires TWO independent checks. The quantization system and the
layout system are separate — both must recognize the operator.

```
CHECK 1: Is the operator in EspdlQuantizer.quant_operation_types?
├── YES → OK, no change needed for quantization
└── NO  → Add to quant_operation_types in EspdlQuantizer.py

CHECK 2: Is the operator in a layout op set in espdl_typedef.py?
(This check is ALWAYS required — even if CHECK 1 passed)
├── YES → OK, layout transformation will handle it
└── NO  → Add to the correct op set (see mapping table below)
          WITHOUT THIS, export will fail with "Can not reset {op_type} layout"

The layout system (reset_graph_layout in layout_patterns.py) processes every
operator during NCHW→NHWC conversion. Each op set maps to a specific pattern:
  - ADD_LIKE_OP_SET     → BypassAddLikePattern (handles input shape broadcasting)
  - PASSIVE_LAYOUT_OP_SET → BypassPassiveLayoutPattern (pass-through)
  - CONV_LAYOUT_OP_SET  → ResetConvLayoutPattern (spatial layout transform)
  - AXIS_TRANSFORM_OP_SET → AxisTransformPattern (axis attribute adjustment)
  - OTHER_OP_SET        → RestoreOriginLayoutPattern (restore original layout)

THEN check for special cases:
    ├── Does it have bias inputs (like Conv/Gemm)?
    │   └── YES → Add bias config in create_espdl_quant_config()
    │
    ├── Is it a non-linear activation?
    │   └── YES → Add to ACTIVATION_OP_SET in espdl_typedef.py
    │             Add to AddLUTPattern in export_patterns.py
    │             Add LUT forward in executor/op/torch/espdl.py
    │
    ├── Does it need special output quantization (like Softmax)?
    │   └── YES → Add output FP32 rule in create_espdl_quant_config()
    │
    ├── Does it have inputs with different platform needs?
    │   └── YES → Add custom OpSocket in IR/base/opdef.py
    │
    └── Does it need special weight layout (like Conv NCHW→NHWC)?
        └── YES → Add layout rule in export_patterns.py
```

## Files and What to Modify

### 1. EspdlQuantizer.py

**Path**: `esp-ppq/esp_ppq/quantization/quantizer/EspdlQuantizer.py`

**quant_operation_types** (always needed for new ops):
```python
@property
def quant_operation_types(self) -> set:
    return {
        # ... existing ops ...
        "MyNewOp",  # Add here
    }
```

**create_espdl_quant_config()** (only if special rules needed):
```python
def create_espdl_quant_config(self, operation, num_of_bits, quant_min, quant_max, bias_bits):
    base_quant_config = self.create_default_quant_config(...)

    # Example: operator with bias
    if operation.type in {"Conv", "ConvTranspose", "Gemm", "MyNewOp"}:
        if operation.num_of_input > 2:
            bias_config = base_quant_config.input_quantization_config[-1]
            bias_config.num_of_bits = bias_bits
            bias_config.state = QuantizationStates.PASSIVE_INIT

    # Example: output stays FP32
    elif operation.type in {"Softmax", "MyNewOp"}:
        base_quant_config.output_quantization_config[0].state = QuantizationStates.FP32
```

### 2. espdl_typedef.py

**Path**: `esp-ppq/esp_ppq/parser/espdl/espdl_typedef.py`

Add to the correct op set based on operator category:

| Op Category | Op Set | Examples |
|------------|--------|----------|
| Activation | `ACTIVATION_OP_SET` | Relu, Sigmoid, HardSwish |
| Math unary | `MATH_OP_SET` | Exp, Log, Sqrt, Neg |
| Elementwise binary | `ADD_LIKE_OP_SET` | Add, Sub, Mul, Div |
| Conv-like | `CONV_LAYOUT_OP_SET` | Conv, MaxPool, DepthToSpace |
| Reduce | `REDUCE_OP_SET` | ReduceSum, ReduceMean |
| Softmax-like | `SOFTMAX_LIKE_OP_SET` | Softmax, Split |
| Other | `OTHER_OP_SET` | Reshape, Gather, GRU |

### 3. export_patterns.py

**Path**: `esp-ppq/esp_ppq/parser/espdl/export_patterns.py`

Rarely needed. Modify when:
- Adding LUT activation: update `AddLUTPattern.check_op()`
- Adding conv-like op: update `ResetParamLayoutPattern`
- Adding fusion pattern: create new pattern class

### 4. opdef.py (OpSocket)

**Path**: `esp-ppq/esp_ppq/IR/base/opdef.py`

Only needed when inputs have heterogeneous platform requirements.
Most ops use `DEFAULT_SOCKET_CREATOR`. Custom example:

```python
# Input 0 is data (follows platform), Input 1 is index (stays FP32)
MyOp_Socket = OpSocket(
    in_plat=[SocketType.UNSPECIFIED, SocketType.FP32],
    out_plat=[SocketType.UNSPECIFIED],
)

DEFAULT_SOCKET_TABLE = {
    # ... existing entries ...
    'MyOp': lambda: MyOp_Socket,
}
```

### 5. executor/op/torch/espdl.py (LUT Backend)

**Path**: `esp-ppq/esp_ppq/executor/op/torch/espdl.py`

Only needed for LUT-based activation ops. Register the forward computation
that generates the LUT during export:

```python
ESPDL_QUANT_BACKEND_TABLE = {
    # ... existing entries ...
    'MyActivation': my_activation_forward_func,
}
```

## Common Patterns

### Pattern A: Binary Elementwise Op (like Add, Mul, Mod, Pow)
- Verify/add to `quant_operation_types` ✓
- Verify/add to `ADD_LIKE_OP_SET` in `espdl_typedef.py` ✓ (critical for BypassAddLikePattern to handle input shape broadcasting)
- No special quant rules
- No additional export patterns
- No OpSocket changes

### Pattern B: Unary Math Op (like Abs, Sqrt, Neg)
- Verify/add to `quant_operation_types` ✓
- Verify/add to `MATH_OP_SET` in `espdl_typedef.py` ✓ (covered by PASSIVE_LAYOUT_OP_SET via union)
- No special quant rules
- No additional export patterns
- No OpSocket changes

### Pattern C: Op with Bias (like Conv, Gemm)
- Verify/add to `quant_operation_types` ✓
- Verify/add to `CONV_LAYOUT_OP_SET` ✓
- Add bias config in `create_espdl_quant_config()` ✓
- May need weight layout in `export_patterns.py` ✓
- No OpSocket changes

### Pattern D: LUT Activation (like HardSwish, Sigmoid)
- Verify/add to `quant_operation_types` ✓
- Verify/add to `ACTIVATION_OP_SET` ✓ (covered by PASSIVE_LAYOUT_OP_SET via union)
- Add LUT handling in `export_patterns.py` ✓
- Add LUT forward in `executor/op/torch/espdl.py` ✓
- No OpSocket changes

### Pattern E: Shape/Index Op (like Gather, Slice)
- Verify/add to `quant_operation_types` ✓
- Verify/add to `OTHER_OP_SET` ✓
- May need custom OpSocket (if index inputs should stay FP32) ✓
- Set `is_active_quant_op = False` if passive
