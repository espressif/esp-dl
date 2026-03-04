import os
import torch
import torch.nn.functional as F
from esp_ppq.IR import Operation
from esp_ppq.executor.base import OPERATION_FORWARD_TABLE
from esp_ppq.executor.op.torch.default import DEFAULT_BACKEND_TABLE
from .utils import generate_comparison_plot, to_c_header, update_verification_manifest

def run_deep_verification(graph, executor, dataloader=None, output_dir="outputs", verbose=False):
    """
    Performs an exhaustive 65,536-point sweep for every LUT operation in the graph.
    
    > [!IMPORTANT]
    > **Isolated Activation Verification Required**
    > This utility is designed to verify individual Activation Layers (Swish, Sigmoid, etc.) 
    > in isolation. To ensure accurate results, the model used for this test should ideally 
    > consist only of the activation function being verified (with optional identity 
    > layers such as a Conv2D with weight=1.0).
    """
    if verbose:
        print("\n" + "="*80)
        print("ESP32-P4 DEEP VERIFICATION: EXHAUSTIVE SWEEP")
        print("="*80)
    
    verify_dir = os.path.join(output_dir, "lut_verification")
    os.makedirs(verify_dir, exist_ok=True)
    
    # Initialize/Clear manifest
    manifest_path = os.path.join(verify_dir, "mapping.json")
    if os.path.exists(manifest_path):
        os.remove(manifest_path)
    
    lut_ops = [op for op in graph.operations.values() if op.type == 'LUT']
    if not lut_ops:
        print("[SKIP] No LUT operations found in graph for deep verification.")
        return

    # Extract calibration sample if available
    calib_data = None
    if dataloader is not None:
        calib_data = dataloader[0] if isinstance(dataloader, list) else next(iter(dataloader))

    for i, op in enumerate(lut_ops):
        index = i + 1
        if verbose:
            print(f"[Verifier] Processing Layer: {op.name}")
        
        original_op_type = op.attributes.get('original_op_type', 'Unknown')
        scale = op.input_quant_config[0].scale.item()
        
        metadata = {
            "layer_name": op.name,
            "original_type": original_op_type,
            "parity_plot": f"lut_{index}_parity.png" # Path assumes fusion pass parity check also ran
        }

        # 1. Exhaustive INT16 sweep + Out-of-Bounds
        full_sweep = torch.arange(-32768, 32768, dtype=torch.float)
        oob = torch.tensor([-40000.0, -32769.0, 32768.0, 40000.0])
        test_data = torch.cat([oob[:2], full_sweep, oob[2:]]).view(1, 1, 1, -1) * scale
        
        # 2. Run Simulation
        y_sim = executor(test_data)[0]
        
        # 3. Dynamic Ideal Math Lookup
        if original_op_type not in DEFAULT_BACKEND_TABLE:
            if verbose:
                print(f"[WARNING] Math implementation for '{original_op_type}' not found. Skipping plot.")
            continue
            
        math_fn = DEFAULT_BACKEND_TABLE[original_op_type]
        y_ideal = math_fn(op, [test_data])
        
        # 4. Calibration Data Plotting
        y_calib_sim = None
        if calib_data is not None:
            y_calib_sim = executor(calib_data)[0]

        # 5. Generate Artifacts
        sweep_filename = f"lut_{index}_sweep.png"
        header_filename = f"lut_{index}_test.h"
        
        generate_comparison_plot(
            test_data, y_sim, y_ideal, 
            x_calib=calib_data, y_calib=y_calib_sim,
            output_path=os.path.join(verify_dir, sweep_filename),
            verbose=verbose
        )
        
        header_out = os.path.join(verify_dir, header_filename)
        if os.path.exists(header_out): os.remove(header_out) # New file for each run
        
        to_c_header(test_data, f"input_{index}", header_out)
        to_c_header(y_sim, f"output_{index}", header_out)
        
        metadata["sweep_plot"] = sweep_filename
        metadata["c_header"] = header_filename

        # Update JSON Manifest
        update_verification_manifest(verify_dir, index, metadata)

    if verbose:
        print(f"\n[Verifier] Deep Verification Complete. Artifacts in: {verify_dir}")
