
from typing import List
from esp_ppq.IR import BaseGraph, Operation
from esp_ppq.quantization.optim import QuantizationOptimizationPass
from esp_ppq.core import TargetPlatform
from esp_ppq.executor.torch import OPERATION_FORWARD_TABLE

class EspdlLUTFusionPass(QuantizationOptimizationPass):
    """
    Graph Re-writing pass for ESP-DL Deployment.
    Converts activation operations (Swish, Sigmoid) into 'LUT' operations
    to trigger the Hardware-Accelerated Fast Path in the ESP-DL Runtime.
    """
    def __init__(self, target_ops: List[str] = ['Swish', 'Sigmoid', 'Tanh'], 
                 verify: bool = False, plot: bool = False, 
                 output_dir: str = "outputs", lut_step: int = 32,
                 verbose: bool = False):
        super().__init__(name='ESPDL LUT Fusion Pass')
        self.target_ops = target_ops
        self.verify = verify
        self.plot = plot
        self.output_dir = output_dir
        self.lut_step = lut_step
        self.lut_count = 0
        self.verbose = verbose

    def verify_compatibility(self, op: Operation) -> bool:
        """
        Ensures the operation is valid for LUT fusion.
        Checks for:
        1. Correct Target Platform (Strictly ESP-DL 16-bit)
        2. TQC precision is exactly 16-bit
        3. Quantable Operation (Must be calibrated)
        """
        # 1. Platform Check (Restrict to 16-bit domains)
        if op.platform not in {
            TargetPlatform.ESPDL_INT16, TargetPlatform.ESPDL_S3_INT16
        }:
            # Soft skip: Just ignore INT8 or other platforms without crashing
            return False

        # 2. Check for Quantization Configuration
        if not hasattr(op, 'config') or op.config is None:
            print(f"\033[91m[ESPDL Pass] Skipping {op.name}: Operation is not quantized (No TQC found).\033[0m")
            return False
            
        # 3. Precision Check (Deep Verification of TQC)
        # The LUT fast-path on ESP32-P4/S3 requires 16-bit input precision.
        if op.input_quant_config[0].num_of_bits != 16:
            print(f"\033[91m[ESPDL Pass] Warning: Skipping {op.name} because it is not 16-bit (Found {op.input_quant_config[0].num_of_bits}-bit).\033[0m")
            return False
            
        return True

    def _self_audit(self, op: Operation, index: int):
        """
        [DEPRECATED] Python-vs-Python parity check.
        Use firmware dual-mode validation (HW vs SIMULATION / HW vs IDEAL_MATH) instead.
        """
        import warnings
        warnings.warn(
            "_self_audit is deprecated. Use firmware dual-mode test for bit-exact validation.",
            DeprecationWarning, stacklevel=2
        )

    def optimize(self, graph: BaseGraph, **kwargs):
        """
        Traverses the graph and performs the 'Topology Swap'.
        """
        import os
        self.lut_count = 0
        
        # Clear/Init verification directory if enabled
        if self.verify:
            verify_dir = os.path.join(self.output_dir, "lut_verification")
            os.makedirs(verify_dir, exist_ok=True)
            manifest_file = os.path.join(verify_dir, "mapping.json")
            if os.path.exists(manifest_file):
                os.remove(manifest_file)

        for op in graph.operations.values():
            if op.type in self.target_ops:
                # 1. Safety Check
                if not self.verify_compatibility(op):
                    continue

                # 2. Store Metadata (The 'Shadow Attributes')
                op.attributes['original_op_type'] = op.type
                op.attributes['int16_lut_step'] = self.lut_step
                
                # 3. Rename the Operation Type
                op.type = 'LUT'
                if self.verbose:
                    print(f"[ESPDL Pass] Fused {op.attributes['original_op_type']} -> LUT for operation: {op.name}")

                # 4. Self-Audit (Deprecated — use firmware dual-mode test instead)
                self.lut_count += 1
                if self.verify:
                    print(f"\n[ESPDL Pass] Note: 'verify' flag is deprecated. "
                          f"Use firmware dual-mode validation (HW vs SIMULATION / HW vs IDEAL_MATH) "
                          f"for bit-exact proof. Skipping Python-side parity check.")

    @property
    def is_post_quantization_pass(self) -> bool:
        return True
