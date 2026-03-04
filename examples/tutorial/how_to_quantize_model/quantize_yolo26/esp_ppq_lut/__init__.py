from .emulator import register_lut_op_handler, set_simulation_mode, SimulationMode
from . import emulator
from .exporter import register_espdl_exporter
from .passes import EspdlLUTFusionPass
from .utils import run_numerical_verification, to_c_header, generate_comparison_plot
from .verifier import run_deep_verification
from .patches import patch_esp_ppq_library

def initialize(step=32, verbose=False):
    """
    The entry point for the ESP-PPQ LUT extension.
    Initializes hardware-aware execution and deployment plugins.
    """
    # Apply essential library fixes first
    patch_esp_ppq_library(verbose=verbose)
    
    # Register handlers and exporters
    register_lut_op_handler(verbose=verbose)
    register_espdl_exporter(verbose=verbose)
    
    if verbose:
        print("[ESP-PPQ-LUT] Activation forwarders registered for simulation.")
        print(f"[ESP-PPQ-LUT] Extension Initialized (Default Step={step})")
