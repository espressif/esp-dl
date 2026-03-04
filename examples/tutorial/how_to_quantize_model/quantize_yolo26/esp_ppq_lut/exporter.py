
from esp_ppq.parser import EspdlExporter
from esp_ppq.core import TargetPlatform
import esp_ppq.lib as PFL
from .emulator import GlobalMode, SimulationMode

class HardwareAwareEspdlExporter(EspdlExporter):
    """
    ESP-DL Exporter that automatically manages the Simulation vs Ideal Math context.
    Ensures that when the exporter generates LUT tables, it uses perfect math,
    then automatically reverts to bit-exact hardware simulation for everything else.
    """
    def export(self, *args, **kwargs):
        # 1. Flip to IDEAL Mode before export starts
        # This ensures that AddLUTPattern calls op.forward() in 'Pure Math' mode
        if getattr(self, 'verbose', False):
            print("[ESPDL Exporter] Switching to IDEAL MATH for Table Generation...")
        GlobalMode.set(SimulationMode.IDEAL_MATH)
        
        try:
            # 2. Run the actual Export logic
            super().export(*args, **kwargs)
        finally:
            # 3. Always flip back to SIMULATION Mode
            # This ensures that your next validation/verification step is bit-exact
            if getattr(self, 'verbose', False):
                print("[ESPDL Exporter] Reverting to HARDWARE SIMULATION for Validation...")
            GlobalMode.set(SimulationMode.SIMULATION)

def register_espdl_exporter(verbose=False):
    """
    Registers the Hardware-Aware Exporter as the official handler for ESP-DL platforms.
    """
    # Store verbosity as a class attribute for the singleton exporter behavior
    HardwareAwareEspdlExporter.verbose = verbose
    
    platforms = [
        TargetPlatform.ESPDL_INT8,
        TargetPlatform.ESPDL_INT16,
        TargetPlatform.ESPDL_S3_INT8,
        TargetPlatform.ESPDL_S3_INT16
    ]
    
    for platform in platforms:
        # PFL.register_network_exporter replaces the library's default exporter
        # with our mode-aware decorator.
        PFL.register_network_exporter(HardwareAwareEspdlExporter, platform)
        if verbose:
            print(f"[ESPDL Exporter] Registered HardwareAwareExporter for: {platform.name}")
