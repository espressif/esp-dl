"""
Transformation 3: Conv1x1 Input-Channel Pruning on YOLO26n (FP32)
=================================================================
Pipeline:
  1. Export ONNX -> PPQ graph (or load pre-morphed .native)
  2. Baseline mAP (FP32 graph)
  3. Prune: zero N least-important input channels per 1x1 Conv
  4. Distill: train remaining weights to compensate
  5. Post-prune mAP + report
  6. Save .native checkpoint
"""

import os
import sys
import types
import logging
import torch

# ─────────────────────────────────────────────
# PATH SETUP (relative to quantize_yolo26/)
# ─────────────────────────────────────────────
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
QUANTIZE_ROOT  = os.path.normpath(os.path.join(SCRIPT_DIR, '..', '..', '..', '..'))  # quantize_yolo26/
SCRIPTS_DIR    = os.path.join(QUANTIZE_ROOT, 'scripts')
REPO_ROOT      = os.path.join(QUANTIZE_ROOT, 'neural_morphing')

for p in [REPO_ROOT, SCRIPTS_DIR, SCRIPT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
from esp_ppq.api import get_target_platform

IMG_SZ   = 512
PLATFORM = "p4"

class QATConfig:
    IMG_SZ              = IMG_SZ
    DEVICE              = "cuda" if torch.cuda.is_available() else "cpu"
    DATA_YAML_FILE      = "coco.yaml"
    BATCH_SIZE          = 32
    CALIB_MAX_IMAGES    = 3072
    CALIB_VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    DATA_FALLBACK_PATH  = "coco2017/images/train2017"
    CALIB_STEPS         = 64
    TARGET_PLATFORM     = get_target_platform("esp32" + PLATFORM, 8)
    INT16_LUT_STEP      = 32

    BASE_DIR     = os.getcwd()
    MODEL_NAME   = "yolo26n"
    PT_FILE      = f"{MODEL_NAME}.pt"
    ONNX_FILE    = f"{MODEL_NAME}_export.onnx"
    ESPDL_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output", f"T3_conv1x1_prune_{IMG_SZ}_{PLATFORM}")
    ONNX_PATH    = os.path.join(ESPDL_OUTPUT_DIR, ONNX_FILE)

# Inject config into sys.modules for shared helpers
if 'config' not in sys.modules:
    sys.modules['config'] = types.ModuleType('config')
sys.modules['config'].QATConfig = QATConfig

os.makedirs(QATConfig.ESPDL_OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# OPTIONAL: Load pre-morphed graph from another transformation
# Set to None to start from fresh ONNX export.
# Example: chain T1 (SiLU->HardSiLU) then T3 (prune 1x1):
#   INPUT_NATIVE = os.path.join(QUANTIZE_ROOT, "neural_morphing", "integrations",
#       "yolo26n", "Transformation1_SiLU_to_HardSiLU", "output",
#       "T1_silu_hsilu_512_p4", "morphed_hsilu.native")
# ─────────────────────────────────────────────
INPUT_NATIVE = None

# Number of input channels to prune per 1x1 Conv layer
N_CHANNELS_TO_PRUNE = 16


# ─────────────────────────────────────────────
# IMPORTS (after config injection + path setup)
# ─────────────────────────────────────────────
from utils import seed_everything, register_mod_op
from dataset import get_calibration_loader, SubsetCaliDataset
from esp_ppq_patch import apply_esp_ppq_patches
from esp_ppq_patch_2 import apply_addlut_patch
from notebook_helpers import extract_model_meta, prepare_onnx
from trainer import QATTrainer
from ultralytics.data.utils import check_det_dataset

from esp_ppq.api.interface import load_onnx_graph
from esp_ppq.api import load_native_graph
from esp_ppq.executor import TorchExecutor
from esp_ppq.parser import NativeExporter

from neural_morphing.engine import AdaptiveSubgraphOptimizationPass
from conv1x1_prune_strategy import Conv1x1PruneStrategy

# Custom ops: registers HardSiluPie8 forward, socket, quantizer, layout
import custom_ops_patch  # noqa: F401  - auto-executes on import


# ─────────────────────────────────────────────
# ENVIRONMENT SETUP
# ─────────────────────────────────────────────
seed_everything(1234)
register_mod_op()
apply_esp_ppq_patches()
apply_addlut_patch()

NATIVE_CHECKPOINT = os.path.join(QATConfig.ESPDL_OUTPUT_DIR, "morphed_pruned.native")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"  [Device] {QATConfig.DEVICE.upper()}")
    print(f"  [Model]  YOLO26n @ {IMG_SZ}x{IMG_SZ}")
    print()
    print("=" * 62)
    print("TRANSFORMATION 3: Conv1x1 Channel Pruning (FP32)")
    print(f"  Channels to prune per layer: {N_CHANNELS_TO_PRUNE}")
    print("=" * 62)

    # ── PHASE 1: ONNX Export (skipped if loading pre-morphed) ─
    if not INPUT_NATIVE:
        print("\n  [Phase 1] ONNX export...")
        prepare_onnx()
    model_meta = extract_model_meta()

    # ── PHASE 2: Load Graph ──────────────────────────────────
    if INPUT_NATIVE:
        print(f"\n  [Phase 2] Loading pre-morphed graph from {os.path.basename(INPUT_NATIVE)}...")
        graph = load_native_graph(INPUT_NATIVE)
    else:
        print("\n  [Phase 2] Loading ONNX -> PPQ graph...")
        graph = load_onnx_graph(onnx_import_file=QATConfig.ONNX_PATH)

    # Trace operation meta (needed for TorchExecutor)
    executor = TorchExecutor(graph=graph)
    dummy_input = torch.zeros([1, 3, IMG_SZ, IMG_SZ]).to(QATConfig.DEVICE)
    executor.tracing_operation_meta(inputs=dummy_input)

    # Count 1x1 convolutions
    conv1x1_ops = [n for n, op in graph.operations.items()
                   if op.type == 'Conv' and op.attributes.get("kernel_shape") == [1, 1]]
    print(f"    1x1 Conv ops: {len(conv1x1_ops)}")

    # ── PHASE 3: Baseline mAP ────────────────────────────────
    print("\n  [Phase 3] Baseline mAP (FP32, before pruning)...")
    baseline_cache = os.path.join(QATConfig.ESPDL_OUTPUT_DIR, "baseline_mAP.txt")
    if os.path.exists(baseline_cache):
        with open(baseline_cache, 'r') as f:
            baseline_mAP = float(f.read().strip())
        print(f"    [Cache] Loaded from {baseline_cache}")
    else:
        trainer_baseline = QATTrainer(graph=graph, model_meta=model_meta, device=QATConfig.DEVICE)
        baseline_mAP = trainer_baseline.eval()
        with open(baseline_cache, 'w') as f:
            f.write(f"{baseline_mAP:.6f}")
        print(f"    [Cache] Saved -> {baseline_cache}")
    print(f"    Baseline mAP50-95: {baseline_mAP:.4f}")

    # ── PHASE 4: Neural Morphing (Pruning + Distillation) ────
    print(f"\n  [Phase 4] Pruning 1x1 Convs ({N_CHANNELS_TO_PRUNE} channels each)...")

    # Build dataloaders for distillation
    data_cfg = check_det_dataset(QATConfig.DATA_YAML_FILE)
    cali_loader = get_calibration_loader(data_cfg)

    train_batches = []
    for batch in cali_loader:
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        train_batches.append(batch.float().to('cpu'))

    val_path = data_cfg.get('val')
    if isinstance(val_path, list):
        val_path = val_path[0]
    dataset_root = data_cfg.get('path', '')
    if not os.path.isabs(val_path) and dataset_root:
        val_path = os.path.join(dataset_root, val_path)
    if not os.path.isdir(val_path):
        val_path = os.path.join(dataset_root, 'images', 'val2017')

    val_dataset = SubsetCaliDataset(val_path, max_images=512)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=QATConfig.BATCH_SIZE, shuffle=False)
    val_batches = []
    for batch in val_loader:
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        val_batches.append(batch.float().to('cpu'))

    print(f"    Distillation data: {len(train_batches)} train batches, {len(val_batches)} val batches")

    # Layers to keep unpruned
    frozen_prefixes = [
        "/model.0/",             # stem conv (3 input channels, too few to prune)
        "/model.23/",            # detection head (accuracy-critical)
    ]
    print(f"    Freezing layers matching {len(frozen_prefixes)} prefixes")

    strategy = Conv1x1PruneStrategy(
        n_prune=N_CHANNELS_TO_PRUNE,
        lr=1e-3,
        steps=2000,
        patience=2,
        min_cosine=0.9908,
        scale_bounds=(0.985, 1.015),
        skip_prefixes=frozen_prefixes,
    )

    plots_dir = os.path.join(SCRIPT_DIR, "plots")

    engine = AdaptiveSubgraphOptimizationPass()
    logging.getLogger('PPQ').setLevel(logging.ERROR)
    engine.optimize(
        graph,
        target_op_types=['Conv'],
        block_size=1,
        dataloader=train_batches,
        val_dataloader=val_batches,
        executing_device=QATConfig.DEVICE,
        caching_device='cpu',
        strategy=strategy,
        plots_dir=plots_dir,
    )
    logging.getLogger('PPQ').setLevel(logging.WARNING)

    # Post-prune stats
    pruned_ops = [n for n in graph.operations
                  if '_parent_op' in graph.operations[n].attributes]
    remaining_conv1x1 = [n for n, op in graph.operations.items()
                         if op.type == 'Conv' and op.attributes.get("kernel_shape") == [1, 1]]
    print(f"\n    Pruned 1x1 ops    : {len(pruned_ops)}")
    print(f"    Remaining 1x1     : {len(remaining_conv1x1)}")

    # ── PHASE 5: Post-prune mAP ──────────────────────────────
    print("\n  [Phase 5] Post-prune mAP (FP32)...")
    trainer_pruned = QATTrainer(graph=graph, model_meta=model_meta, device=QATConfig.DEVICE)
    pruned_mAP = trainer_pruned.eval()
    print(f"    Post-prune mAP50-95: {pruned_mAP:.4f}")

    # ── Save .native checkpoint ──────────────────────────────
    print(f"\n  [Checkpoint] Saving -> {NATIVE_CHECKPOINT}")
    exporter = NativeExporter()
    exporter.export(file_path=NATIVE_CHECKPOINT, graph=graph)
    print(f"    Saved.")

    # ── PHASE 6: Report ───────────────────────────────────────
    delta = baseline_mAP - pruned_mAP
    original_count = len(conv1x1_ops)
    pruned_count = len(pruned_ops)
    print("\n" + "=" * 62)
    print("RESULTS")
    print("=" * 62)
    print(f"  Channels pruned/layer: {N_CHANNELS_TO_PRUNE}")
    print(f"  Baseline mAP50-95   : {baseline_mAP:.4f}")
    print(f"  Pruned mAP50-95     : {pruned_mAP:.4f}  (delta: {delta:+.4f})")
    print(f"  Recovery            : {pruned_mAP/baseline_mAP*100:.1f}%")
    print(f"  Original 1x1 Convs  : {original_count}")
    print(f"  Pruned layers       : {pruned_count}")
    print(f"  Remaining 1x1       : {len(remaining_conv1x1)}")
    print("=" * 62)
