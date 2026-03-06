import torch
import os
import cv2
import numpy as np
from export import apply_export_patches, ESP_YOLO
from esp_ppq.IR import BaseGraph
from config import QATConfig

def extract_model_meta():
    tmp_model = ESP_YOLO(QATConfig.PT_FILE)
    detect_head = tmp_model.model.model[-1]
    ch = [m[0].conv.in_channels for m in detect_head.cv2]
    meta = {
        'nc': detect_head.nc,
        'reg_max': detect_head.reg_max,
        'stride': detect_head.stride,
        'ch': ch
    }
    if isinstance(meta['stride'], torch.Tensor):
        meta['stride'] = meta['stride'].tolist()
    print(f"Metadata: NC={meta['nc']}, RegMax={meta['reg_max']}, Stride={meta['stride']}")
    return meta

def prepare_onnx():
    model = ESP_YOLO(QATConfig.PT_FILE)
    # Patches the ESP_Attention nodes for static reshaping
    apply_export_patches(model)
    model.export(
        format="onnx", opset=13,
        simplify=True, imgsz=QATConfig.IMG_SZ,
        dynamic=False
    )
    print(f"Exported base ONNX to {QATConfig.ONNX_PATH}")

def prune_graph_safely(graph: BaseGraph) -> BaseGraph:
    """Removes disconnected operations and unused variables."""
    round_count = 0
    while True:
        ops_removed, vars_removed = 0, 0
        dead_ops = []
        for op in list(graph.operations.values()):
            is_output = any(var.name in graph.outputs for var in op.outputs)
            has_consumers = any(len(var.dest_ops) > 0 for var in op.outputs)
            if not is_output and not has_consumers: dead_ops.append(op)

        for op in dead_ops:
            for var in list(op.inputs):
                op.inputs.remove(var)
                if op in var.dest_ops: var.dest_ops.remove(op)
            graph.remove_operation(op, keep_coherence=False)
            ops_removed += 1

        dead_vars = []
        for var in list(graph.variables.values()):
            if var.name in graph.inputs or var.name in graph.outputs: continue
            if len(var.dest_ops) == 0: dead_vars.append(var)

        for var in dead_vars:
            if var.name in graph.variables:
                graph.variables.pop(var.name)
                vars_removed += 1
        round_count += 1
        if ops_removed == 0 and vars_removed == 0: break
    return graph


# ==========================================================================
# ESP-DL Inference Helpers
# ==========================================================================

def espdl_preprocess(img_bgr, dst_shape, pad_val=114):
    """
    Geometrically exact clone of the ESP-DL C++ resize_nn + letterbox logic.
    Matches the hardware preprocessing pixel-for-pixel.

    Args:
        img_bgr   : input image in BGR format (OpenCV standard)
        dst_shape : (dst_h, dst_w) target resolution
        pad_val   : padding fill value (114 = ESP-DL default)

    Returns:
        Padded/resized uint8 BGR image of shape (dst_h, dst_w, 3)
    """
    src_h, src_w = img_bgr.shape[:2]
    dst_h, dst_w = dst_shape

    scale_x = dst_w / float(src_w)
    scale_y = dst_h / float(src_h)

    border_top = border_bottom = border_left = border_right = 0
    if scale_x < scale_y:
        pad_h = dst_h - int(min(scale_x, scale_y) * src_h)
        border_top = pad_h // 2
        border_bottom = pad_h - border_top
    else:
        pad_w = dst_w - int(min(scale_x, scale_y) * src_w)
        border_left = pad_w // 2
        border_right = pad_w - border_left

    act_dst_w = dst_w - border_left - border_right
    act_dst_h = dst_h - border_top - border_bottom
    inv_scale_x = float(src_w) / act_dst_w
    inv_scale_y = float(src_h) / act_dst_h

    out_img = np.full((dst_h, dst_w, 3), pad_val, dtype=np.uint8)
    for y_dst in range(act_dst_h):
        y_src = min(int(y_dst * inv_scale_y), src_h - 1)
        for x_dst in range(act_dst_w):
            x_src = min(int(x_dst * inv_scale_x), src_w - 1)
            out_img[y_dst + border_top, x_dst + border_left] = img_bgr[y_src, x_src]
    return out_img


def eval_espdl_model(test_image_path, graph, target_img_sz, data_yaml,
                     platform="p4", conf_thresh=0.25, iou_thresh=0.45,
                     output_dir="results"):
    """
    Run ESP-DL emulated inference on a single image using the quantized graph.
    Uses the same preprocessing as the ESP-DL C++ runtime for bit-exact simulation.
    Saves the annotated image and displays it inline in the notebook.

    Args:
        test_image_path : str  — path to the input image (e.g. 'results/bus.jpg')
        graph           : BaseGraph — quantized PPQ graph after graph surgery
        target_img_sz   : int  — model input square size (e.g. 512)
        data_yaml       : str  — path to dataset YAML (e.g. 'coco.yaml', 'data.yaml')
                                 Used to read class names — works for any dataset.
        platform        : str  — target chip identifier, e.g. 'p4' or 's3'
                                 Embedded in the output filename: bus_512_s8_p4.jpg
        conf_thresh     : float — confidence threshold (default 0.25)
        iou_thresh      : float — NMS IoU threshold (default 0.45)
        output_dir      : str  — directory where annotated image is saved

    Returns:
        predictions : list of dicts, each with keys:
                      {'box': [x1,y1,x2,y2], 'score': float,
                       'class': str, 'class_id': int}
        output_path : str — absolute path to the saved annotated image
    """
    import matplotlib.pyplot as plt
    from torchvision.ops import nms
    from esp_ppq.executor import TorchExecutor
    from ultralytics.data.utils import check_det_dataset

    # ── 1. Load class names from YAML (generic — any dataset) ─────────────
    dataset_info = check_det_dataset(data_yaml)
    names_dict   = dataset_info['names']   # {0: 'person', 1: 'car', ...}
    class_names  = [names_dict[i] for i in sorted(names_dict.keys())]
    print(f"[eval_espdl_model] Dataset: {data_yaml} | Classes: {len(class_names)}")

    # ── 2. Load & preprocess image ─────────────────────────────────────────
    im0 = cv2.imread(test_image_path)
    if im0 is None:
        print(f"[eval_espdl_model] ERROR: could not read '{test_image_path}'")
        return None

    im      = espdl_preprocess(im0, dst_shape=(target_img_sz, target_img_sz))
    im_draw = im.copy()   # keep BGR copy for drawing

    im_chw     = np.ascontiguousarray(im[..., ::-1].transpose(2, 0, 1))  # BGR→RGB, HWC→CHW
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    test_input = torch.from_numpy(im_chw).to(device).float().div(255.0).unsqueeze(0)

    # ── 3. Run inference through quantized graph ───────────────────────────
    executor    = TorchExecutor(graph=graph)
    raw_outputs = executor.forward(test_input)

    output_keys = list(graph.outputs.keys())
    tv = {name: raw_outputs[i].detach().cpu() for i, name in enumerate(output_keys)}

    # ── 4. Decode P3 / P4 / P5 head outputs ───────────────────────────────
    strides = [8, 16, 32]
    scales  = ["p3", "p4", "p5"]
    all_boxes, all_scores = [], []

    for stride, scale in zip(strides, scales):
        box = tv[f'one2one_{scale}_box'].float()   # (1, 4,  H, W)
        cls = tv[f'one2one_{scale}_cls'].float()   # (1, NC, H, W)
        cls = torch.sigmoid(cls)

        _, _, H, W = box.shape
        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        grid = torch.stack((grid_x, grid_y), 0).float().add(0.5).unsqueeze(0)  # (1,2,H,W)

        x1 = (grid[:, 0:1] - box[:, 0:1]) * stride
        y1 = (grid[:, 1:2] - box[:, 1:2]) * stride
        x2 = (grid[:, 0:1] + box[:, 2:3]) * stride
        y2 = (grid[:, 1:2] + box[:, 3:4]) * stride

        decoded = torch.cat([x1, y1, x2, y2], 1)  # (1,4,H,W)
        all_boxes.append(decoded.flatten(2))         # (1,4,H*W)
        all_scores.append(cls.flatten(2))            # (1,NC,H*W)

    all_boxes  = torch.cat(all_boxes,  2).squeeze(0).T  # (N, 4)
    all_scores = torch.cat(all_scores, 2).squeeze(0).T  # (N, NC)

    # ── 5. Confidence filter + NMS ─────────────────────────────────────────
    max_scores, class_ids = torch.max(all_scores, dim=1)
    mask    = max_scores > conf_thresh
    fboxes, fscores, fclasses = all_boxes[mask], max_scores[mask], class_ids[mask]

    keep          = nms(fboxes, fscores, iou_threshold=iou_thresh)
    final_boxes   = fboxes[keep]
    final_scores  = fscores[keep]
    final_classes = fclasses[keep]

    # ── 6. Draw bounding boxes (matplotlib style matching visualize_logs.py) ─
    COLORS = ['#FF3333', '#3399FF', '#33CC66', '#FF9900', '#CC33FF',
              '#00CCCC', '#FF66AA', '#99FF33', '#FF6600', '#6633FF']

    predictions = []
    import matplotlib.patches as patches

    # Collect predictions first
    for j in range(len(final_boxes)):
        box      = final_boxes[j].numpy()
        score    = final_scores[j].item()
        cls_id   = final_classes[j].item()
        cls_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        predictions.append({"box": box.tolist(), "score": score,
                             "class": cls_name, "class_id": cls_id})

    # ── 7. Build matplotlib figure (high quality rendering) ───────────────
    canvas_rgb = cv2.cvtColor(im_draw, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(canvas_rgb)

    for j, p in enumerate(predictions):
        x1, y1, x2, y2 = [int(v) for v in p['box']]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(target_img_sz, x2); y2 = min(target_img_sz, y2)

        color = COLORS[j % len(COLORS)]
        label = f"{p['class']} {p['score']:.2f}"

        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(
            x1, max(y1 - 6, 0), label,
            color='white', fontsize=11, fontweight='bold',
            bbox=dict(facecolor=color, alpha=0.75, lw=0, pad=2)
        )

    ax.axis('off')
    ax.set_title(
        f"ESP-DL Emulated Inference [{platform.upper()}] — {len(predictions)} detection(s)",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()

    # ── 8. Save (high quality) ─────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    stem, ext   = os.path.splitext(os.path.basename(test_image_path))
    output_path = os.path.join(output_dir, f"{stem}_{target_img_sz}_s8_{platform}{ext}")
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[eval_espdl_model] {len(predictions)} detection(s) → saved: {output_path}")

    # ── 9. Display inline (use %matplotlib inline in notebook cell) ────────
    plt.show()

    return predictions, output_path
