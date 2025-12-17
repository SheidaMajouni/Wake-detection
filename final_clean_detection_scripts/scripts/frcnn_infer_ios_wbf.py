"""
Inference for Wake Detection (Faster R-CNN + IoS/WBF)
-----------------------------------------------------
- Loads a trained Faster R-CNN checkpoint (.pth).
- Runs detection on images in a folder (recursively).
- Applies IoS + WBF (IoS=0.92, 'fuse') to reduce duplicate boxes.
- Saves annotated images and a JSON of detections.

Run:
  python frcnn_infer_ios_wbf.py \
      --weights frcnn_best_model_trained.pth \
      --yaml ../Dataset/vessel_wakes.yaml \
      --images frcnn_samples \
      --out predictions_frcnn_out

Dependencies: torch, torchvision, pillow, numpy, opencv-python, pyyaml
"""

import os, json, argparse
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import yaml

from torchvision.transforms import functional as FT
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator

# ---------------- IoS + WBF ----------------

def _area(b):   # [x1,y1,x2,y2]
    return max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])

def _inter(a,b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    return max(0.0, x2-x1) * max(0.0, y2-y1)

def _ios(a,b):
    I = _inter(a,b)
    return I / (min(_area(a), _area(b)) + 1e-9)

def _wbf_single_class(boxes, scores, ios_thr=0.92, p=3.0, score_fuse="max", strategy="fuse"):
    if len(scores) == 0:
        return np.empty((0,4)), np.empty((0,))
    used = np.zeros(len(scores), dtype=bool)
    order = np.argsort(scores)[::-1]
    fused_boxes, fused_scores = [], []

    for i in order:
        if used[i]:
            continue
        cluster = [i]; used[i] = True
        for j in order:
            if used[j]: continue
            if _ios(boxes[i], boxes[j]) >= ios_thr:
                cluster.append(j); used[j] = True

        b = boxes[cluster]; s = scores[cluster]
        if strategy == "best":
            k = int(np.argmax(s)); fused = b[k]; score = float(s[k])
        elif strategy == "min_area":
            k = int(np.argmin((b[:,2]-b[:,0])*(b[:,3]-b[:,1]))); fused = b[k]; score = float(s[k])
        else:  # "fuse"
            w = (s ** p)[:, None]
            fused = (w * b).sum(axis=0) / w.sum()
            score = float(s.max() if score_fuse == "max" else s.mean())
        fused_boxes.append(fused); fused_scores.append(score)

    return np.vstack(fused_boxes), np.array(fused_scores)

def ios_wbf_per_class(boxes, scores, labels, ios_thr=0.92, p=3.0, score_fuse="max", strategy="fuse"):
    if len(scores) == 0:
        return boxes, scores, labels
    out_b, out_s, out_l = [], [], []
    for c in np.unique(labels):
        m = labels == c
        fb, fs = _wbf_single_class(boxes[m], scores[m], ios_thr=ios_thr, p=p,
                                   score_fuse=score_fuse, strategy=strategy)
        out_b.append(fb); out_s.append(fs); out_l.append(np.full(len(fs), c, dtype=labels.dtype))
    boxes_f = np.concatenate(out_b, axis=0) if out_b else boxes
    scores_f = np.concatenate(out_s, axis=0) if out_s else scores
    labels_f = np.concatenate(out_l, axis=0) if out_l else labels
    order = np.argsort(scores_f)[::-1]
    return boxes_f[order], scores_f[order], labels_f[order]

# ---------------- Model (must match training) ----------------

def build_frcnn(num_classes):
    sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect = (0.2, 0.33, 0.5, 1.0, 2.0, 3.0, 5.0)  # long/thin friendly
    anchor_generator = AnchorGenerator(sizes=sizes,
                                       aspect_ratios=(aspect, aspect, aspect, aspect, aspect))

    model = fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        rpn_anchor_generator=anchor_generator,
        rpn_nms_thresh=0.5,
        box_score_thresh=0.35,
        box_nms_thresh=0.30,
        box_detections_per_img=75,
        box_fg_iou_thresh=0.60,
        box_bg_iou_thresh=0.40,
        box_batch_size_per_image=512,
        box_positive_fraction=0.20,
        min_size=1000,
        max_size=1800,
    )

    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
    return model

# ---------------- Utilities ----------------

def load_class_names(yaml_path):
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    names = cfg["names"]
    if isinstance(names, dict):
        class_names = [names[i] for i in sorted(names.keys())]
    else:
        class_names = list(names)
    # Insert background at index 0 for model mapping
    return ["__background__"] + class_names

def list_images(root):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = []
    for d, _, fns in os.walk(root):
        for fn in fns:
            if os.path.splitext(fn.lower())[1] in exts:
                files.append(os.path.join(d, fn))
    return sorted(files)

def draw_boxes_pil(img_pil, boxes, scores, labels, class_names, score_thresh=0.5):
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    for b, s, l in zip(boxes, scores, labels):
        if s < score_thresh: 
            continue
        x1,y1,x2,y2 = b
        draw.rectangle([x1,y1,x2,y2], outline=(0,255,0), width=2)
        text = f"{class_names[l]} {s:.2f}" if l < len(class_names) else f"{l} {s:.2f}"
        tw, th = draw.textbbox((0,0), text, font=font)[2:]
        draw.rectangle([x1, y1-th-4, x1+tw+4, y1], fill=(0,255,0))
        draw.text((x1+2, y1-th-2), text, fill=(0,0,0), font=font)
    return img_pil

# ---------------- Inference ----------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    # classes
    class_names = load_class_names(args.yaml)
    num_classes = len(class_names)

    # model
    model = build_frcnn(num_classes)
    state = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(state, strict=False)  # strict=False is tolerant if heads match
    model.to(device).eval()
    print(f"[INFO] Loaded weights: {args.weights}")

    # input images
    img_files = list_images(args.images)
    if not img_files:
        print(f"[WARN] No images found under: {args.images}")
        return
    print(f"[INFO] Found {len(img_files)} images")

    # outputs
    vis_dir = os.path.join(args.out, "annotated")
    os.makedirs(vis_dir, exist_ok=True)
    pred_json_path = os.path.join(args.out, "predictions.json")
    preds_all = []

    with torch.no_grad():
        for i, path in enumerate(img_files, 1):
            im = Image.open(path).convert("RGB")
            tensor = FT.to_tensor(im).to(device)[None, ...]  # (1,C,H,W)

            out = model(tensor)[0]
            b = out["boxes"].detach().cpu().numpy()
            s = out["scores"].detach().cpu().numpy()
            l = out["labels"].detach().cpu().numpy()

            # filter then IoS+WBF
            keep = s >= args.score_thresh
            b, s, l = b[keep], s[keep], l[keep]
            b, s, l = ios_wbf_per_class(
                b, s, l,
                ios_thr=args.ios,
                p=args.wbf_p,
                score_fuse="max",
                strategy="fuse"
            )

            # save per-image predictions (xyxy)
            rel = os.path.relpath(path, args.images)
            preds_all.append({
                "image": rel.replace("\\", "/"),
                "detections": [
                    {"bbox_xyxy": [float(x) for x in bb], "score": float(sc), "label": int(lb),
                     "label_name": class_names[lb] if lb < len(class_names) else str(lb)}
                    for bb, sc, lb in zip(b, s, l)
                ]
            })

            # draw & save visualization
            vis = im.copy()
            vis = draw_boxes_pil(vis, b, s, l, class_names, score_thresh=args.score_thresh)
            out_path = os.path.join(vis_dir, os.path.basename(path))
            vis.save(out_path)

            if i % 50 == 0 or i == len(img_files):
                print(f"[{i}/{len(img_files)}] saved {out_path}")

    with open(pred_json_path, "w") as f:
        json.dump(preds_all, f, indent=2)
    print(f"[DONE] Annotated images -> {vis_dir}")
    print(f"[DONE] Predictions JSON -> {pred_json_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True, help="Path to trained .pth")
    ap.add_argument("--yaml",    type=str, required=True, help="YAML with class names (same used in training)")
    ap.add_argument("--images",  type=str, required=True, help="Folder with new images")
    ap.add_argument("--out",     type=str, default="predictions_out", help="Output folder")
    ap.add_argument("--score_thresh", type=float, default=0.50, help="Filter before IoS+WBF")
    ap.add_argument("--ios",     type=float, default=0.92, help="IoS threshold for clustering")
    ap.add_argument("--wbf_p",   type=float, default=3.0,  help="Weight exponent for WBF")
    args = ap.parse_args()
    main(args)
