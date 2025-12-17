"""
Train Faster R-CNN for Wake Detection + Validate with IoS + WBF
----------------------------------------------------------------
- Expects a dataset folder like:
    Dataset/
      images/
        train/  *.jpg|*.png
        valid/  *.jpg|*.png
        test/   *.jpg|*.png   (optional for this script)
      faster_rcnn_annotations.json   (COCO-like, produced from your YOLO labels)
      vessel_wakes.yaml              (with names: {0: "wake"} etc.)

- Trains fasterrcnn_resnet50_fpn with long/thin-friendly anchors.
- Validates each epoch with IoS + WBF (IoS=0.92, fused) post-processing.
- Selects BEST checkpoint by F1 (IoU≥0.5) on the validation split.

Run:
  python frcnn_train_ios_wbf.py \
    --dataset_root ../Dataset \
    --ann_json ../Dataset/faster_rcnn_annotations.json \
    --yaml ../Dataset/vessel_wakes.yaml \
    --out runs-fasterrcnn-ioswbf

Dependencies:
  torch, torchvision, numpy, pillow, opencv-python, pyyaml
"""

import os, json, math, argparse, datetime
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as FT
from PIL import Image
import yaml

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator

# ============ Dataset ============

class WakeDetectionDataset(Dataset):
    """
    Loads images and annotations from the COCO-like JSON you generated earlier.
    JSON must contain:
      - images: [{id, file_name, width, height}, ...]
      - annotations: [{image_id, bbox[x,y,w,h], category_id, iscrowd, area}, ...]
    We will convert boxes to [x1,y1,x2,y2].
    """
    def __init__(self, root_dir, annotation_file, split, transform=None):
        self.root_dir = root_dir
        self.split = split     # 'train' or 'valid'
        self.transform = transform

        with open(annotation_file, "r") as f:
            all_images = json.load(f)

        # Filter by split (we assume file_name starts with images/<split>/...)
        expected_prefix = os.path.join("images", split) + os.sep
        self.items = [im for im in all_images if im.get("file_name", "").startswith(expected_prefix)]

        # Build a quick map id->image dict
        self.id_to_item = {im["image_id"]: im for im in self.items}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        data = self.items[idx]
        img_id = data["image_id"]
        img_path = os.path.join(self.root_dir, data["file_name"])
        image = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in data["annotations"]:
            # ann["bbox"] is [xmin, ymin, xmax, ymax] in your JSON
            x1, y1, x2, y2 = ann["bbox"]
            boxes.append([x1, y1, x2, y2])
            labels.append(ann["category_id"])  # 1-indexed (0 is background in the model)
            areas.append(ann["area"])
            iscrowd.append(ann.get("iscrowd", 0))

        boxes = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0,4), dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32) if areas else torch.zeros((0,), dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64) if iscrowd else torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
            "area": areas,
            "iscrowd": iscrowd,
        }

        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target

def collate_fn(batch):
    return tuple(zip(*batch))

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor:
    def __call__(self, image, target):
        return FT.to_tensor(image), target

def get_transform(is_train):
    # You can add Albumentations here if you want train-time aug.
    return Compose([ToTensor()])


# ============ Model ============

def build_frcnn(num_classes):
    """
    Faster R-CNN (ResNet-50 FPN) with long/thin-friendly anchor aspect ratios.
    """
    sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect = (0.2, 0.33, 0.5, 1.0, 2.0, 3.0, 5.0)
    anchor_generator = AnchorGenerator(sizes=sizes,
                                       aspect_ratios=(aspect, aspect, aspect, aspect, aspect))

    model = fasterrcnn_resnet50_fpn(
        weights=None,  # no COCO head to avoid mismatch
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


# ============ IoS + WBF (post-processing for validation) ============

def _area(b):  # [x1,y1,x2,y2]
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


# ============ Matching-based metrics (IoU≥0.5) ============

def iou_xyxy(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    Aa = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    Ab = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    den = Aa + Ab - inter + 1e-9
    return inter / den

def match_and_scores(gt_boxes, pred_boxes, iou_thr=0.5):
    TP, FP, FN = 0, 0, 0
    matched_gt = set(); matched_pred = set()
    for i, p in enumerate(pred_boxes):
        for j, g in enumerate(gt_boxes):
            if j in matched_gt: continue
            if iou_xyxy(p, g) >= iou_thr:
                TP += 1; matched_pred.add(i); matched_gt.add(j); break
    FP += len(pred_boxes) - len(matched_pred)
    FN += len(gt_boxes) - len(matched_gt)
    return TP, FP, FN

def prf1(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


# ============ Train + Validate ============

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    run_name = datetime.datetime.now().strftime("frcnn_%Y%m%d-%H%M%S")
    out_dir = os.path.join(args.out, run_name)
    os.makedirs(out_dir, exist_ok=True)

    # infer classes from YAML (names)
    with open(args.yaml, "r") as f:
        cfg = yaml.safe_load(f)
    names = cfg["names"]
    if isinstance(names, dict):
        class_names = [names[i] for i in sorted(names.keys())]
    else:
        class_names = list(names)
    num_classes = len(class_names) + 1  # + background
    print(f"[INFO] Classes: {class_names}  -> num_classes={num_classes}")

    # datasets/loaders
    train_ds = WakeDetectionDataset(args.dataset_root, args.ann_json, "train", get_transform(True))
    val_ds   = WakeDetectionDataset(args.dataset_root, args.ann_json, "valid", get_transform(False))
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=max(1, os.cpu_count()//2), collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                              num_workers=max(1, os.cpu_count()//2), collate_fn=collate_fn)
    print(f"[INFO] train: {len(train_ds)}  |  valid: {len(val_ds)}")

    # model/opt/sched
    model = build_frcnn(num_classes).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_f1, best_path = -1.0, None
    for epoch in range(1, args.epochs+1):
        model.train()
        running = 0.0
        for step, (images, targets) in enumerate(train_loader, 1):
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k,v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            running += loss.item()
            if step % 100 == 0:
                print(f"[Epoch {epoch} | {step}/{len(train_loader)}] loss={loss.item():.4f}")
        scheduler.step()
        print(f"[Epoch {epoch}] train_loss={running/len(train_loader):.4f}")

        # ----- validation with IoS + WBF -----
        model.eval()
        TP=FP=FN=0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [im.to(device) for im in images]
                outputs = model(images)

                for k, out in enumerate(outputs):
                    # GT
                    gt_boxes = targets[k]["boxes"].cpu().numpy()
                    # PRED (apply score threshold first)
                    b = out["boxes"].detach().cpu().numpy()
                    s = out["scores"].detach().cpu().numpy()
                    l = out["labels"].detach().cpu().numpy()
                    keep = s >= args.score_thresh
                    b, s, l = b[keep], s[keep], l[keep]
                    # IoS+WBF (0.92, fused) per class
                    b, s, l = ios_wbf_per_class(b, s, l, ios_thr=args.ios, p=args.wbf_p,
                                                score_fuse="max", strategy="fuse")
                    # Match
                    tp, fp, fn = match_and_scores(gt_boxes, b, iou_thr=args.match_iou)
                    TP += tp; FP += fp; FN += fn

        prec, rec, f1 = prf1(TP, FP, FN)
        print(f"[Epoch {epoch}] val  TP={TP} FP={FP} FN={FN}  |  P={prec:.3f} R={rec:.3f} F1={f1:.3f}")

        # save best by F1
        if f1 > best_f1:
            best_f1 = f1
            best_path = os.path.join(out_dir, f"best_f1_{f1:.3f}_epoch{epoch}.pth")
            torch.save(model.state_dict(), best_path)
            print(f"[SAVE] {best_path}")

        # periodic checkpoint
        if epoch % 5 == 0:
            ckpt = os.path.join(out_dir, f"checkpoint_epoch{epoch}.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"[CKPT] {ckpt}")

    print(f"[DONE] best F1={best_f1:.3f}  saved at: {best_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=str, required=True)
    ap.add_argument("--ann_json",     type=str, required=True)
    ap.add_argument("--yaml",         type=str, required=True)
    ap.add_argument("--out",          type=str, default="runs-frcnn-ioswbf")
    ap.add_argument("--epochs",       type=int, default=20)
    ap.add_argument("--batch",        type=int, default=4)
    ap.add_argument("--lr",           type=float, default=0.005)
    ap.add_argument("--momentum",     type=float, default=0.9)
    ap.add_argument("--wd",           type=float, default=5e-4)

    # validation / postproc
    ap.add_argument("--score_thresh", type=float, default=0.50)   # filter before IoS+WBF
    ap.add_argument("--ios",          type=float, default=0.92)   # IoS threshold
    ap.add_argument("--wbf_p",        type=float, default=3.0)    # weight exponent
    ap.add_argument("--match_iou",    type=float, default=0.50)   # matching threshold
    args = ap.parse_args()
    main(args)
