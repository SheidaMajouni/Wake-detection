## How to run frcnn_train_ios_wbf.py

1. Install deps

    pip install torch torchvision pillow opencv-python pyyaml numpy


2. Prepare data

    Dataset/
      images/{train,valid,test}/...    # your spectrogram images
      faster_rcnn_annotations.json     # your COCO-like JSON (from your converter)
      vessel_wakes.yaml                # contains `names: {0: "wake"}`


3. Train + validate (with IoS+WBF during validation)

    python frcnn_train_ios_wbf.py \
      --dataset_root ../Dataset \
      --ann_json ../Dataset/faster_rcnn_annotations.json \
      --yaml ../Dataset/vessel_wakes.yaml \
      --out runs-fasterrcnn-ioswbf \
      --epochs 25 --batch 4


4. Best checkpoint is saved as:
runs-fasterrcnn-ioswbf/frcnn_YYYYMMDD-HHMMSS/best_f1_0.xxx_epochE.pth

###############################################################################

## How to run infer_frcnn_ios_wbf.py
Goal: run with your trained Faster R-CNN checkpoint to predict on new images, apply IoS + WBF (IoS=0.92, fused) to reduce duplicates, and save annotated outputs + a JSON of predictions.

1. Install

    pip install torch torchvision pillow numpy opencv-python pyyaml

2. Have these ready

    Trained checkpoint: best_model.pth

    YAML with class names (same used during training): vessel_wakes.yaml
    (e.g., names: {0: wake})

3. Put new images in a folder, e.g., frcnn_samples/ (subfolders OK).

4. Run

    python frcnn_infer_ios_wbf.py \
      --weights frcnn_best_model_trained.pth \
      --yaml ../Dataset/vessel_wakes.yaml \
      --images Dataset/frcnn_samples \
      --out predictions_frcnn_out

5. Outputs

    Annotated images saved to: predictions_out/annotated/
    JSON with all detections (xyxy boxes, scores, labels): predictions_out/predictions.json
    
    
    ls Dataset/images/valid | head -n 10 | xargs -I {} cp Dataset/images/valid/{} To_share_with_marine_labDatasetfrcnn_samples/
