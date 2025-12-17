import argparse
from pathlib import Path
from ultralytics import YOLO

def train_model(model_path, config_path):
    # Load a model
    model = YOLO(str(model_path))  # load a pretrained model (recommended for training)

    # Train the model using the 'coco8.yaml' dataset for 3 epochs
    results = model.train(data=str(config_path), epochs=300)

    # Evaluate the model's performance on the validation set
    results = model.val()

    print(results)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an object detection model using YoloV11.")
    parser.add_argument(
        "-m",
        "--model_path",
        type=Path,
        required=True,
        help="Path to pretrained YOLO model",
    )
    parser.add_argument(
        "-c",
        "--config_path",
        type=Path,
        required=True,
        help="Path to YAML config file."
    )

    args = parser.parse_args()
    params = {}
    for key, value in vars(args).items():
        params[key] = value

    train_model(
        model_path=args.model_path,
        config_path=args.config_path
    )

    print("\nDone!")