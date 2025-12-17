from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

# Train the model using the 'coco8.yaml' dataset for 3 epochs
results = model.train(data="Dataset/vessel_wakes.yaml", epochs=300)

# Evaluate the model's performance on the validation set
results = model.val()