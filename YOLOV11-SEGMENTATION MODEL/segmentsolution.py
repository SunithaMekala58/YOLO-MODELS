# Instance Segementation model using YOLO-V11
from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-seg.pt")

# Train the model
train_results = model.train(
    data="C:\\Users\\SUNITHA\\A VS CODE\\PYDEVCODE\\BRAIN-TUMOR11SEG\\data.yaml",  # path to dataset YAML
    epochs=3,  # number of training epochs
    imgsz=640,  # training image size
    device="cpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)