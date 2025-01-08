# Object detection in images using YOLO11
from ultralytics import YOLO

# Load a pretrained YOLO detection model
model = YOLO("yolo11n.pt")

# # Train the model
#model.train(data = 'coco8.yaml', epochs=3)

results = model('C:\\Users\\SUNITHA\\A VS CODE\\PYDEVCODE\\YOLO\\img\\1.jpg', save = True)
#results = model('C:\\Users\\SUNITHA\\A VS CODE\\PYDEVCODE\\YOLO\\video\\video1.mp4', save = True)
results[0].show()