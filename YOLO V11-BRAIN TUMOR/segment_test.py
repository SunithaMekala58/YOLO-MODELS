# Testing the images

from ultralytics import YOLO

model = YOLO("C:\\Users\\SUNITHA\\A VS CODE\\runs\\segment\\train\\weights\\best.pt")

results = model("C:\\Users\\SUNITHA\\A VS CODE\\PYDEVCODE\\BRAIN-TUMOR11SEG\\test_images\1.jpg", save = True)
#results = model("C:\\Users\\SUNITHA\\A VS CODE\\PYDEVCODE\\BRAIN-TUMOR11SEG\\test_images\2.jpg", save = True)
#results = model("C:\\Users\\SUNITHA\\A VS CODE\\PYDEVCODE\\BRAIN-TUMOR11SEG\\test_images\3.jpg", save = True)

#results[0].show()
