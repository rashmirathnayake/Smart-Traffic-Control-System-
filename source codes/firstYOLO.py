from ultralytics import YOLO
import supervision as sv



model = YOLO("yolov8x.pt") #prtrained model


results = model.track(source="0",conf=0.3, iou=0.5, show=True, tracker="bytetrack.yaml",device="0")

print(results)