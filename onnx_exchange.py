from ultralytics import YOLO
model = YOLO(r"C:\Users\Owner\PycharmProjects\YOLO_nail_seg\runs\segment\runs\segment\nail_seg_v1\weights\best.pt")
model.export(format="onnx", imgsz=[160, 320], simplify=False)