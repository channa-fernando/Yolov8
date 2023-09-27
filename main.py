from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)
results = model.train(data='C:\\Users\\Channa CC\\PycharmProjects\\YoloV8\\Dataset', epochs=200, imgsz=640)
