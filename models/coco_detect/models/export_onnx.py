from ultralytics import YOLO
from ultralytics.nn.modules import Detect


class ESP_Detect(Detect):
    def forward(self, x):
        """Returns predicted bounding boxes and class probabilities respectively."""
        # self.nl = 3
        box0 = self.cv2[0](x[0])
        score0 = self.cv3[0](x[0])

        box1 = self.cv2[1](x[1])
        score1 = self.cv3[1](x[1])

        box2 = self.cv2[2](x[2])
        score2 = self.cv3[2](x[2])

        return box0, score0, box1, score1, box2, score2


model = YOLO("yolo11n.pt")
for m in model.modules():
    if isinstance(m, Detect):
        m.forward = ESP_Detect.forward.__get__(m)

model.export(format="onnx", simplify=True, opset=13, imgsz=640)
