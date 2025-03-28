from ultralytics import YOLO
from ultralytics.nn.modules import Detect, Attention

import torch


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


class ESP_Attention(Attention):
    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(
            -1, self.num_heads, self.key_dim * 2 + self.head_dim, N
        ).split([self.key_dim, self.key_dim, self.head_dim], dim=2)
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(-1, C, H, W) + self.pe(
            v.reshape(-1, C, H, W)
        )
        x = self.proj(x)
        return x


model = YOLO("yolo11n.pt")
for m in model.modules():
    if isinstance(m, Attention):
        m.forward = ESP_Attention.forward.__get__(m)
    if isinstance(m, Detect):
        m.forward = ESP_Detect.forward.__get__(m)


model.export(format="onnx", simplify=True, opset=13, imgsz=640)
