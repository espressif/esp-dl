from typing import Iterable, Tuple
import numbers
import torch
from ppq.executor import TorchExecutor
from ppq.IR import BaseGraph, TrainableGraph
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
import numpy as np
from PIL import Image
from torch.nn.functional import linear, normalize
from typing import Callable
from ultralytics import YOLO
import onnx
import onnxruntime as ort
from ppq.core import TargetPlatform
import ppq.lib as PFL
from ppq.parser import NativeExporter
from yolo11n_eval import make_quant_validator_class


class CaliDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((640, 640)),
                transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
            ]
        )

        self.imgs_path = []
        self.path = path
        for img_name in os.listdir(self.path):
            img_path = os.path.join(self.path, img_name)
            self.imgs_path.append(img_path)

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, idx):
        img = Image.open(self.imgs_path[idx])  # 0~255 hwc #RGB
        img = self.transform(img)
        return img


class TrainDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((640, 640)),
                transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
            ]
        )
        self.imgs_path = []
        self.path = path
        for img_name in os.listdir(self.path):
            if img_name.endswith(".jpg"):
                img_path = os.path.join(self.path, img_name)
                self.imgs_path.append(img_path)

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, idx):
        img = Image.open(self.imgs_path[idx])
        if img.mode == "L":
            img = img.convert("RGB")
        img = self.transform(img)
        return img


class Trainer:
    """
    Member Functions:
    1. epoch(): do one epoch training.
    2. step(): do one step training.
    3. eval(): evaluation on given dataset.
    4. save(): save trained model.
    5. clear(): clear training cache.

    PPQ will create a TrainableGraph on you graph, a wrapper class that
        implements a set of useful functions that enable training. You are recommended
        to edit its code to add new feature on graph.

    Optimizer controls the learning process and determine the parameters values ends up learning,
        you can rewrite the defination of optimizer and learning scheduler in __init__ function.
        Tuning them carefully, as hyperparameters will greatly affects training result.
    """

    def __init__(self, graph: BaseGraph, onnx_path, device: str = "cpu") -> None:

        self._epoch = 0
        self._step = 0
        self._executor = TorchExecutor(graph, device=device)
        self._training_graph = TrainableGraph(graph)
        self._loss_fn = torch.nn.MSELoss()
        self._graph = graph
        self._onnx_path = onnx_path
        self._device = device

        for tensor in self._training_graph.parameters():
            tensor.requires_grad = True
        self._optimizer = torch.optim.SGD(
            params=[{"params": self._training_graph.parameters()}], lr=3e-5
        )
        self._lr_scheduler = None

    def epoch(self, dataloader: Iterable) -> float:
        """Do one epoch Training with given dataloader.

        Given dataloader is supposed to be a iterable container of batched data,
            for example it can be a list of [img(torch.Tensor), label(torch.Tensor)].

        If your data has other layout that is not supported by this function,
            then you are supposed to rewrite the logic of epoch function by yourself.
        """
        epoch_loss = 0
        for bidx, batch in enumerate(
            tqdm(dataloader, desc=f"Epoch {self._epoch}: ", total=len(dataloader))
        ):
            data = batch
            data = data.to(self._device)
            _, loss = self.step(data, True)
            epoch_loss += loss

        self._epoch += 1
        print(f"Epoch Loss: {epoch_loss / len(dataloader):.4f}")
        return epoch_loss

    def step(self, data: torch.Tensor, training: bool) -> Tuple[torch.Tensor, float]:
        """Do one step Training with given data(torch.Tensor) and label(torch.Tensor).

        This one-step-forward function assume that your model have only one input and output variable.

        If the training model has more input or output variable, then you might need to
            rewrite this function by yourself.
        """
        if training:
            # quantized model inference
            pred = self._executor.forward_with_gradient(data)
            # fp32 model inference
            session = ort.InferenceSession(self._onnx_path)
            input_name = session.get_inputs()[0].name
            output_fp32 = session.run(None, {input_name: np.array(data.cpu())})
            # init QAT training loss
            loss = 0
            for i in range(len(pred)):
                loss += self._loss_fn(
                    pred[i], torch.Tensor(output_fp32[i]).to(self._device)
                )
            loss.backward()
            if self._lr_scheduler is not None:
                self._lr_scheduler.step(epoch=self._epoch)
            self._optimizer.step()
            self._training_graph.zero_grad()

            self._step += 1
            return pred, loss.item()

    def eval(self) -> float:
        """Do Evaluation process on given dataloader.

        Split your dataset into training and evaluation dataset at first, then
            use eval function to monitor model performance on evaluation dataset.
        """
        QuantDetectionValidator = make_quant_validator_class(self._executor)
        # load yolo11n.pt to enter val method and run BaseGraph inference
        model = YOLO("yolo11n.pt")
        results = model.val(
            data="coco.yaml",  # change to "coco.yaml"
            imgsz=640,
            device=self._device,
            validator=QuantDetectionValidator,
            rect=False,
            save_json=True,
        )

        return results.box.map

    def save(self, file_path: str, file_path2: str):
        """Save model to given path.
        Saved model can be read by ppq.api.load_native_model function.
        """
        # export .espdl
        PFL.Exporter(platform=TargetPlatform.ESPDL_INT8).export(
            file_path=file_path, graph=self._graph
        )
        qat_graph = self._graph
        # export .native
        exporter = NativeExporter()
        exporter.export(file_path=file_path2, graph=self._graph)
        return qat_graph

    def clear(self):
        """Clear training state."""
        for tensor in self._training_graph.parameters():
            tensor.requires_grad = False
            tensor._grad = None
