from typing import Iterable, Tuple
import torch
from esp_ppq.executor import TorchExecutor
from esp_ppq.IR import BaseGraph, TrainableGraph
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
from ultralytics.nn.tasks import PoseModel
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
from yolo11n_pose_eval import make_quant_validator_class
from ultralytics import YOLO

imgsz = 640


class CaliDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((imgsz, imgsz)),
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

    def __init__(self, graph: BaseGraph, device: str = "cuda") -> None:

        self._epoch = 0
        self._step = 0
        self._best_metric = 0
        self._device = device
        self._executor = TorchExecutor(graph, device=self._device)
        self._training_graph = TrainableGraph(graph)
        self.graph = graph

        for tensor in self._training_graph.parameters():
            tensor.requires_grad = True
        self._optimizer = torch.optim.SGD(
            params=[{"params": self._training_graph.parameters()}],
            lr=1e-6,
            momentum=0.937,
            weight_decay=5e-4,
        )
        self._lr_scheduler = None

        self.model = PoseModel(nc=1)
        self.model = self.model.to(self._device)
        overrides = {
            "data": DEFAULT_CFG_DICT["data"],
            "model": "yolo11n-pose",
            "task": "pose",
            "device": "cuda",
        }
        cfg = {**DEFAULT_CFG_DICT, **overrides}
        self.model.args = IterableSimpleNamespace(**cfg)

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        batch["img"] = batch["img"].to(self._device, non_blocking=True).float() / 255
        return batch

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

            _, loss = self.step(batch, True)
            epoch_loss += loss

        self._epoch += 1

        print(f"Epoch Loss: {epoch_loss / len(dataloader):.4f}")
        return epoch_loss

    def step(self, data: torch.Tensor, training: bool) -> Tuple[torch.Tensor, float]:
        """
        Do one step Training with given data(torch.Tensor) and label(torch.Tensor).
        """
        # supervised training
        if training:
            data = self.preprocess_batch(data)
            graph_outputs = self._executor.forward_with_gradient(data["img"])

            x = [
                torch.cat((graph_outputs[i], graph_outputs[i + 1]), 1)
                for i in range(0, 6, 2)
            ]
            bs = x[0].shape[0]
            kpt = torch.cat(
                [graph_outputs[i].view(bs, 51, -1) for i in range(6, 9)], dim=-1
            )
            preds = (x, kpt)

            # loss
            loss = self.model.loss(data, preds)[0]

            loss.backward()
            if self._lr_scheduler is not None:
                self._lr_scheduler.step(epoch=self._epoch)
            self._optimizer.step()
            self._training_graph.zero_grad()

            self._step += 1
            return preds, loss.item()

    def eval(self) -> float:
        """Do Evaluation process on given dataloader.

        Split your dataset into training and evaluation dataset at first, then
            use eval function to monitor model performance on evaluation dataset.
        """
        QuantPoseValidator = make_quant_validator_class(self._executor)
        # load yolo11n-pose.pt to enter val method and run BaseGraph inference
        model = YOLO("yolo11n-pose.pt")
        results = model.val(
            data="coco-pose.yaml",  # change to "coco-pose.yaml"
            imgsz=640,
            device=self._device,
            validator=QuantPoseValidator,
            rect=False,
            save_json=True,
        )
        return results.pose.map

    def save(self, file_path: str, file_path2: str):
        """Save model to given path.
        Saved model can be read by esp_ppq.api.load_native_model function.
        """

        from esp_ppq.core import TargetPlatform
        import esp_ppq.lib as PFL

        PFL.Exporter(platform=TargetPlatform.ESPDL_INT8).export(
            file_path=file_path, graph=self.graph
        )
        final_graph = self.graph

        from esp_ppq.parser import NativeExporter

        exporter = NativeExporter()
        exporter.export(file_path=file_path2, graph=self.graph)
        return final_graph

    def clear(self):
        """Clear training state."""
        for tensor in self._training_graph.parameters():
            tensor.requires_grad = False
            tensor._grad = None
