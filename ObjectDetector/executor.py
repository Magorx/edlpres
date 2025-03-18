import torch
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights as ModelWeights
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2 as ModelClass

from jina import Executor, requests
from docarray import DocList, BaseDoc
from docarray.typing import TorchTensor, ImageUrl
from typing import List, Optional


class ImageInput(BaseDoc):
    url: ImageUrl
    data: Optional[TorchTensor]


class DetectedObjects(BaseDoc):
    objects: List[str]


class ObjectDetector(Executor):
    def __init__(self, device='cpu', **kwargs):
        super().__init__(**kwargs)
        self.device = device
        print(f'Device: {device}')

        model = ModelClass(weights=ModelWeights.COCO_V1)
        model = model.to(device).eval()
        
        self.model = model
        self.transform = ModelWeights.COCO_V1.transforms()
        self.labels = ModelWeights.COCO_V1.meta['categories']
    
    @requests
    def fallback(self, **kwargs):
        print("SOMEONE GONA FALL")

    @requests(on='/detect')
    def generate(self, doc: ImageInput, **kwargs) -> DetectedObjects:
        doc.data = doc.url.load()
        doc.data = self.transform(doc.data).to(self.device).transpose(0, 2).transpose(1, 2).unsqueeze(0)

        output = self.model(doc.data)[0]
        objects = []
        for i in range(len(output['labels'])):
            if output['scores'][i] > 0.75:
                objects.append(self.labels[output['labels'][i]])
        
        return DetectedObjects(objects=objects)
