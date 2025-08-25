# -*- coding: utf-8 -*-
import torch
import torchvision
from torchvision.transforms.functional import to_tensor

class PersonDetector:
    def __init__(self, score_th: float = 0.6, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT").to(self.device).eval()
        self.score_th = score_th

    @torch.inference_mode()
    def detect(self, pil_image):
        img = to_tensor(pil_image).to(self.device)
        outputs = self.model([img])[0]
        labels = outputs["labels"].detach().cpu().numpy()
        scores = outputs["scores"].detach().cpu().numpy()
        boxes  = outputs["boxes"].detach().cpu().numpy()
        persons = []
        for lbl, sc, box in zip(labels, scores, boxes):
            # COCO 클래스 1 == person
            if lbl == 1 and sc >= self.score_th:
                persons.append((box, float(sc)))
        return persons
