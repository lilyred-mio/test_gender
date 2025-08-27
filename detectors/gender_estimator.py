# -*- coding: utf-8 -*-
import numpy as np
from deepface import DeepFace

class GenderEstimator:
    def __init__(self, conf_th: float = 0.85):
        self.conf_th = conf_th

    def infer(self, np_image):
        """
        returns (label, conf, face_detected)
          - label: 'M' / 'W' / 'U'
          - conf : 0~1
          - face_detected: True/False (DeepFace가 얼굴 영역을 잡았는지 힌트)
        """
        try:
            res = DeepFace.analyze(np_image, actions=['gender'], enforce_detection=False)
            item = res[0] if isinstance(res, list) else res

            gender_dict = item.get('gender', {})
            w = float(gender_dict.get('Woman', 0)) / 100.0
            m = float(gender_dict.get('Man', 0)) / 100.0
            conf = max(w, m)

            # 얼굴 검출 여부 추정
            region = item.get('region') or item.get('facial_area') or {}
            face_detected = bool(region) and (region.get('w', 0) * region.get('h', 0) > 0)

            if conf < self.conf_th:
                return 'U', conf, face_detected
            return ('W', w, face_detected) if w >= m else ('M', m, face_detected)

        except Exception:
            return 'U', 0.0, False
