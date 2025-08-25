# -*- coding: utf-8 -*-
import numpy as np
from deepface import DeepFace

class GenderEstimator:
    def __init__(self, conf_th: float = 0.6):
        self.conf_th = conf_th

    def infer(self, np_image):
        """
        returns (label, conf)
        label: 'M' or 'W' or 'U' (when low confidence)
        conf : confidence score in [0,1]
        """
        try:
            # DeepFace: actions=['gender'] 로 분석. 여러 얼굴이 있으면 첫 얼굴 기준.
            # enforce_detection=False: 얼굴 검출 실패 시 예외 대신 진행.
            res = DeepFace.analyze(np_image, actions=['gender'], enforce_detection=False)
            # DeepFace 0.0.93: res가 dict 또는 list(dict)일 수 있음.
            item = res[0] if isinstance(res, list) else res
            gender_dict = item.get('gender', {})
            # 예: {'Woman': 95.0, 'Man': 5.0}
            w = float(gender_dict.get('Woman', 0)) / 100.0
            m = float(gender_dict.get('Man', 0)) / 100.0
            if max(w, m) < self.conf_th:
                return 'U', max(w, m)
            return ('W', w) if w >= m else ('M', m)
        except Exception:
            return 'U', 0.0
