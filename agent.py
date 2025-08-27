# -*- coding: utf-8 -*-
import os
import io
import argparse
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from rules import (
    Decision,
    decide_no_model,
    PERSON_CONF_TH,
    GENDER_CONF_TH,
    is_woman_only,   # 여성 전용 키워드 최우선 규칙
)
from detectors.person_detector import PersonDetector
from detectors.gender_estimator import GenderEstimator

# ---------------------------
# HTTP 세션 (재시도 + 헤더 보강)
# ---------------------------
def _make_session():
    s = requests.Session()
    retries = Retry(
        total=2,               # 재시도 축소 (빠른 실패/진단)
        backoff_factor=0.3,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Language": "ko,en;q=0.9",
