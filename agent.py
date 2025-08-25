# -*- coding: utf-8 -*-
import os
import io
import csv
import time
import argparse
import requests
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from rules import Decision, decide_no_model, PERSON_CONF_TH, GENDER_CONF_TH
from detectors.person_detector import PersonDetector
from detectors.gender_estimator import GenderEstimator

HEADERS = {"User-Agent": "Mozilla/5.0"}


def fetch_image(url: str, timeout: int = 20) -> Image.Image:
    r = requests.get(url, timeout=timeout, headers=HEADERS)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    return img


def pil_to_np(img: Image.Image):
    return np.array(img)[:, :, ::-1]  # RGB→BGR (DeepFace 내부 OpenCV 호환)


def decide_with_image(img: Image.Image, goods_nm: str, full_nm: str,
                      person_det: PersonDetector, gender_est: GenderEstimator) -> Decision:
    # 1) 사람(모델) 존재 여부
    persons = person_det.detect(img)
    if len(persons) > 0:
        # 1-1) 성별 추정 (낮은 신뢰도면 U)
        np_img = pil_to_np(img)
        g_label, g_conf = gender_est.infer(np_img)
        if g_label in ('M', 'W') and g_conf >= GENDER_CONF_TH:
            if g_label == 'M':
                return Decision('M', '모델 착용컷: 자동 성별 추정(남성)')
            else:
                return Decision('W', '모델 착용컷: 자동 성별 추정(여성)')
        else:
            return Decision('U', '모델 착용컷: 자동 추정 불확실 → U')

    # 2) 모델 없음 → 상품명/카테고리 규칙
    return decide_no_model(goods_nm, full_nm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=os.getenv('INPUT_CSV', 'sample_set.csv'))
    ap.add_argument('--out', default=os.getenv('OUTPUT_CSV', 'sample_set_labeled.csv'))
    ap.add_argument('--url-col', default='imgurl')
    ap.add_argument('--name-col', default='goods_nm')
    ap.add_argument('--cat-col', default='full_nm')
    ap.add_argument('--pred-col', default='gender')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if 'answer_reason' not in df.columns:
        df['answer_reason'] = ''

    person_det = PersonDetector(score_th=PERSON_CONF_TH)
    gender_est = GenderEstimator(conf_th=GENDER_CONF_TH)

    rows = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        goods_nm = str(row.get(args.name_col, ''))
        full_nm  = str(row.get(args.cat_col, ''))
        img_url  = str(row.get(args.url_col, ''))
        predicted= str(row.get(args.pred-col, ''))

        try:
            img = fetch_image(img_url)
            decision = decide_with_image(img, goods_nm, full_nm, person_det, gender_est)
        except Exception as e:
            decision = decide_no_model(goods_nm, full_nm)
            decision.reason = f"이미지 로드 실패 → {decision.reason}"

        df.at[idx, 'answer_reason'] = f"{decision.label} | {decision.reason}"

    df.to_csv(args.out, index=False, encoding='utf-8-sig')
    print(f"[DONE] Saved: {args.out}")

if __name__ == '__main__':
    main()
