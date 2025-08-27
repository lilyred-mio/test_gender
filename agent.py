# -*- coding: utf-8 -*-
# GPU 비활성화(러너에서 CUDA 오류 방지)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_FORCE_CPU"] = "1"

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
    PERSON_AREA_FRAC_TH,
    is_woman_only,
)
from detectors.person_detector import PersonDetector
from detectors.gender_estimator import GenderEstimator

# ── HTTP 세션(재시도/Referer/우회) ────────────────────────────────────────
def _make_session():
    s = requests.Session()
    retries = Retry(total=2, backoff_factor=0.3,
                    status_forcelist=[429, 500, 502, 503, 504],
                    allowed_methods=["GET"], raise_on_status=False)
    adapter = HTTPAdapter(max_retries=retries)
    s.mount("http://", adapter); s.mount("https://", adapter)
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/124.0.0.0 Safari/537.36",
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Language": "ko,en;q=0.9",
    })
    return s
SESSION = _make_session()

def fetch_image(url: str, timeout: int = 8) -> Image.Image:
    r = SESSION.get(url, timeout=timeout)
    if r.status_code == 200:
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    ref = {"Referer": "https://www.musinsa.com/"}
    r = SESSION.get(url, timeout=timeout, headers=ref)
    if r.status_code == 200:
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    if "/thumbnails/" in url:
        alt = url.replace("/thumbnails", "")
        r = SESSION.get(alt, timeout=timeout)
        if r.status_code != 200:
            r = SESSION.get(alt, timeout=timeout, headers=ref)
        if r.status_code == 200:
            return Image.open(io.BytesIO(r.content)).convert("RGB")
    r.raise_for_status()

def pil_to_np(img: Image.Image):
    return np.array(img)[:, :, ::-1]

def _largest_person_area_frac(persons, img_w, img_h):
    """가장 큰 person 박스의 면적 비율(이미지 대비) 반환"""
    if not persons:
        return 0.0
    img_area = float(img_w * img_h)
    largest = 0.0
    for (x1, y1, x2, y2), _sc in persons:
        w = max(0.0, x2 - x1); h = max(0.0, y2 - y1)
        largest = max(largest, (w * h) / img_area)
    return largest

def decide_with_image(img: Image.Image, goods_nm: str, full_nm: str,
                      predicted_init: str,
                      person_det: PersonDetector, gender_est: GenderEstimator) -> Decision:
    # 0) 여성 전용 키워드 최우선
    if is_woman_only(goods_nm, full_nm):
        return Decision("W", "여성 전용 키워드(원피스/스커트/여성 등) → W")

    # 1) 모델(사람) 존재 강판정: 높은 score + (얼굴 검출 OR 큰 면적)
    persons_raw = person_det.detect(img)  # [(box, score)]
    persons = [p for p in persons_raw if p[1] >= PERSON_CONF_TH]

    img_w, img_h = img.size
    max_area_frac = _largest_person_area_frac(persons, img_w, img_h)

    # 얼굴 검출 여부(힌트)와 성별 추정
    np_img = pil_to_np(img)
    g_label, g_conf, face_detected = gender_est.infer(np_img)

    model_present = (len(persons) > 0) and (face_detected or max_area_frac >= PERSON_AREA_FRAC_TH)

    if model_present:
        # 성별 확신도 충분한 경우에만 덮어씀, 아니면 입력값 유지
        if g_label in ("M", "W") and g_conf >= GENDER_CONF_TH:
            return Decision(g_label, f"모델컷: 자동 성별 추정({g_label}, conf={g_conf:.2f})")
        else:
            return Decision(predicted_init, "모델컷: 성별 불확실 → 최초 입력 성별 유지")

    # 2) 모델 없음 → 규칙 기반
    return decide_no_model(goods_nm, full_nm)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=os.getenv('INPUT_CSV', 'sample_set.csv'))
    ap.add_argument('--out', default=os.getenv('OUTPUT_CSV', 'sample_set_labeled.csv'))
    ap.add_argument('--url-col', default='imgurl')
    ap.add_argument('--name-col', default='goods_nm')
    ap.add_argument('--cat-col', default='full_nm')
    ap.add_argument('--pred-col', default='gender')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if 'answer_reason' not in df.columns:
        df['answer_reason'] = ''

    total = len(df)
    limit = min(args.limit, total) if args.limit and args.limit > 0 else total
    work_idx = list(range(limit))

    person_det = PersonDetector(score_th=PERSON_CONF_TH, device="cpu")
    gender_est = GenderEstimator(conf_th=GENDER_CONF_TH)

    for idx in tqdm(work_idx, total=len(work_idx)):
        row = df.iloc[idx]
        goods_nm  = str(row.get(args.name_col, ''))
        full_nm   = str(row.get(args.cat_col, ''))
        img_url   = str(row.get(args.url_col, ''))
        predicted = str(row.get(args.pred_col, ''))

        try:
            img = fetch_image(img_url)
            decision = decide_with_image(img, goods_nm, full_nm, predicted, person_det, gender_est)
        except requests.HTTPError as e:
            status = getattr(e.response, "status_code", "unknown")
            decision = decide_no_model(goods_nm, full_nm)
            decision.reason = f"이미지 로드 실패(HTTP {status}) → {decision.reason}"
        except Exception:
            decision = decide_no_model(goods_nm, full_nm)
            decision.reason = f"이미지 로드 실패(기타) → {decision.reason}"

        df.at[idx, 'answer_reason'] = f"{decision.label} | {decision.reason}"

    df.to_csv(args.out, index=False, encoding='utf-8-sig')
    print("="*60); print(f"[DONE] Saved CSV: {args.out}"); print("="*60)

if __name__ == '__main__':
    main()
