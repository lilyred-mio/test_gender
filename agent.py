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
    })
    return s

SESSION = _make_session()

def fetch_image(url: str, timeout: int = 8) -> Image.Image:
    """
    이미지 다운로드 안정화:
      1) 기본 GET
      2) 실패 시 Referer 추가 재시도
      3) 실패 시 thumbnails/ 제거한 대체 경로 재시도
    실패 시 HTTPError 발생
    """
    # 1) 기본 시도
    r = SESSION.get(url, timeout=timeout)
    if r.status_code == 200:
        return Image.open(io.BytesIO(r.content)).convert("RGB")

    # 2) Referer 추가
    ref_headers = {"Referer": "https://www.musinsa.com/"}
    r = SESSION.get(url, timeout=timeout, headers=ref_headers)
    if r.status_code == 200:
        return Image.open(io.BytesIO(r.content)).convert("RGB")

    # 3) 썸네일 경로 우회
    if "/thumbnails/" in url:
        alt = url.replace("/thumbnails", "")
        r = SESSION.get(alt, timeout=timeout)
        if r.status_code != 200:
            r = SESSION.get(alt, timeout=timeout, headers=ref_headers)
        if r.status_code == 200:
            return Image.open(io.BytesIO(r.content)).convert("RGB")

    # 모두 실패 → 예외
    r.raise_for_status()

def pil_to_np(img: Image.Image):
    # DeepFace(OpenCV) 호환을 위해 RGB→BGR
    return np.array(img)[:, :, ::-1]

def decide_with_image(
    img: Image.Image,
    goods_nm: str,
    full_nm: str,
    predicted_init: str,
    person_det: PersonDetector,
    gender_est: GenderEstimator
) -> Decision:
    """
    우선순위:
      0) 여성 전용 키워드(원피스/스커트/여성/woman/women/dress/skirt) → 무조건 W
      1) 모델 감지됨 →
           - 성별 추정 확신(>= GENDER_CONF_TH) 있으면 M/W
           - 불확실하면 '최초 입력 성별(predicted_init)' 사용
      2) 모델 없음 → 카테고리/키워드 규칙(decide_no_model)
    """
    if is_woman_only(goods_nm, full_nm):
        return Decision("W", "여성 전용 키워드 매칭(원피스/스커트/여성/woman/women/dress/skirt) → W")

    persons = person_det.detect(img)
    if len(persons) > 0:
        np_img = pil_to_np(img)
        g_label, g_conf = gender_est.infer(np_img)

        if g_label in ("M", "W") and g_conf >= GENDER_CONF_TH:
            return Decision(g_label, f"모델 착용컷: 자동 성별 추정({g_label})")
        else:
            return Decision(predicted_init, "모델컷: 성별 불확실 → 최초 입력 성별 사용")

    return decide_no_model(goods_nm, full_nm)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=os.getenv('INPUT_CSV', 'sample_set.csv'))
    ap.add_argument('--out', default=os.getenv('OUTPUT_CSV', 'sample_set_labeled.csv'))
    ap.add_argument('--url-col', default='imgurl')
    ap.add_argument('--name-col', default='goods_nm')
    ap.add_argument('--cat-col', default='full_nm')
    ap.add_argument('--pred-col', default='gender')
    ap.add_argument('--limit', type=int, default=0, help='처리할 최대 행 수(테스트용). 전체 실행 시 크게 올리거나 제거.')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if 'answer_reason' not in df.columns:
        df['answer_reason'] = ''

    # 부분 실행을 위해 슬라이스
    total = len(df)
    limit = min(args.limit, total) if args.limit and args.limit > 0 else total
    work = df.iloc[:limit].copy()

    print(f"[INFO] total={total}, limit={limit}")

    # 모델 준비
    person_det = PersonDetector(score_th=PERSON_CONF_TH)
    gender_est = GenderEstimator(conf_th=GENDER_CONF_TH)

    for idx, row in tqdm(work.iterrows(), total=len(work)):
        goods_nm  = str(row.get(args.name_col, ''))
        full_nm   = str(row.get(args.cat_col, ''))
        img_url   = str(row.get(args.url_col, ''))
        predicted = str(row.get(args.pred_col, ''))

        print(f"[START] row={idx} name={goods_nm[:40]!r}")
        print(f"[FETCH] {img_url}")

        try:
            img = fetch_image(img_url)
            decision = decide_with_image(
                img, goods_nm, full_nm, predicted,
                person_det, gender_est
            )
        except requests.HTTPError as e:
            status = getattr(e.response, "status_code", "unknown")
            decision = decide_no_model(goods_nm, full_nm)
            decision.reason = f"이미지 로드 실패(HTTP {status}) → {decision.reason}"
        except Exception as e:
            decision = decide_no_model(goods_nm, full_nm)
            decision.reason = f"이미지 로드 실패(기타) → {decision.reason}"

        df.at[idx, 'answer_reason'] = f"{decision.label} | {decision.reason}"

    # 저장 (워크플로에서 절대경로 업로드 권장)
    df.to_csv(args.out, index=False, encoding='utf-8-sig')
    print("="*60)
    print(f"[DONE] Saved CSV: {args.out}")
    print("="*60)

if __name__ == '__main__':
    main()
