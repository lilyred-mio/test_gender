# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Tuple

WOMAN_KEYS = ["WOMAN", "WOMEN", "WOMAN VER", "WOMEN VER", "여성", "우먼", "레이디", "걸", "여아"]
MAN_KEYS   = ["MAN", "MEN", "남성", "맨즈", "남아", "보이"]
UNISEX_KEYS= ["UNISEX", "공용"]
WOMAN_ONLY_CATS = [
    "원피스", "스커트", "여성",
    "woman", "women", "dress", "skirt"
]

# 카테고리에 'Beauty/뷰티', 'Life/라이프/리빙/홈/주방/가전/디지털/Tech' 포함 시 L 허용
BEAUTY_KEYS = ["BEAUTY", "뷰티", "스킨케어", "메이크업", "향수", "바디", "헤어"]
LIFE_KEYS   = ["LIFE", "라이프", "리빙", "홈", "주방", "가전", "디지털", "DIGITAL", "TECH"]

# 신뢰도 임계치
PERSON_CONF_TH = 0.60  # 사람 검출 신뢰도(라벨=person)
GENDER_CONF_TH = 0.60  # gender 확신도(DeepFace) 부족 시 U

@dataclass
class Decision:
    label: str  # M/W/U/L
    reason: str # 한국어 간단 근거


def has_any(text: str, keys) -> bool:
    up = (text or "").upper()
    return any(k in up for k in keys)

def is_woman_only(goods_nm: str, full_nm: str) -> bool:
    full = f"{goods_nm} || {full_nm}".lower()  # 소문자로 변환
    return any(k in full for k in WOMAN_ONLY_CATS)

def decide_no_model(goods_nm: str, full_nm: str) -> Decision:
    full = f"{goods_nm} || {full_nm}".upper()
    if has_any(full, BEAUTY_KEYS) or has_any(full, LIFE_KEYS):
        return Decision("L", "모델 없음 + Beauty/Life 카테고리 → L")
    if is_woman_only(goods_nm, full_nm):
        return Decision("W", "여성 전용 키워드 매칭 → W")
    if has_any(full, WOMAN_KEYS):
        return Decision("W", "모델 없음 + 상품명/경로 여성 단서 → W")
    if has_any(full, MAN_KEYS):
        return Decision("M", "모델 없음 + 상품명/경로 남성 단서 → M")
    if has_any(full, UNISEX_KEYS):
        return Decision("U", "모델 없음 + 상품명/경로 공용 단서 → U")
    return Decision("U", "모델 없음 + 성별 단서 없음 → U")
