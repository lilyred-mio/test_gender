# -*- coding: utf-8 -*-
from dataclasses import dataclass

# ── 키워드들 ─────────────────────────────────────────────────────────────
WOMAN_KEYS   = ["WOMAN", "WOMEN", "WOMAN VER", "WOMEN VER", "여성", "우먼", "레이디", "걸", "여아"]
MAN_KEYS     = ["MAN", "MEN", "남성", "맨즈", "남아", "보이"]
UNISEX_KEYS  = ["UNISEX", "공용"]

# 여성 전용(최우선 강제 W) – 간소화 + 영문 포함
WOMAN_ONLY_CATS = ["원피스", "스커트", "여성", "woman", "women", "dress", "skirt"]

# 카테고리 기반 L 허용
BEAUTY_KEYS = ["BEAUTY", "뷰티", "스킨케어", "메이크업", "향수", "바디", "헤어"]
LIFE_KEYS   = ["LIFE", "라이프", "리빙", "홈", "주방", "가전", "디지털", "DIGITAL", "TECH"]

# ── 임계치(요청 반영) ─────────────────────────────────────────────────────
PERSON_CONF_TH      = 0.70   # 사람 검출 score 하한 ↑ (오탐 감소)
GENDER_CONF_TH      = 0.85   # 성별 추정 확신도 하한 ↑ (불확실 → 입력값 유지)
PERSON_AREA_FRAC_TH = 0.18   # '사람' 박스가 이미지에서 차지하는 최소 면적 비율 (모델컷 판단용)

@dataclass
class Decision:
    label: str   # M/W/U/L
    reason: str  # 근거

def has_any(text: str, keys) -> bool:
    up = (text or "").upper()
    return any(k in up for k in keys)

def is_woman_only(goods_nm: str, full_nm: str) -> bool:
    low = f"{goods_nm} || {full_nm}".lower()
    return any(k in low for k in WOMAN_ONLY_CATS)

def decide_no_model(goods_nm: str, full_nm: str) -> Decision:
    full_up = f"{goods_nm} || {full_nm}".upper()

    # 0) 여성 전용 키워드 최우선
    if is_woman_only(goods_nm, full_nm):
        return Decision("W", "여성 전용 키워드 매칭 → W")

    # 1) Beauty/Life
    if has_any(full_up, BEAUTY_KEYS) or has_any(full_up, LIFE_KEYS):
        return Decision("L", "모델 없음 + Beauty/Life 카테고리 → L")

    # 2) 명시 키워드
    if has_any(full_up, WOMAN_KEYS):
        return Decision("W", "모델 없음 + 여성 키워드 → W")
    if has_any(full_up, MAN_KEYS):
        return Decision("M", "모델 없음 + 남성 키워드 → M")
    if has_any(full_up, UNISEX_KEYS):
        return Decision("U", "모델 없음 + 공용 키워드 → U")

    # 3) 단서 없음
    return Decision("U", "모델 없음 + 성별 단서 없음 → U")
