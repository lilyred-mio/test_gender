# 이미지 기반 성별 판정 에이전트

## 준비
1) 이 레포지토리를 GitHub에 생성 후, 위 파일들 그대로 추가
2) `sample_set.csv`(입력 데이터) 업로드 – 컬럼: goods_no, goods_nm, gender, full_nm, imgurl (등)

## 실행
- GitHub Actions 탭 → **Run Gender Agent** 워크플로 → **Run workflow**
- 완료 후 **Artifacts**에서 `labeled-csv` 다운로드 → `sample_set_labeled.csv`에 `answer_reason` 열 포함

## 판정 로직
- 사람(모델) 검출: Torchvision Faster R-CNN
- 성별 추정: DeepFace(확신도 낮으면 U)
- 모델 없으면: 카테고리/상품명 규칙 (Beauty/Life→L, 남/여 단어→M/W, 공용→U, 기타→U)

## 주의
- 자동 성별 추정은 100% 정확하지 않음 → 불확실시 U로 보수적 처리
- 네트워크/모델 다운로드에 1~2분 소요 가능
