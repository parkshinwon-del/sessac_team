# 한 끼 양 추정 프로젝트

사진 한 장으로 음식 종류를 인식하고, 담긴 양(Q1~Q5)을 추정해서
예상 무게(g)와 칼로리(kcal)를 알려주는 딥러닝 파이프라인입니다.

```
사진 업로드 → 음식분류(YOLO) → 양추정(ResNet) → 무게/칼로리 환산 → 결과 출력
```

## 데모 실행 화면

Streamlit으로 사진을 올리면 음식 종류, 확신도, 양 등급, 예상 무게/칼로리가 바로 나옵니다.

---

## 1. 폴더 구조

```
sessac_team/
├── main.py                    # 전체 파이프라인 연결
├── requirements.txt           # 필요 패키지 목록
├── yolo_setting.yaml          # YOLO 학습 설정(클래스 목록 등)
├── yolov8n.pt                 # YOLO 사전학습 베이스 체크포인트
│
├── Classification/            # 음식분류 (YOLO)
│   ├── predict.py             # predict_food() 함수 - main.py에서 사용
│   └── runs/detect/food_train01-5/weights/best.pt   # 학습된 분류 모델
│
├── portion/                   # 양추정 (ResNet)
│   ├── portion_estimate.py    # 모델 정의 + 학습 코드
│   └── weights/
│       └── portion_estimate_best.pt
│
├── data/                      # 라벨/메타데이터
│   ├── labels.csv             # 이미지별 food_id, q_grade, weight_base_g, kcal_base 등
│   └── foodmap.csv            # 음식 종류 목록
│
├── app/
│   └── streamlit_app.py       # 로컬 데모 UI
│
├── models/                    # YOLO 학습 스크립트 (yolo.py 등)
├── preprocessing/              # 라벨/데이터셋 전처리 스크립트
└── outputs/                    # 학습 기록, Grad-CAM 결과 등
```
---

## 2. 실행 방법

### 2-1. 환경 준비

```bash
conda activate CV
python -m pip install -r requirements.txt
```

### 2-2. 콘솔에서 파이프라인만 테스트

```bash
python main.py
```

`main.py` 안 `image_path` 변수를 원하는 사진 경로로 바꾼 뒤 실행하면,
터미널에 분류 결과 → 양 등급 → 무게/칼로리가 순서대로 출력됩니다.

### 2-3. 웹 데모 실행

```bash
python -m streamlit run app/streamlit_app.py
```
---

## 3. 파이프라인 상세

| 단계 | 담당 | 방식 | 입력 | 출력 |
|---|---|---|---|---|
| 음식분류 | Classification | YOLOv8 (Object Detection) | 사진 | food_id, food_name, 확신도(conf) |
| 양추정 | portion | ResNet18 Fine-tuning + MLP | 사진, food_id | Q등급 (1~5) |
| 칼로리 계산 | main.py | labels.csv 기준 환산 | food_id, Q등급 | 무게(g), 칼로리(kcal) |

### 양추정 모델 선택 과정

- **1차 시도 (Feature Extraction)**: ResNet18을 고정하고 특징벡터만 뽑아 MLP 학습 → val_acc 최고 약 0.63
- **2차 시도 (Fine-tuning)**: ResNet18도 함께 학습 → val_acc 최고 **0.9383** → 이 방식으로 최종 확정
- 두 방식 비교 기록: `outputs/results_finetune_final.csv`

### 발견한 한계 (Grad-CAM으로 진단)

- 학습 데이터가 스튜디오 환경(단일 음식, 고정 구도)에서 촬영되어, 모델이 "실제 양"이 아니라
  "카메라 확대 정도"에 의존해 판단하는 경향을 Grad-CAM으로 확인함
- `RandomResizedCrop` 등 증강 추가로 완화 시도
- 여러 음식/배경이 섞인 복잡한 실제 식탁 사진에서는 정확도가 떨어짐
  (학습 데이터가 단일 음식 위주라 생기는 구조적 한계)

---

## 4. 팀 구성

| 파트 | 담당 내용 |
|---|---|
| 음식분류 | YOLOv8 기반 음식 탐지/분류 |
| 양추정 | ResNet18 Fine-tuning + MLP 기반 Q등급(1~5) 예측 |
| 통합/UI | main.py 파이프라인 연결, Streamlit 데모 |

## 5. 앞으로 개선하면 좋을 것

- 복잡한 배경/여러 음식이 섞인 사진 데이터 추가
- 음식 영역만 미리 crop해서 양추정 모델에 넣는 방식 검토
- ColorJitter, RandomRotation 등 추가 증강 실험
- 밥류 외 다른 음식 카테고리(국물류, 반찬류 등)로 확장