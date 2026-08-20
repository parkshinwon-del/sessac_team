# ============================================================
# main.py - 전체 파이프라인 연결
# 흐름: 사진 → 음식분류 → 양추정 → 칼로리 계산 → 결과 출력
# ============================================================

import pandas as pd
import torch
from Portion.portion_estimate import PortionNet, eval_transform, NUM_FOOD_CLASSES
from PIL import Image
# from food_classifier import predict_food

# ---- 1. 음식분류 모델 (분류 코드에서 가져오기) ----
# predict_food(image_path) -> (food_id: int, food_name: str) 형태라고 가정

# ---- 2. 양추정 모델 ----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PORTION_MODEL_PATH = "./portion/weights/portion_estimate_best.pt"
LABELS_CSV = "./data/labels.csv"  # weight_base_g, kcal_base 등 참고용

# ------------------------------------------------------------
# 양추정 모델 - 준비
# ------------------------------------------------------------
portion_model = PortionNet(num_food=NUM_FOOD_CLASSES).to(DEVICE)
portion_model.load_state_dict(torch.load(PORTION_MODEL_PATH, map_location=DEVICE))
portion_model.eval()

# food_id 기준으로 기준 무게/칼로리를 빨리 찾기 위한 참고표
labels_df = pd.read_csv(LABELS_CSV)  # 구분자/인코딩은 실제 파일에 맞게 조정
food_base_info = labels_df.drop_duplicates(subset="food_id").set_index("food_id")[
    ["food_name", "weight_base_g", "kcal_base"]
]

# ------------------------------------------------------------
# 1. 음식분류 결과 받아오기
# ------------------------------------------------------------
# def get_food_class(image_path):
#     food_id, food_name = predict_food(image_path)
#     return food_id, food_name

# ------------------------------------------------------------
# 2. 양추정 결과(Q등급) 받아오기
# ------------------------------------------------------------
def get_portion_grade(image_path, food_id):
    image = Image.open(image_path).convert("RGB")
    image_tensor = eval_transform(image).unsqueeze(0).to(DEVICE)

    food_onehot = torch.zeros(1, NUM_FOOD_CLASSES).to(DEVICE)
    food_onehot[0, food_id] = 1.0

    with torch.no_grad():
        output = portion_model(image_tensor, food_onehot)
        pred_class = output.argmax(dim=1).item()  # 0~4

    q_grade = pred_class + 1  # 1~5
    return q_grade

# ------------------------------------------------------------
# 3. Q등급 기준 실제 무게/칼로리 환산
# ------------------------------------------------------------
Q_RATIO_MAP = {1: 0.25, 2: 0.5, 3: 0.75, 4: 1.0, 5: 1.25}

def estimate_weight_and_kcal(food_id, q_grade):
    base = food_base_info.loc[food_id]
    ratio = Q_RATIO_MAP[q_grade]

    weight_g = base["weight_base_g"] * ratio
    kcal = base["kcal_base"] * ratio

    return round(weight_g, 1), round(kcal, 1)

# ------------------------------------------------------------
# 4. 결과 출력
# ------------------------------------------------------------
def show_result(food_name, q_grade, weight_g, kcal):

    print("=" * 40)
    print(f"음식: {food_name}")
    print(f"양 등급: Q{q_grade}")
    print(f"추정 무게: {weight_g}g")
    print(f"추정 칼로리: {kcal}kcal")
    print("=" * 40)

# ------------------------------------------------------------
# 실행
# ------------------------------------------------------------
if __name__ == "__main__":
    image_path = "./ssalbap.jpg"  # 테스트할 사진 경로

    # 1. 음식분류
    #food_id, food_name = get_food_class(image_path)

    # 2. 양추정
    #q_grade = get_portion_grade(image_path, food_id)

    # 3. 무게/칼로리 환산
    #weight_g, kcal = estimate_weight_and_kcal(food_id, q_grade)

    # 5. 결과 출력
    #show_result(food_name, q_grade, weight_g, kcal)
