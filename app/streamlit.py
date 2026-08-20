# ============================================================
# streamlit_app.py - 로컬 데모 UI
# 실행: streamlit run app/streamlit_app.py  (프로젝트 최상위 폴더)
# ============================================================

import os
import sys

import streamlit as st
from PIL import Image

# 최상위 폴더(sessac_team/)에 있는 main.py를 불러오기 위한 경로 설정
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import main as pipeline  # main.py 안의 함수/데이터를 pipeline.xxx 로 사용

# ------------------------------------------------------------
# 음식분류 모델 연결 여부 확인
#   - 분류 담당자 코드가 아직이면 자동으로 "수동 선택" 모드로 전환됨
#   - 나중에 main.py의 get_food_class 주석을 풀면 자동으로 이 앱도 연결됨
# ------------------------------------------------------------
HAS_CLASSIFIER = hasattr(pipeline, "get_food_class")

# 수동 선택용 음식 목록 (food_base_info에서 그대로 가져옴 → 이름/id 항상 일치)
FOOD_OPTIONS = pipeline.food_base_info["food_name"].to_dict()  # {food_id: food_name}

# ------------------------------------------------------------
# 화면 구성
# ------------------------------------------------------------
st.set_page_config(page_title="한 끼 양 추정 데모", page_icon="🍚")
st.title("🍚 한 끼 양 추정 데모")
st.write("사진을 올리면 음식 종류와 양(Q등급), 예상 무게/칼로리를 알려드려요.")

uploaded_file = st.file_uploader("음식 사진을 올려주세요", type=["jpg", "jpeg", "png"])

# 분류 모델이 아직 없으면, 사람이 직접 음식 종류를 선택하게 함
selected_food_id = None
if uploaded_file is not None and not HAS_CLASSIFIER:
    st.info("음식분류 모델이 아직 연결되지 않아, 음식 종류를 직접 선택해주세요.")
    selected_name = st.selectbox("음식 종류", list(FOOD_OPTIONS.values()))
    selected_food_id = [k for k, v in FOOD_OPTIONS.items() if v == selected_name][0]

run_button = st.button("결과 확인하기", disabled=(uploaded_file is None))


# ------------------------------------------------------------
# 버튼 눌렀을 때 실행되는 부분
# ------------------------------------------------------------
if run_button and uploaded_file is not None:
    with st.spinner("분석 중입니다..."):
        # 업로드된 이미지를 임시로 저장 (기존 함수들이 '경로'를 받는 구조라서)
        temp_path = "temp_uploaded_image.jpg"
        image = Image.open(uploaded_file).convert("RGB")
        image.save(temp_path)

        # 1. 음식분류
        if HAS_CLASSIFIER:
            food_id, food_name = pipeline.get_food_class(temp_path)
        else:
            food_id = selected_food_id
            food_name = FOOD_OPTIONS[food_id]

        # 2. 양추정
        q_grade = pipeline.get_portion_grade(temp_path, food_id)

        # 3. 무게/칼로리 환산
        weight_g, kcal = pipeline.estimate_weight_and_kcal(food_id, q_grade)

        os.remove(temp_path)  # 임시 파일 정리

    # ---- 결과 화면 ----
    st.image(image, caption="업로드한 사진", use_column_width=True)

    st.subheader("분석 결과")
    col1, col2, col3 = st.columns(3)
    col1.metric("음식 종류", food_name)
    col2.metric("양 등급", f"Q{q_grade}")
    col3.metric("추정 칼로리", f"{kcal} kcal")

    st.write(f"**추정 무게**: 약 {weight_g}g")
