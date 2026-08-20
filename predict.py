from ultralytics import YOLO

# 학습된 최고 성능 모델
model = YOLO(
    r"C:\Users\user\Desktop\team_project2_code\runs\detect\food_train01-5\weights\best.pt"
)

# 예측할 이미지
image_path = r"C:\Users\user\Desktop\gimchibob.jpg"

# 예측
results = model.predict(
    source=image_path,
    conf=0.25,
    save=True,
    show=True
)

print("예측 완료")