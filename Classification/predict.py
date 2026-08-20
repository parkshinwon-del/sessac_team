from ultralytics import YOLO

# 학습된 최고 성능 모델
model = YOLO(r"./classification/runs/detect/food_train01-6/weights/best.pt")

def predict_food(image_path):
    '''
    image_path : 예측할 사진 경로
    
    return 값 : food_id, food_name, conf
        food_id : 클래스 번호,
        food_name : 음식 이름,
        conf : 확신도
        
    사진에 음식이 없거나 분류하지 못할 시 예외 발생
    '''

    # 예측
    results = model.predict(
        source=image_path,
        conf=0.25,
        save=False,
        verbose=False
    )

    boxes = results[0].boxes

    if len(boxes) == 0:
        raise ValueError("사진에서 음식을 찾을 수 없습니다.")

    # 음식이 여러 개 존재할 시, 확신도가 가장 높은 것을 대표 음식으로 사용
    best_idx = boxes.conf.argmax().item()

    food_id = int(boxes.cls[best_idx].item())
    food_name = model.names[food_id]
    conf = float(boxes.conf[best_idx].item())

    return food_id, food_name, conf

# ------------------------------------------------------------
# 테스트용
# ------------------------------------------------------------
if __name__ == "__main__":
    # 예측할 이미지
    # image_path = r"C:\Users\user\Downloads\ssalbap.jpg"

    # # 예측
    # results = model.predict(
    #     source=image_path,
    #     conf=0.25,
    #     #save=True,
    #     show=True
    # )
    
    image_path = r"C:\Users\user\Downloads\ssalbap.jpg"

    food_id, food_name, conf = predict_food(image_path)
    # print(f"class_id: {food_id}, 예측된 음식: {food_name}, 확신도: {conf:.2f}")
