from ultralytics import YOLO


def yolo_train():

    yaml_path = r'./yolo_setting.yaml'

    # Yolo 훈련
    # 학원 수업 코드의 구조를 그대로 유지
    result = YOLO('yolov8n.pt').train(
        data=yaml_path,
        epochs=5,
        imgsz=640,
        batch=16,
        save=True,
        device=0,       # GPU가 없으면 'cpu'로 변경
        plots=True,
        name='food_train01'
    )

    print('훈련완료')
    return result
