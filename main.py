from ultralytics import YOLO

from models.yolo import yolo_train


if __name__ == '__main__':

    # 머신러닝 객체 생성 -> 훈련 -> 예측 -> 평가
    # 학원 수업 코드의 흐름을 최대한 유지

    yaml_path = r'./yolo_setting.yaml'

    # YOLO 훈련
    yolo_train()

    # ------------------------------------------
    # YOLO 평가 / 예측
    # ------------------------------------------
    # 학습이 끝나면 아래처럼 best.pt를 불러서 예측할 수 있음.
    #
    # model = YOLO('./runs/detect/food_train01/weights/best.pt')
    #
    # source = r'C:\Users\user\Desktop\team_project2\test.jpg'
    #
    # model.predict(
    #     source=source,
    #     device=0,
    #     save=True
    # )

    # ------------------------------------------
    # 딥러닝 시퀀스
    # 1. 데이터 가져옴
    # 2. 데이터 정제(preprocessing)
    # 3. 알고리즘 선택
    # 4. 훈련
    # 5. 검증
    # 6. 평가
    # 7. 배포
    # ------------------------------------------
