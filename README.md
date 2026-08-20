# team_project2 YOLO - 학원 수업 스타일

학원 자료의 YOLO 구조와 코드 흐름을 최대한 유지하면서,
현재 음식 데이터셋에 맞게 경로와 클래스만 바꾼 프로젝트입니다.

## 1. 현재 데이터

원본:
- `C:\Users\user\Desktop\team_project2\image`
- `C:\Users\user\Desktop\team_project2\label`

## 2. YOLO 표준 구조

`preprocessing/yolo_preprocessing.py`를 실행하면 다음 구조가 만들어집니다.

```text
C:\Users\user\Desktop\team_project2\YOLODataset
├─ images
│  ├─ train
│  └─ valid
└─ labels
   ├─ train
   └─ valid
```

학원 자료의 `Data/PeachDataset/YoloDataset` 구조와 같은 형태입니다.

## 3. 실행 순서

### 1) YOLO 데이터셋 구조 만들기

프로젝트 루트에서:

```powershell
python preprocessing/yolo_preprocessing.py
```

### 2) 데이터/라벨 확인

```powershell
python check_dataset.py
```

### 3) YOLO 학습

```powershell
python main.py
```

학원 자료의 `models/yolo.py`에서 사용한
`YOLO('yolov8n.pt').train(...)` 흐름을 그대로 유지했습니다.

## 4. 학습 설정

`models/yolo.py`에서:

- epochs = 50
- imgsz = 640
- batch = 16
- device = 0
- plots = True
- name = `food_train01`

로 설정했습니다.

GPU가 없는 경우:

```python
device='cpu'
```

로 변경합니다.

## 5. 클래스

```text
0  김밥
1  김치볶음밥
2  삼선볶음밥
3  새우볶음밥
4  소고기김밥
5  쌀밥
6  알밥
7  영양돌솥밥
8  잡탕밥
9  전주비빔밥
10 제육덮밥
11 참치마요삼각김밥
12 콩밥
13 전주콩나물국밥
```

`yolo_setting.yaml`의 클래스 번호와 현재 라벨의 class_id가 반드시 동일해야 합니다.
