# ============================================================
# 양 추정 모델 (Q1~Q5 예측)
# 흐름: 사진 → CNN(ResNet18)으로 특징벡터 추출 → 음식 카테고리와 합침 → MLP → Q등급 예측
# ============================================================

import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

# ------------------------------------------------------------
# 0. 경로 설정 (★ 본인 환경에 맞게 이 3개만 수정하세요)
# ------------------------------------------------------------
LABELS_CSV = "labels.csv"  # labels.csv 경로
IMAGE_DIR = (
    "./images"  # 이미지들이 들어있는 폴더 경로 (train/val 이미지가 여기 있다고 가정)
)
NUM_FOOD_CLASSES = 15  # foodmap.csv 기준 음식 종류 개수 (food_id: 0~14)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"사용 디바이스: {DEVICE}")

# ------------------------------------------------------------
# 1. 이미지 전처리 규칙 정의
#    - CNN(ResNet)이 학습됐을 때와 똑같은 방식으로 이미지를 맞춰줘야 함
#    - 224x224 크기로 자르고, ImageNet 학습 시 쓰인 평균/표준편차로 정규화
# ------------------------------------------------------------
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# ------------------------------------------------------------
# 2. CNN 특징 추출기 준비
#    - ImageNet으로 이미 학습된 ResNet18을 가져와서
#    - 마지막 분류층(fc)만 빼고 씀 → 이러면 "분류 결과"가 아니라
#      "512개 숫자로 요약된 이미지 특징벡터"가 출력됨
# ------------------------------------------------------------
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet.fc = nn.Identity()  # 마지막 층을 "그냥 통과"로 바꿔치기
resnet = resnet.to(DEVICE)
resnet.eval()  # 이 CNN은 추가 학습 안 하고 특징 추출 용도로만 씀

FEATURE_DIM = 512  # ResNet18 기준 특징벡터 차원 (ResNet50 쓰면 2048로 바뀜)


# ------------------------------------------------------------
# 3. Dataset 클래스
#    - labels.csv 한 줄 = 이미지 1장에 대한 정보
#    - __getitem__ 호출될 때마다: 이미지 열기 → CNN 특징 추출 → 음식 one-hot 붙이기
# ------------------------------------------------------------
class PortionDataset(Dataset):
    def __init__(self, df, image_dir):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["filename"])

        image = Image.open(img_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(DEVICE)  # (1, 3, 224, 224)

        with torch.no_grad():
            feature = resnet(image).squeeze(0).cpu()  # (512,)

        # 음식 카테고리를 one-hot 벡터로 (예: food_id=2 → [0,0,1,0,0,...])
        food_onehot = torch.zeros(NUM_FOOD_CLASSES)
        food_onehot[int(row["food_id"])] = 1.0

        # CNN 특징벡터 + 음식 one-hot 을 이어붙여서 MLP 입력으로 사용
        mlp_input = torch.cat([feature, food_onehot])  # (512+15,) = (527,)

        label = int(row["q_label"])  # 0~4 (Q1~Q5)

        return mlp_input, label


# ------------------------------------------------------------
# 4. MLP 모델 정의
#    - 입력: CNN 특징벡터 + 음식 one-hot
#    - 출력: 5개 클래스(Q1~Q5) 중 하나
# ------------------------------------------------------------
class PortionMLP(nn.Module):
    def __init__(self, input_dim, num_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------
# 5. 학습 함수
# ------------------------------------------------------------
def train(model, train_loader, val_loader, epochs=20, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        # ---- 학습 ----
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # ---- 검증 ----
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                output = model(x)
                pred = output.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        val_acc = correct / total if total > 0 else 0
        print(
            f"[Epoch {epoch+1}/{epochs}] train_loss={total_loss:.4f}  val_acc={val_acc:.4f}"
        )

    return model


# ------------------------------------------------------------
# 6. 새 사진 한 장 넣으면 Q값 예측하는 함수 (최종 사용 시나리오)
# ------------------------------------------------------------
def predict_single_image(model, image_path, food_id):
    """
    image_path: 예측하고 싶은 사진 경로
    food_id: 이 사진이 어떤 음식인지 (분류 모델이 먼저 맞춰준 값이라고 가정)
    """
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        feature = resnet(image).squeeze(0).cpu()

    food_onehot = torch.zeros(NUM_FOOD_CLASSES)
    food_onehot[food_id] = 1.0

    mlp_input = torch.cat([feature, food_onehot]).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(mlp_input)
        pred_class = output.argmax(dim=1).item()  # 0~4

    q_grade = f"Q{pred_class + 1}"  # 0→Q1, 1→Q2 ...
    print(f"예측 결과: {q_grade}")
    return q_grade


# ------------------------------------------------------------
# 7. 실행부
# ------------------------------------------------------------
if __name__ == "__main__":
    df = pd.read_csv(LABELS_CSV)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)

    print(f"train: {len(train_df)}개, val: {len(val_df)}개")

    # ★ 처음엔 파이프라인이 잘 도는지만 확인하려면 아래 두 줄 주석 풀어서 소량만 테스트하세요
    # train_df = train_df.sample(50, random_state=42).reset_index(drop=True)
    # val_df = val_df.sample(20, random_state=42).reset_index(drop=True)

    train_dataset = PortionDataset(train_df, IMAGE_DIR)
    val_dataset = PortionDataset(val_df, IMAGE_DIR)

    # num_workers=0 으로 시작 (에러 나면 이미지 로딩 문제일 확률 높음)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    mlp_model = PortionMLP(input_dim=FEATURE_DIM + NUM_FOOD_CLASSES).to(DEVICE)

    trained_model = train(mlp_model, train_loader, val_loader, epochs=20)

    # 모델 저장
    torch.save(trained_model.state_dict(), "portion_mlp.pt")
    print("모델 저장 완료: portion_mlp.pt")

    # 사용 예시 (경로/food_id는 실제 값으로 바꿔서 테스트)
    # predict_single_image(trained_model, "./images/side_주먹밥김밥류_접시_김밥_Q1_00001.JPG", food_id=0)
