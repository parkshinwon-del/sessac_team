import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

# ------------------------------
# 1. 데이터 불러오기
# ------------------------------
df = pd.read_csv("labels.csv")

# food_id를 one-hot 인코딩용으로 정리 (15종 - 0~14)
NUM_FOOD_CLASSES = df["food_id"].nunique()


# ------------------------------
# 2. 양추정 Dataset 클래스 정의
# ------------------------------
class PortionDataset(Dataset):
    def __init__(self, df, image_dir, cnn_model, transform):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.cnn_model = cnn_model  # 특징벡터 뽑을 CNN (미리 학습된 것)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["filename"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # CNN 특징벡터 추출 (학습 안 시키고 그냥 통과만 시킴)
        with torch.no_grad():
            feature = self.cnn_model(image.unsqueeze(0)).squeeze(0)  # 예: 512차원

        # 음식 카테고리 one-hot
        food_onehot = torch.zeros(NUM_FOOD_CLASSES)
        food_onehot[row["food_id"]] = 1.0

        # MLP 입력 = CNN 특징벡터 + 음식 one-hot 이어붙이기
        mlp_input = torch.cat([feature, food_onehot])

        label = row["q_label"]  # 0~4

        return mlp_input, label


# ------------------------------
# 3. MLP 모델 정의
# ------------------------------
class PortionMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5)  # Q1~Q5
        )

    def forward(self, x):
        return self.net(x)


# ------------------------------
# 4. 훈련 및 검증
# ------------------------------
def train(model, train_loader, val_loader, epochs=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 검증
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                output = model(x)
                pred = output.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        print(f"Epoch {epoch+1}: loss={total_loss:.4f}, val_acc={correct/total:.4f}")
