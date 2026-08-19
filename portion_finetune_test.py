# ============================================================
# 양 추정 모델 - 최종 버전 (Fine-tuning 방식)
# ============================================================
#
# ■ 비교 실험 결과 (results_finetune.csv)
#   frozen(팀원 portion_mlp 방식):  최고 val_acc 약 0.63 (11epoch)
#   finetune(이 코드):              최고 val_acc 약 0.94 (15epoch)
#   → finetune 방식으로 최종 확정
#
# ■ 이전 실험용 코드에서 바뀐 점
#   1. predict_single_image()  : 새 사진 한 장 넣으면 Q등급 예측하는 함수 추가
#   2. tqdm                    : 학습/검증 진행 상황 표시 추가
#   3. --mode 기본값을 finetune으로 고정 (frozen과 비교할 필요 없어졌으므로)
#   4. food_id 연동 지점        : 분류(CNN) 담당자 모델 결과를 받는 부분에 주석으로 표시
#
# ------------------------------------------------------------
# ■ 실행 방법
#   1) 경로 설정 3줄을 본인 환경에 맞게 수정
#   2) 학습:  python portion_finetune_final.py
#   3) 예측만 다시 하고 싶을 때는 맨 아래 predict_single_image() 부분 참고
# ============================================================

import os
import argparse

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

# ------------------------------------------------------------
# 0. 경로 설정 (★ 본인 환경에 맞게 수정)
# ------------------------------------------------------------
LABELS_CSV = "./labels.csv"
IMAGE_ROOT = "./images"  # 이 안에 train/, val/ 폴더
NUM_FOOD_CLASSES = 14  # foodmap.csv 기준 음식 종류 수

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------
# 1. 이미지 전처리
#    - 학습셋에는 좌우반전 증강 적용, 검증/예측에는 적용 안 함
# ------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

eval_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


# ------------------------------------------------------------
# 2. Dataset
# ------------------------------------------------------------
class PortionDataset(Dataset):
    def __init__(self, df, image_root, transform):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(self.image_root, row["split"], row["filename"])
        if not os.path.exists(img_path):
            alt = os.path.join(self.image_root, row["filename"])
            img_path = alt if os.path.exists(alt) else img_path

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        food_onehot = torch.zeros(NUM_FOOD_CLASSES)
        food_onehot[int(row["food_id"])] = 1.0

        label = int(row["q_label"])  # 0~4 (Q1~Q5)

        return image, food_onehot, label


# ------------------------------------------------------------
# 3. 모델
# ------------------------------------------------------------
class PortionNet(nn.Module):
    def __init__(self, num_food=NUM_FOOD_CLASSES, num_classes=5):
        super().__init__()

        self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feature_dim = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()

        self.head = nn.Sequential(
            nn.Linear(feature_dim + num_food, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, image, food_onehot):
        feat = self.cnn(image)
        x = torch.cat([feat, food_onehot], dim=1)
        return self.head(x)


# ------------------------------------------------------------
# 4. 학습 / 평가 (tqdm 진행 표시 추가)
# ------------------------------------------------------------
def run_epoch(model, loader, criterion, optimizer=None, desc=""):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    near = 0  # ±1등급 이내 맞춘 개수

    with torch.set_grad_enabled(is_train):
        for image, food, y in tqdm(loader, desc=desc, leave=False):
            image = image.to(DEVICE)
            food = food.to(DEVICE)
            y = y.to(DEVICE)

            out = model(image, food)
            loss = criterion(out, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * y.size(0)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            near += (pred - y).abs().le(1).sum().item()
            total += y.size(0)

    return total_loss / total, correct / total, near / total


def train(model, train_loader, val_loader, epochs, lr, tag):
    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)

    history = []
    best_acc = 0.0

    for ep in range(1, epochs + 1):
        tr_loss, tr_acc, _ = run_epoch(
            model, train_loader, criterion, optimizer, desc=f"Epoch {ep} [train]"
        )
        va_loss, va_acc, va_near = run_epoch(
            model, val_loader, criterion, desc=f"Epoch {ep} [val]"
        )

        history.append(
            {
                "method": tag,
                "epoch": ep,
                "train_loss": round(tr_loss, 4),
                "train_acc": round(tr_acc, 4),
                "val_loss": round(va_loss, 4),
                "val_acc": round(va_acc, 4),
                "val_acc_within1": round(va_near, 4),
            }
        )

        mark = ""
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), f"portion_{tag}_best.pt")
            mark = "  <- 최고 성능 저장"

        print(
            f"[{tag}] Epoch {ep:2d}/{epochs}  "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}  "
            f"val_acc={va_acc:.4f}  ±1등급={va_near:.4f}{mark}"
        )

    return history, best_acc


# ------------------------------------------------------------
# 5. 새 사진 한 장 넣으면 Q등급 예측하는 함수
# ------------------------------------------------------------
def predict_single_image(model, image_path, food_id):
    """
    image_path : 예측하고 싶은 사진 경로
    food_id    : 이 사진의 음식 종류 번호 (0~14)

    ※ 연동 지점 (4번 요청사항)
      지금은 food_id를 사람이 직접 넣어주고 있음.
      실제 서비스에서는 "음식분류(CNN) 담당자의 모델이 예측한 food_id"를
      여기로 그대로 넘겨받아야 함.
      예: food_id = classification_model.predict(image_path)
          predict_single_image(model, image_path, food_id)
      → 이 연결 코드는 통합 담당자와 함께 별도로 작성 필요
    """
    model.eval()

    image = Image.open(image_path).convert("RGB")
    image = eval_transform(image).unsqueeze(0).to(DEVICE)

    food_onehot = torch.zeros(NUM_FOOD_CLASSES).unsqueeze(0).to(DEVICE)
    food_onehot[0, food_id] = 1.0

    with torch.no_grad():
        output = model(image, food_onehot)
        pred_class = output.argmax(dim=1).item()  # 0~4

    q_grade = f"Q{pred_class + 1}"
    print(f"예측 결과: {q_grade}")
    return q_grade


# ------------------------------------------------------------
# 6. 실행
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)  # finetune 기본 학습률
    ap.add_argument(
        "--quick",
        action="store_true",
        help="소량(train 200 / val 100)으로 빠르게 동작 확인",
    )
    args = ap.parse_args()

    # print(f"사용 디바이스: {DEVICE}")
    # if DEVICE == "cpu":
    #     print("  [주의] GPU가 없어 학습이 매우 느립니다. --quick 으로 먼저 확인하세요.")

    df = pd.read_csv(LABELS_CSV)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)

    if args.quick:
        train_df = train_df.sample(
            min(200, len(train_df)), random_state=42
        ).reset_index(drop=True)
        val_df = val_df.sample(min(100, len(val_df)), random_state=42).reset_index(
            drop=True
        )
        print("  [빠른 확인 모드] 소량 데이터만 사용합니다.")

    #print(f"train {len(train_df):,}장 / val {len(val_df):,}장")

    train_ds = PortionDataset(train_df, IMAGE_ROOT, train_transform)
    val_ds = PortionDataset(val_df, IMAGE_ROOT, eval_transform)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    torch.manual_seed(42)
    model = PortionNet().to(DEVICE)

    # history, best_acc = train(
    #     model, train_loader, val_loader, args.epochs, args.lr, "finetune"
    # )

    # hist_df = pd.DataFrame(history)
    # hist_df.to_csv("results_finetune_final.csv", index=False, encoding="utf-8-sig")

    # print("\n" + "=" * 60)
    # print(f"최종 최고 val_acc: {best_acc:.4f}")
    # print("모델 저장: portion_finetune_best.pt")
    # print("기록 저장: results_finetune_final.csv")
    # print("=" * 60)

    # ---- 사용 예시 (통합 전, 개별 테스트용) ----
    best_model = PortionNet().to(DEVICE)
    best_model.load_state_dict(torch.load("portion_finetune_best.pt"))
    predict_single_image(best_model, "./gimbap.jpg", food_id=0)


if __name__ == "__main__":
    main()
