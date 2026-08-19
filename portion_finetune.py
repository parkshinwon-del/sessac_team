# ============================================================
# 양 추정 모델 - 비교 실험용 (Fine-tuning 방식)
# ============================================================
#
# ■ 이전 코드(portion_mlp.py)와 무엇이 다른가
#
#   [이전 방식 - Feature Extraction]
#     ResNet18을 얼려둠(고정) → 특징 512개만 뽑아냄 → MLP만 학습
#     · ResNet은 ImageNet(강아지/자동차 등)으로 학습된 상태 그대로
#     · "밥이 얼마나 많은지"에 특화되어 있지 않음
#     · 학습이 빠름
#
#   [이 코드 - Fine-tuning]
#     ResNet18도 함께 학습시킴 → 밥의 양을 구별하도록 CNN 자체가 적응
#     · 보통 정확도가 더 높게 나옴
#     · 학습이 느리고 GPU 메모리를 더 씀
#
#   두 방식을 같은 데이터로 돌려 정확도를 비교하는 것이 이 실험의 목적이다.
#
# ------------------------------------------------------------
# ■ 실행 방법
#
#   1) 아래 "경로 설정" 3줄을 본인 환경에 맞게 수정
#   2) 먼저 소량 테스트:   python portion_finetune.py --quick
#   3) 정상이면 전체 학습: python portion_finetune.py
#
#   결과는 results_finetune.csv 로 저장되어 이전 결과와 비교할 수 있다.
# ============================================================

import os
import argparse

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image


# ------------------------------------------------------------
# 0. 경로 설정  (★ 이 3개만 본인 환경에 맞게 수정)
# ------------------------------------------------------------
LABELS_CSV = "labels.csv"
IMAGE_ROOT = "C:/Users/user/Downloads/rice_dataset_final/images"        # 이 안에 train/ 과 val/ 폴더가 있음
NUM_FOOD_CLASSES = 15          # foodmap.csv 기준 음식 종류 수

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------
# 1. 이미지 전처리
#
#    전처리 단계에서 이미 224x224로 맞춰두었으므로 Resize는 안전장치다.
#    정규화 값은 ImageNet 표준값으로, 사전학습 ResNet을 쓰므로 그대로 사용한다.
#
#    ※ 학습셋에만 증강(좌우반전)을 적용한다.
#      검증셋에 증강을 넣으면 평가가 매번 달라져 비교가 불가능해진다.
# ------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),      # 음식은 좌우 뒤집혀도 같은 음식
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ------------------------------------------------------------
# 2. Dataset
#
#    ※ 이전 코드와의 중요한 차이:
#      이미지가 images/train/ 과 images/val/ 로 나뉘어 저장되어 있으므로
#      split 값에 따라 하위 폴더를 붙여야 파일을 찾을 수 있다.
#
#    ※ 또 하나의 차이:
#      이전 코드는 Dataset 안에서 CNN을 돌려 특징을 뽑는다.
#      Fine-tuning에서는 CNN도 학습해야 하므로 Dataset은 이미지만 넘기고
#      CNN 통과는 모델 안에서 수행한다.
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

        # split 하위 폴더까지 포함해 경로를 만든다
        img_path = os.path.join(self.image_root, row["split"], row["filename"])
        if not os.path.exists(img_path):
            # 하위 폴더 없이 한곳에 모아둔 경우도 대비
            alt = os.path.join(self.image_root, row["filename"])
            img_path = alt if os.path.exists(alt) else img_path

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # 음식 종류 one-hot (분류 모델이 알려준다고 가정하는 정보)
        food_onehot = torch.zeros(NUM_FOOD_CLASSES)
        food_onehot[int(row["food_id"])] = 1.0

        label = int(row["q_label"])          # 0~4 (Q1~Q5)

        return image, food_onehot, label


# ------------------------------------------------------------
# 3. 모델
#
#    이미지 → ResNet18 → 512차원 특징
#    특징(512) + 음식 one-hot(15) = 527 → MLP → 5개 클래스
#
#    freeze_cnn=True 로 두면 팀원 코드와 동일한 방식(특징 추출만)이 되어
#    같은 파일로 두 조건을 모두 실험할 수 있다.
# ------------------------------------------------------------
class PortionNet(nn.Module):
    def __init__(self, num_food=NUM_FOOD_CLASSES, num_classes=5, freeze_cnn=False):
        super().__init__()

        self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feature_dim = self.cnn.fc.in_features      # ResNet18 = 512
        self.cnn.fc = nn.Identity()                # 분류층 제거 → 특징벡터 출력

        self.freeze_cnn = freeze_cnn
        if freeze_cnn:
            for p in self.cnn.parameters():
                p.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(feature_dim + num_food, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, image, food_onehot):
        if self.freeze_cnn:
            with torch.no_grad():
                feat = self.cnn(image)
        else:
            feat = self.cnn(image)
        x = torch.cat([feat, food_onehot], dim=1)
        return self.head(x)


# ------------------------------------------------------------
# 4. 학습 / 평가
# ------------------------------------------------------------
def run_epoch(model, loader, criterion, optimizer=None):
    """optimizer가 있으면 학습, 없으면 평가."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    # 인접 등급 오차(±1등급 이내)도 함께 측정한다.
    # Q3과 Q4는 사진상 차이가 작아, 완전정답만 보면 성능이 과소평가된다.
    near = 0

    with torch.set_grad_enabled(is_train):
        for image, food, y in loader:
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
    # 학습 대상 파라미터만 옵티마이저에 전달 (고정된 CNN은 제외)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)

    history = []
    best_acc = 0.0

    for ep in range(1, epochs + 1):
        tr_loss, tr_acc, _ = run_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_acc, va_near = run_epoch(model, val_loader, criterion)

        history.append({
            "method": tag, "epoch": ep,
            "train_loss": round(tr_loss, 4), "train_acc": round(tr_acc, 4),
            "val_loss": round(va_loss, 4), "val_acc": round(va_acc, 4),
            "val_acc_within1": round(va_near, 4),
        })

        mark = ""
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), f"portion_{tag}.pt")
            mark = "  <- 최고 성능 저장"

        print(f"[{tag}] Epoch {ep:2d}/{epochs}  "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}  "
              f"val_acc={va_acc:.4f}  ±1등급={va_near:.4f}{mark}")

    return history, best_acc


# ------------------------------------------------------------
# 5. 실행
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None,
                    help="미지정 시 방식에 맞는 기본값 사용")
    ap.add_argument("--quick", action="store_true",
                    help="소량(train 200 / val 100)으로 빠르게 동작 확인")
    ap.add_argument("--mode", default="both",
                    choices=["finetune", "frozen", "both"],
                    help="finetune=CNN 함께 학습 / frozen=이전 방식 / both=둘 다 비교")
    args = ap.parse_args()

    print(f"사용 디바이스: {DEVICE}")
    if DEVICE == "cpu":
        print("  [주의] GPU가 없어 학습이 매우 느립니다. --quick 으로 먼저 확인하세요.")

    df = pd.read_csv(LABELS_CSV)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)

    if args.quick:
        train_df = train_df.sample(min(200, len(train_df)), random_state=42).reset_index(drop=True)
        val_df = val_df.sample(min(100, len(val_df)), random_state=42).reset_index(drop=True)
        print("  [빠른 확인 모드] 소량 데이터만 사용합니다.")

    print(f"train {len(train_df):,}장 / val {len(val_df):,}장")

    train_ds = PortionDataset(train_df, IMAGE_ROOT, train_transform)
    val_ds = PortionDataset(val_df, IMAGE_ROOT, eval_transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    modes = ["frozen", "finetune"] if args.mode == "both" else [args.mode]
    all_history, summary = [], []

    for m in modes:
        freeze = (m == "frozen")
        # 고정 방식은 MLP만 학습하므로 학습률을 크게,
        # fine-tuning은 사전학습 가중치가 망가지지 않도록 작게 잡는다.
        lr = args.lr if args.lr else (1e-3 if freeze else 1e-4)

        print("\n" + "=" * 60)
        print(f"[{m}] {'CNN 고정 - 특징 추출만 (이전 방식)' if freeze else 'CNN 함께 학습 - Fine-tuning'}")
        print(f"학습률 {lr}")
        print("=" * 60)

        torch.manual_seed(42)
        model = PortionNet(freeze_cnn=freeze).to(DEVICE)
        hist, best = train(model, train_loader, val_loader,
                           args.epochs, lr, m)
        all_history.extend(hist)
        summary.append({"방식": m, "최고 val_acc": round(best, 4)})

    # ---- 결과 저장 및 비교 ----
    hist_df = pd.DataFrame(all_history)
    hist_df.to_csv("results_finetune.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 60)
    print("실험 결과 비교")
    print("=" * 60)
    print(pd.DataFrame(summary).to_string(index=False))
    print("\n상세 기록: results_finetune.csv")
    print("\n※ 무작위로 찍었을 때의 정확도는 0.2(1/5)입니다.")
    print("  이보다 확실히 높아야 모델이 실제로 학습된 것입니다.")


if __name__ == "__main__":
    main()
