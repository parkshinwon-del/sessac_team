# ============================================================
# Grad-CAM 시각화
# "모델이 사진에서 어느 부분을 보고 Q등급을 판단했는지" 히트맵으로 확인
# ============================================================
#
# 사용법:
#   1) MODEL_PATH, IMAGE_PATH, FOOD_ID 를 본인 상황에 맞게 수정
#   2) python gradcam_visualize.py 실행
#   3) gradcam_result.png 파일이 생성됨 (원본 + 히트맵 오버레이)
#
# 결과 해석:
#   - 빨간색/노란색이 진할수록 모델이 "그 부분을 많이 봤다"는 뜻
#   - 음식 부분이 빨갛게 나오면 정상
#   - 배경이나 접시 테두리, 사진 프레임 전체가 빨갛게 나오면
#     "음식이 아니라 다른 걸 보고 판단하고 있다"는 신호 (지름길 학습 의심)
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models, transforms

# portion_finetune_final.py 에 정의된 모델 클래스를 그대로 가져다 씀
# (같은 폴더에 두거나, import 경로를 맞춰주세요)
from portion_finetune_test import PortionNet, eval_transform, NUM_FOOD_CLASSES

# ------------------------------------------------------------
# 0. 설정 (★ 본인 환경에 맞게 수정)
# ------------------------------------------------------------
MODEL_PATH = "portion_finetune_best.pt"
IMAGE_PATH = "./gimbab.jpg"  # 확인하고 싶은 사진 경로
FOOD_ID = 0  # 이 사진의 음식 종류 번호

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def generate_gradcam(model, image_tensor, food_onehot, target_layer):
    """
    image_tensor : (1, 3, 224, 224) 전처리된 이미지
    target_layer : Grad-CAM을 뽑을 CNN의 마지막 conv layer
    """
    activations = []
    gradients = []

    # 마지막 conv layer를 지나갈 때의 출력값(activation)과
    # 역전파 시 그래디언트를 붙잡아두는 hook 설치
    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    # 예측 실행
    model.eval()
    output = model(image_tensor, food_onehot)
    pred_class = output.argmax(dim=1).item()

    # 예측한 클래스에 대해서만 역전파 (그 클래스를 왜 그렇게 예측했는지 추적)
    model.zero_grad()
    output[0, pred_class].backward()

    h1.remove()
    h2.remove()

    # activation과 gradient로 각 채널의 중요도를 계산
    act = activations[0].squeeze(0)  # (C, H, W)
    grad = gradients[0].squeeze(0)  # (C, H, W)
    weights = grad.mean(dim=(1, 2))  # 채널별 중요도 (C,)

    cam = torch.zeros(act.shape[1:], dtype=torch.float32)
    for i, w in enumerate(weights):
        cam += w * act[i]

    cam = F.relu(cam)  # 음수 값(예측에 방해되는 부분)은 제외
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)  # 0~1로 정규화

    return cam.detach().cpu().numpy(), pred_class


def main():
    # ---- 모델 불러오기 ----
    model = PortionNet(num_food=NUM_FOOD_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    # ---- 이미지 준비 ----
    raw_image = Image.open(IMAGE_PATH).convert("RGB").resize((224, 224))
    image_tensor = eval_transform(raw_image).unsqueeze(0).to(DEVICE)

    food_onehot = torch.zeros(1, NUM_FOOD_CLASSES).to(DEVICE)
    food_onehot[0, FOOD_ID] = 1.0

    # ---- Grad-CAM 계산 ----
    # ResNet18의 마지막 conv 블록 = model.cnn.layer4
    target_layer = model.cnn.layer4[-1]
    cam, pred_class = generate_gradcam(model, image_tensor, food_onehot, target_layer)

    # ---- 히트맵을 원본 이미지 크기(224x224)로 확대 ----
    cam_resized = np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224))
    )

    # ---- 원본 + 히트맵 겹쳐서 저장 ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(raw_image)
    axes[0].set_title("원본 사진")
    axes[0].axis("off")

    axes[1].imshow(raw_image)
    axes[1].imshow(cam_resized, cmap="jet", alpha=0.5)  # 반투명하게 히트맵 겹치기
    axes[1].set_title(f"모델이 본 부분 (예측: Q{pred_class + 1})")
    axes[1].axis("off")

    plt.tight_layout()
    plt.rcParams["font.family"] = "Malgun Gothic"  # 윈도우 기본 한글 폰트
    plt.rcParams["axes.unicode_minus"] = False
    plt.savefig("gradcam_result.png", dpi=150)
    print("저장 완료: gradcam_result.png")
    print(f"예측 결과: Q{pred_class + 1}")


if __name__ == "__main__":
    main()
