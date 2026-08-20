from pathlib import Path
import shutil


# ============================================================
# 1. 이미 수정한 원본 Train 라벨
# ============================================================
SOURCE_ROOT = Path(
    r"C:\Users\user\Desktop\team_project2\label\train"
)

# ============================================================
# 2. YOLO가 실제 사용하는 Train 라벨
# ============================================================
DEST_ROOT = Path(
    r"C:\Users\user\Desktop\team_project2\YOLODataset\labels\train"
)


# ============================================================
# 3. 경로 확인
# ============================================================
print("=" * 80)
print("TRAIN YOLO 라벨 재구성")
print("=" * 80)

print()
print(f"원본 라벨 : {SOURCE_ROOT}")
print(f"YOLO 라벨 : {DEST_ROOT}")


if not SOURCE_ROOT.exists():
    print("[ERROR] 원본 train 라벨 폴더가 없습니다.")
    raise SystemExit


# ============================================================
# 4. 기존 YOLO train 라벨 삭제
#
# 이미지 자체는 건드리지 않고
# labels/train 안의 txt만 초기화
# ============================================================
if DEST_ROOT.exists():

    deleted = 0

    for txt_file in DEST_ROOT.glob("*.txt"):

        txt_file.unlink()
        deleted += 1

    print()
    print(
        f"기존 YOLO train 라벨 삭제 : "
        f"{deleted}개"
    )

else:

    DEST_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("YOLO train 라벨 폴더 생성")


# ============================================================
# 5. source의 모든 음식 폴더에서 txt 수집
# ============================================================
copied = 0
duplicate_names = 0

# 같은 파일명이 여러 음식 폴더에 있는지 검사
name_owner = {}


for food_folder in sorted(
    SOURCE_ROOT.iterdir()
):

    if not food_folder.is_dir():
        continue

    print()
    print(f"처리 폴더 : {food_folder.name}")

    txt_files = list(
        food_folder.glob("*.txt")
    )

    print(
        f"  TXT : {len(txt_files)}개"
    )

    for txt_file in txt_files:

        filename = txt_file.name

        # ----------------------------------------------------
        # 동일 파일명 중복 검사
        # ----------------------------------------------------
        if filename in name_owner:

            duplicate_names += 1

            print(
                f"[중복 파일명] {filename}"
            )

            print(
                f"  기존 : {name_owner[filename]}"
            )

            print(
                f"  현재 : {txt_file}"
            )

            # 일단 현재 파일은 복사하지 않음
            continue

        name_owner[filename] = txt_file

        # ----------------------------------------------------
        # YOLO labels/train 바로 아래로 복사
        # ----------------------------------------------------
        destination = DEST_ROOT / filename

        shutil.copy2(
            txt_file,
            destination
        )

        copied += 1


# ============================================================
# 6. 결과
# ============================================================
print()
print("=" * 80)
print("TRAIN 라벨 재구성 완료")
print("=" * 80)

print(
    f"복사된 TXT       : {copied}개"
)

print(
    f"중복 파일명      : {duplicate_names}개"
)

print(
    f"최종 YOLO 라벨   : "
    f"{DEST_ROOT}"
)

print("=" * 80)