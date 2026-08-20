from pathlib import Path
import random
import shutil
import xml.etree.ElementTree as ET


# ============================================================
# 1. 원본 이미지 위치
# ============================================================
SOURCE_IMAGE_ROOT = Path(
    r"H:\team project\food\image"
)


# ============================================================
# 2. 원본 라벨 위치
# ============================================================
SOURCE_LABEL_ROOT = Path(
    r"H:\team project\food\Training\[label]sorting_TRAIN"
)

SOURCE_TXT_ROOT = SOURCE_LABEL_ROOT / "txt"
SOURCE_XML_ROOT = SOURCE_LABEL_ROOT / "xml"


# ============================================================
# 3. 최종 YOLO Dataset
# ============================================================
YOLO_ROOT = Path(
    r"C:\Users\user\Desktop\team_project2\YOLODataset"
)

YOLO_IMAGE_ROOT = YOLO_ROOT / "images" / "train"
YOLO_LABEL_ROOT = YOLO_ROOT / "labels" / "train"


# ============================================================
# 4. 음식 코드 → 음식명 → YOLO class
# ============================================================
CLASS_MAP = {
    "01016001": ("김밥", 0),
    "01014001": ("김치볶음밥", 1),
    "01015008": ("삼선볶음밥", 2),
    "01014007": ("새우볶음밥", 3),
    "01016008": ("소고기김밥", 4),
    "01011001": ("쌀밥", 5),
    "01014008": ("알밥", 6),
    "01012004": ("영양돌솥밥", 7),
    "01014009": ("잡탕밥", 8),
    "01014004": ("전주비빔밥", 9),
    "01015003": ("전주콩나물국밥", 10),
    "01015010": ("제육덮밥", 11),
    "01016015": ("참치마요삼각김밥", 12),
    "01012002": ("콩밥", 13),
}


# ============================================================
# 5. 설정
# ============================================================
TARGET_COUNT = 400
RANDOM_SEED = 42

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# 6. 원본 음식 폴더 찾기
# ============================================================
def find_source_folder(code):

    candidates = [
        folder
        for folder in SOURCE_IMAGE_ROOT.iterdir()
        if folder.is_dir()
        and folder.name.startswith(code)
    ]

    if len(candidates) == 0:
        return None

    if len(candidates) > 1:
        print()
        print(f"[WARNING] {code} 폴더가 여러 개 발견됨")

        for folder in candidates:
            print("   ", folder)

    return candidates[0]


# ============================================================
# 7. 현재 YOLO train 이미지 목록
# ============================================================
def get_existing_images():

    existing = {}

    for file in YOLO_IMAGE_ROOT.iterdir():

        if not file.is_file():
            continue

        if file.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        existing[file.stem] = file

    return existing


# ============================================================
# 8. 원본 TXT/XML 목록 만들기
# ============================================================
def build_label_maps():

    print()
    print("=" * 80)
    print("원본 라벨 목록 생성")
    print("=" * 80)

    txt_map = {}
    xml_map = {}

    # --------------------------------------------------------
    # TXT
    # --------------------------------------------------------
    if SOURCE_TXT_ROOT.exists():

        for txt_file in SOURCE_TXT_ROOT.rglob("*.txt"):

            txt_map[txt_file.stem] = txt_file

    print(
        f"원본 TXT : {len(txt_map)}개"
    )

    # --------------------------------------------------------
    # XML
    # --------------------------------------------------------
    if SOURCE_XML_ROOT.exists():

        for xml_file in SOURCE_XML_ROOT.rglob("*.xml"):

            xml_map[xml_file.stem] = xml_file

    print(
        f"원본 XML : {len(xml_map)}개"
    )

    return txt_map, xml_map


# ============================================================
# 9. TXT → 최종 YOLO TXT
#
# 최종 형식:
# class x_center y_center width height
#
# 이미지 1장당 한 줄
# ============================================================
def create_label_from_txt(
    source_txt,
    output_txt,
    class_id
):

    with open(
        source_txt,
        "r",
        encoding="utf-8"
    ) as f:

        lines = [
            line.strip()
            for line in f.readlines()
            if line.strip()
        ]

    # 유효한 첫 번째 라벨을 사용
    for line in lines:

        parts = line.split()

        if len(parts) < 5:
            continue

        # 클래스 번호를 우리가 정한 번호로 강제
        parts[0] = str(class_id)

        # 정확히 5개만 저장
        final_line = " ".join(parts[:5])

        output_txt.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            output_txt,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(final_line + "\n")

        return True

    return False


# ============================================================
# 10. XML → YOLO TXT
#
# 이미지 1장당 첫 번째 object 하나
# ============================================================
def create_label_from_xml(
    source_xml,
    output_txt,
    class_id
):

    tree = ET.parse(source_xml)
    root = tree.getroot()

    # --------------------------------------------------------
    # 이미지 크기
    # --------------------------------------------------------
    size = root.find("size")

    if size is None:
        return False

    width = float(
        size.findtext("width")
    )

    height = float(
        size.findtext("height")
    )

    if width <= 0 or height <= 0:
        return False

    # --------------------------------------------------------
    # 첫 번째 객체
    # --------------------------------------------------------
    obj = root.find("object")

    if obj is None:
        return False

    bndbox = obj.find("bndbox")

    if bndbox is None:
        return False

    xmin = float(
        bndbox.findtext("xmin")
    )

    ymin = float(
        bndbox.findtext("ymin")
    )

    xmax = float(
        bndbox.findtext("xmax")
    )

    ymax = float(
        bndbox.findtext("ymax")
    )

    # --------------------------------------------------------
    # 좌표 보정
    # --------------------------------------------------------
    xmin = max(0, min(xmin, width))
    xmax = max(0, min(xmax, width))

    ymin = max(0, min(ymin, height))
    ymax = max(0, min(ymax, height))

    box_width = xmax - xmin
    box_height = ymax - ymin

    if box_width <= 0 or box_height <= 0:
        return False

    # --------------------------------------------------------
    # YOLO 변환
    # --------------------------------------------------------
    x_center = (
        ((xmin + xmax) / 2)
        / width
    )

    y_center = (
        ((ymin + ymax) / 2)
        / height
    )

    norm_width = box_width / width
    norm_height = box_height / height

    # --------------------------------------------------------
    # 최종 한 줄
    # --------------------------------------------------------
    final_line = (
        f"{class_id} "
        f"{x_center:.6f} "
        f"{y_center:.6f} "
        f"{norm_width:.6f} "
        f"{norm_height:.6f}"
    )

    output_txt.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_txt,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(final_line + "\n")

    return True


# ============================================================
# 11. 메인
# ============================================================
def main():

    print("=" * 80)
    print("YOLO TRAIN 데이터 100장 → 400장")
    print("=" * 80)

    # --------------------------------------------------------
    # 경로 확인
    # --------------------------------------------------------
    print()
    print("원본 이미지:")
    print(SOURCE_IMAGE_ROOT)

    print()
    print("원본 라벨 TXT:")
    print(SOURCE_TXT_ROOT)

    print()
    print("원본 라벨 XML:")
    print(SOURCE_XML_ROOT)

    print()
    print("YOLO 이미지:")
    print(YOLO_IMAGE_ROOT)

    print()
    print("YOLO 라벨:")
    print(YOLO_LABEL_ROOT)

    # --------------------------------------------------------
    # 필수 폴더 확인
    # --------------------------------------------------------
    paths = [
        SOURCE_IMAGE_ROOT,
        SOURCE_TXT_ROOT,
        SOURCE_XML_ROOT,
        YOLO_IMAGE_ROOT,
        YOLO_LABEL_ROOT,
    ]

    for path in paths:

        if not path.exists():

            print()
            print(
                f"[ERROR] 경로 없음:\n{path}"
            )

            return

    # --------------------------------------------------------
    # 랜덤 시드
    # --------------------------------------------------------
    random.seed(RANDOM_SEED)

    # --------------------------------------------------------
    # 현재 YOLO 이미지
    # --------------------------------------------------------
    existing_images = get_existing_images()

    print()
    print(
        f"현재 YOLO train 이미지 : "
        f"{len(existing_images)}개"
    )

    # --------------------------------------------------------
    # 원본 라벨 map
    # --------------------------------------------------------
    txt_map, xml_map = build_label_maps()

    # ========================================================
    # 클래스별 추가 데이터 선정
    # ========================================================
    selections = {}

    total_needed = 0

    for code, (
        food_name,
        class_id
    ) in CLASS_MAP.items():

        print()
        print("=" * 80)
        print(
            f"{food_name} "
            f"(code={code}, class={class_id})"
        )
        print("=" * 80)

        # ----------------------------------------------------
        # 원본 폴더
        # ----------------------------------------------------
        source_folder = find_source_folder(code)

        if source_folder is None:

            print(
                "[ERROR] 원본 이미지 폴더 없음"
            )

            continue

        # ----------------------------------------------------
        # 원본 이미지
        # ----------------------------------------------------
        source_images = [
            file
            for file in source_folder.iterdir()
            if (
                file.is_file()
                and file.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        ]

        print(
            f"원본 이미지 : "
            f"{len(source_images)}개"
        )

        # ----------------------------------------------------
        # 현재 YOLO에 들어있는 이미지 중
        # 해당 클래스 원본 폴더와 겹치는 것 확인
        # ----------------------------------------------------
        source_stems = {
            file.stem
            for file in source_images
        }

        current_images = [
            stem
            for stem in existing_images
            if stem in source_stems
        ]

        current_count = len(
            current_images
        )

        print(
            f"현재 YOLO 데이터 : "
            f"{current_count}개"
        )

        # ----------------------------------------------------
        # 이미 400개 이상이면 생략
        # ----------------------------------------------------
        if current_count >= TARGET_COUNT:

            print(
                "[SKIP] 이미 400개 이상"
            )

            continue

        # ----------------------------------------------------
        # 추가 필요 수
        # ----------------------------------------------------
        need = (
            TARGET_COUNT
            - current_count
        )

        print(
            f"추가 필요 : "
            f"{need}개"
        )

        # ----------------------------------------------------
        # 현재 YOLO에 없는 원본만 후보
        # ----------------------------------------------------
        available = [
            file
            for file in source_images
            if file.stem not in existing_images
        ]

        print(
            f"추가 가능 원본 : "
            f"{len(available)}개"
        )

        # ----------------------------------------------------
        # 원본 부족
        # ----------------------------------------------------
        if len(available) < need:

            print(
                "[ERROR] 400개를 만들기에 "
                "원본 이미지가 부족합니다."
            )

            continue

        # ----------------------------------------------------
        # 랜덤 300개 선택
        # ----------------------------------------------------
        selected = random.sample(
            available,
            need
        )

        selections[code] = (
            food_name,
            class_id,
            selected
        )

        total_needed += need

        print(
            f"랜덤 선택 완료 : "
            f"{len(selected)}개"
        )

    # ========================================================
    # 라벨 존재 여부를 먼저 검사
    # ========================================================
    needed_stems = {
        image_file.stem
        for (
            food_name,
            class_id,
            selected
        ) in selections.values()
        for image_file in selected
    }

    missing_label_files = []

    for stem in needed_stems:

        if stem not in txt_map and stem not in xml_map:

            missing_label_files.append(stem)

    if missing_label_files:

        print()
        print("=" * 80)
        print(
            f"[ERROR] 라벨을 찾지 못한 이미지: "
            f"{len(missing_label_files)}개"
        )
        print("=" * 80)

        for stem in missing_label_files[:30]:

            print(stem)

        if len(missing_label_files) > 30:

            print(
                f"... 외 "
                f"{len(missing_label_files) - 30}개"
            )

        print()
        print(
            "이미지와 라벨의 개수가 어긋나는 것을 "
            "막기 위해 작업을 중단합니다."
        )

        return

    # ========================================================
    # 실제 복사 및 라벨 생성
    # ========================================================
    copied_images = 0
    created_labels = 0

    used_txt = 0
    used_xml = 0

    print()
    print("=" * 80)
    print("실제 이미지 / 라벨 추가 시작")
    print("=" * 80)

    for code, (
        food_name,
        class_id,
        selected
    ) in selections.items():

        print()
        print(
            f"[{food_name}] "
            f"{len(selected)}개 추가"
        )

        for image_file in selected:

            # ------------------------------------------------
            # 1. 이미지 복사
            # ------------------------------------------------
            destination_image = (
                YOLO_IMAGE_ROOT
                / image_file.name
            )

            shutil.copy2(
                image_file,
                destination_image
            )

            copied_images += 1

            # ------------------------------------------------
            # 2. 라벨 저장 위치
            # ------------------------------------------------
            output_label = (
                YOLO_LABEL_ROOT
                / f"{image_file.stem}.txt"
            )

            # ------------------------------------------------
            # 3. TXT 우선
            # ------------------------------------------------
            if image_file.stem in txt_map:

                success = create_label_from_txt(
                    txt_map[image_file.stem],
                    output_label,
                    class_id
                )

                if success:

                    created_labels += 1
                    used_txt += 1

                else:

                    print(
                        f"[WARNING] TXT 변환 실패:"
                        f" {image_file.name}"
                    )

            # ------------------------------------------------
            # 4. TXT가 없으면 XML
            # ------------------------------------------------
            elif image_file.stem in xml_map:

                success = create_label_from_xml(
                    xml_map[image_file.stem],
                    output_label,
                    class_id
                )

                if success:

                    created_labels += 1
                    used_xml += 1

                else:

                    print(
                        f"[WARNING] XML 변환 실패:"
                        f" {image_file.name}"
                    )

            print(
                f"  [완료] "
                f"{image_file.name}"
                f" → class {class_id}"
            )

    # ========================================================
    # 최종 결과
    # ========================================================
    print()
    print("=" * 80)
    print("완료")
    print("=" * 80)

    print(
        f"추가한 이미지 : "
        f"{copied_images}개"
    )

    print(
        f"생성한 라벨 : "
        f"{created_labels}개"
    )

    print(
        f"TXT 사용 : "
        f"{used_txt}개"
    )

    print(
        f"XML 사용 : "
        f"{used_xml}개"
    )

    print()
    print(
        f"최종 이미지 : "
        f"{YOLO_IMAGE_ROOT}"
    )

    print(
        f"최종 라벨 : "
        f"{YOLO_LABEL_ROOT}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()