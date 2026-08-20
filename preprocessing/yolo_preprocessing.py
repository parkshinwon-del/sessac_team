import os
import shutil


# =========================================================
# 현재 데이터 구조
#
# C:\Users\user\Desktop\team_project2
# ├─ image
# │  ├─ train
# │  │  ├─ 클래스폴더
# │  │  └─ ...
# │  └─ valid
# │     ├─ 클래스폴더
# │     └─ ...
# │
# └─ label
#    ├─ train
#    │  ├─ 클래스폴더
#    │  └─ ...
#    └─ valid
#       ├─ 클래스폴더
#       └─ ...
#
# ↓ 학원 수업과 같은 Ultralytics 표준 구조로 복사
#
# C:\Users\user\Desktop\team_project2\YOLODataset
# ├─ images
# │  ├─ train
# │  └─ valid
# └─ labels
#    ├─ train
#    └─ valid
# =========================================================


IMAGE_SOURCE_ROOT = r'C:\Users\user\Desktop\team_project2\image'
LABEL_SOURCE_ROOT = r'C:\Users\user\Desktop\team_project2\label'

YOLO_DATASET_ROOT = r'C:\Users\user\Desktop\team_project2\YOLODataset'


IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')


def create_yolo_directory(base_dir):

    image = os.path.join(base_dir, 'images')
    label = os.path.join(base_dir, 'labels')

    image_train = os.path.join(base_dir, 'images', 'train')
    image_valid = os.path.join(base_dir, 'images', 'valid')

    label_train = os.path.join(base_dir, 'labels', 'train')
    label_valid = os.path.join(base_dir, 'labels', 'valid')

    for path in [
        image,
        label,
        image_train,
        image_valid,
        label_train,
        label_valid
    ]:
        os.makedirs(path, exist_ok=True)
        print(f'{path} 경로 준비 완료.')


def collect_files(root_dir, extensions):

    file_list = []

    for dirpath, dirnames, filenames in os.walk(root_dir):

        for filename in filenames:

            if filename.lower().endswith(extensions):

                file_list.append(
                    os.path.join(dirpath, filename)
                )

    return file_list


def copy_flatten_dataset(
    source_root,
    target_train,
    target_valid,
    extensions
):
    """
    클래스 하위 폴더에 있는 이미지를
    YOLODataset/images/train, images/valid로 평탄화해서 복사합니다.

    파일명은 원본 그대로 유지합니다.
    같은 파일명이 두 번 나오면 데이터 혼선을 막기 위해 오류를 냅니다.
    """

    for split in ['train', 'valid']:

        source_split = os.path.join(
            source_root,
            split
        )

        if not os.path.exists(source_split):
            print(f'폴더가 없습니다: {source_split}')
            continue

        target_dir = (
            target_train if split == 'train'
            else target_valid
        )

        files = collect_files(
            source_split,
            extensions
        )

        print(f'{split}: {len(files)}개 이미지 발견')

        seen_names = set()

        for source_file in files:

            filename = os.path.basename(source_file)

            if filename in seen_names:
                raise ValueError(
                    f'같은 파일명이 여러 번 발견되었습니다: {filename}'
                )

            seen_names.add(filename)

            target_file = os.path.join(
                target_dir,
                filename
            )

            shutil.copy2(
                source_file,
                target_file
            )

        print(
            f'{split}: {len(files)}개 이미지 복사 완료'
        )


def move_datas(
    source_folder,
    destination_folder
):
    """
    학원 수업 코드와 같은 이름의 단순 복사 함수.
    """

    os.makedirs(
        destination_folder,
        exist_ok=True
    )

    for filename in os.listdir(source_folder):

        source_path = os.path.join(
            source_folder,
            filename
        )

        if os.path.isfile(source_path):

            shutil.copy2(
                source_path,
                os.path.join(
                    destination_folder,
                    filename
                )
            )


def move_label_datas(
    source_dir,
    train_image_pth,
    valid_image_pth,
    train_target,
    valid_target
):
    """
    학원 수업의 move_label_datas 흐름을 유지하면서,
    이미지 이름과 정확히 같은 라벨을 train/valid로 복사합니다.
    """

    source_files = collect_files(
        source_dir,
        ('.txt',)
    )

    train_list = set(
        os.path.splitext(x)[0]
        for x in os.listdir(train_image_pth)
        if x.lower().endswith(IMAGE_EXTENSIONS)
    )

    valid_list = set(
        os.path.splitext(x)[0]
        for x in os.listdir(valid_image_pth)
        if x.lower().endswith(IMAGE_EXTENSIONS)
    )

    os.makedirs(train_target, exist_ok=True)
    os.makedirs(valid_target, exist_ok=True)

    for source_file in source_files:

        label_name = os.path.splitext(
            os.path.basename(source_file)
        )[0]

        if label_name in train_list:

            shutil.copy2(
                source_file,
                os.path.join(
                    train_target,
                    label_name + '.txt'
                )
            )

        elif label_name in valid_list:

            shutil.copy2(
                source_file,
                os.path.join(
                    valid_target,
                    label_name + '.txt'
                )
            )


def prepare_dataset():

    print('=' * 60)
    print('YOLO Dataset 생성 시작')
    print('=' * 60)

    create_yolo_directory(
        YOLO_DATASET_ROOT
    )

    # 이미지 복사
    copy_flatten_dataset(
        IMAGE_SOURCE_ROOT,
        os.path.join(
            YOLO_DATASET_ROOT,
            'images',
            'train'
        ),
        os.path.join(
            YOLO_DATASET_ROOT,
            'images',
            'valid'
        ),
        IMAGE_EXTENSIONS
    )

    # 라벨 복사
    # 현재 최종 label/train, label/valid를 기준으로
    # 학원 수업과 같은 YOLO 표준 구조로 이동
    copy_flatten_dataset(
        LABEL_SOURCE_ROOT,
        os.path.join(
            YOLO_DATASET_ROOT,
            'labels',
            'train'
        ),
        os.path.join(
            YOLO_DATASET_ROOT,
            'labels',
            'valid'
        ),
        ('.txt',)
    )

    print('=' * 60)
    print('YOLO Dataset 생성 완료')
    print('=' * 60)


if __name__ == '__main__':
    prepare_dataset()
