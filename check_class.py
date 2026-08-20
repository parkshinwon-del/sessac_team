import os
from collections import Counter


# =========================================================
# 확인할 폴더
# =========================================================

SOURCE_TRAIN = r'C:\Users\user\Desktop\team_project2\label\train'
SOURCE_VALID = r'C:\Users\user\Desktop\team_project2\label\valid'

YOLO_TRAIN = r'C:\Users\user\Desktop\team_project2\YOLODataset\labels\train'
YOLO_VALID = r'C:\Users\user\Desktop\team_project2\YOLODataset\labels\valid'


# =========================================================
# 라벨 클래스 개수 세기
# =========================================================

def count_classes(label_root):

    counter = Counter()
    file_count = 0

    for dirpath, dirnames, filenames in os.walk(label_root):

        for filename in filenames:

            if not filename.lower().endswith('.txt'):
                continue

            file_count += 1

            path = os.path.join(
                dirpath,
                filename
            )

            with open(
                path,
                'r',
                encoding='utf-8'
            ) as f:

                for line in f:

                    values = line.strip().split()

                    if not values:
                        continue

                    class_id = int(values[0])

                    counter[class_id] += 1

    return file_count, counter


# =========================================================
# 결과 출력
# =========================================================

folders = {
    '원본 TRAIN': SOURCE_TRAIN,
    '원본 VALID': SOURCE_VALID,
    'YOLO TRAIN': YOLO_TRAIN,
    'YOLO VALID': YOLO_VALID,
}


for name, path in folders.items():

    print('\n' + '=' * 60)
    print(name)
    print('=' * 60)

    print('경로:')
    print(path)

    if not os.path.exists(path):
        print('❌ 폴더가 없습니다.')
        continue

    file_count, counter = count_classes(path)

    print(f'라벨 파일 수: {file_count}')
    print('클래스별 객체 수:')

    for class_id in range(14):

        print(
            f'class {class_id:2d} : '
            f'{counter[class_id]}개'
        )