import os


DATASET_ROOT = r'C:\Users\user\Desktop\team_project2\YOLODataset'

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')

NUM_CLASSES = 14


def count_files(path, extensions):

    count = 0

    for _, _, filenames in os.walk(path):

        for filename in filenames:

            if filename.lower().endswith(extensions):
                count += 1

    return count


def check_split(split):

    image_dir = os.path.join(
        DATASET_ROOT,
        'images',
        split
    )

    label_dir = os.path.join(
        DATASET_ROOT,
        'labels',
        split
    )

    images = {
        os.path.splitext(filename)[0]
        for filename in os.listdir(image_dir)
        if filename.lower().endswith(IMAGE_EXTENSIONS)
    }

    labels = {
        os.path.splitext(filename)[0]
        for filename in os.listdir(label_dir)
        if filename.lower().endswith('.txt')
    }

    missing_labels = sorted(
        images - labels
    )

    missing_images = sorted(
        labels - images
    )

    invalid_class_lines = []

    for filename in sorted(labels):

        label_path = os.path.join(
            label_dir,
            filename + '.txt'
        )

        with open(
            label_path,
            'r',
            encoding='utf-8'
        ) as f:

            for line_no, line in enumerate(
                f,
                start=1
            ):

                values = line.strip().split()

                if not values:
                    continue

                if len(values) != 5:
                    invalid_class_lines.append(
                        (
                            filename,
                            line_no,
                            '형식'
                        )
                    )
                    continue

                try:
                    class_id = int(values[0])
                except ValueError:
                    invalid_class_lines.append(
                        (
                            filename,
                            line_no,
                            'class_id'
                        )
                    )
                    continue

                if not 0 <= class_id < NUM_CLASSES:
                    invalid_class_lines.append(
                        (
                            filename,
                            line_no,
                            f'class_id={class_id}'
                        )
                    )

    print('\n' + '=' * 60)
    print(f'{split.upper()} 확인')
    print('=' * 60)

    print(f'이미지 수: {len(images)}')
    print(f'라벨 수  : {len(labels)}')

    print(
        f'라벨 없는 이미지: {len(missing_labels)}'
    )

    print(
        f'이미지 없는 라벨: {len(missing_images)}'
    )

    print(
        f'잘못된 라벨 줄 : {len(invalid_class_lines)}'
    )

    if missing_labels:
        print('\n라벨 없는 이미지 예시:')
        for name in missing_labels[:10]:
            print(name)

    if missing_images:
        print('\n이미지 없는 라벨 예시:')
        for name in missing_images[:10]:
            print(name)

    if invalid_class_lines:
        print('\n잘못된 라벨 예시:')
        for item in invalid_class_lines[:10]:
            print(item)


if __name__ == '__main__':

    check_split('train')
    check_split('valid')
