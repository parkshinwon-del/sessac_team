import os


LABEL_ROOT = r'C:\Users\user\Desktop\team_project2\label'


CLASS_MAP = {
    '01016001_김밥': 0,
    '01014001_김치볶음밥': 1,
    '01015008_삼선볶음밥': 2,
    '01014007_새우볶음밥': 3,
    '01016008_소고기김밥': 4,
    '01011001_쌀밥': 5,
    '01014008_알밥': 6,
    '01012004_영양돌솥밥': 7,
    '01014009_잡탕밥': 8,
    '01014004_전주비빔밥': 9,
    '01015003_전주콩나물국밥': 10,
    '01015010_제육덮밥': 11,
    '01016015_참치마요삼각김밥': 12,
    '01012002_콩밥': 13,
}


for split in ['train', 'valid']:

    split_dir = os.path.join(LABEL_ROOT, split)

    for dirpath, dirnames, filenames in os.walk(split_dir):

        folder_name = os.path.basename(dirpath)

        if folder_name not in CLASS_MAP:
            continue

        target_class = CLASS_MAP[folder_name]

        for filename in filenames:

            if not filename.endswith('.txt'):
                continue

            path = os.path.join(dirpath, filename)

            with open(path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]

            if not lines:
                continue

            # 현재 라벨의 class_id 목록
            class_ids = set(
                int(line.split()[0])
                for line in lines
            )

            # -----------------------------------------
            # 아직 원본 형식(0=접시, 1=음식)인 경우만 변환
            # -----------------------------------------

            if class_ids.issubset({0, 1}):

                new_lines = []

                for line in lines:

                    values = line.split()

                    old_class = int(values[0])

                    # 0 = 접시 → 삭제
                    if old_class == 0:
                        continue

                    # 1 = 음식 → 폴더 클래스 번호
                    if old_class == 1:
                        values[0] = str(target_class)
                        new_lines.append(' '.join(values))

                with open(path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(new_lines))

                print(
                    f'수정: {split} / {folder_name} / {filename}'
                    f' → class {target_class}'
                )

print('라벨 변환 완료')