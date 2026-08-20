def find_xml_for_image(image_file, xml_map):

    stem = image_file.stem

    # -----------------------------------------
    # 1순위: 완전히 동일한 파일명
    # -----------------------------------------
    if stem in xml_map:
        return xml_map[stem]

    # -----------------------------------------
    # 2순위: 이미지 stem이 XML stem 안에 포함
    # -----------------------------------------
    candidates = []

    for xml_stem, xml_file in xml_map.items():

        if stem in xml_stem:
            candidates.append(xml_file)

    if len(candidates) == 1:
        return candidates[0]

    # -----------------------------------------
    # 3순위: XML stem이 이미지 stem 안에 포함
    # -----------------------------------------
    candidates = []

    for xml_stem, xml_file in xml_map.items():

        if xml_stem in stem:
            candidates.append(xml_file)

    if len(candidates) == 1:
        return candidates[0]

    # -----------------------------------------
    # 4순위: 음식 코드로 검색
    # 예: 01015010
    # -----------------------------------------
    parts = stem.split("_")

    for part in parts:

        if len(part) == 8 and part.isdigit():

            code = part

            candidates = [
                xml_file
                for xml_stem, xml_file in xml_map.items()
                if code in xml_stem
            ]

            if len(candidates) == 1:
                return candidates[0]

    # -----------------------------------------
    # 매칭 실패
    # -----------------------------------------
    return None