import requests
from typing import List
from dotenv import load_dotenv
import os
import json

load_dotenv(override=True)

# 관련 법령 정리
def extract_laws_from_precedents(related_laws: List[str]) -> List[str]:
    all_laws = set()
    for law in related_laws:
        if isinstance(law, str) and law.strip():
            all_laws.add(law.strip())
        else:
            raise ValueError("법령 정보가 올바르지 않습니다.")
    return list(all_laws)


# 법령 정보 가져오기 (법령 ID 등)
def get_law_info(result: List[dict], law_name: str) -> List[dict]:
    OC = os.getenv("OC")
    url = f"http://www.law.go.kr/DRF/lawSearch.do?OC={OC}&target=law&type=JSON&query={law_name}"

    response = requests.get(url)
    if response.status_code == 200:
        try:
            data = response.json()
            law_data = data.get("LawSearch", {}).get("law", [])
            if isinstance(law_data, dict):
                law_data = [law_data]  # 단일 객체일 경우 리스트로 변환

            for law in law_data:
                if law.get('법령명한글') == law_name:
                    result.append({
                        "법령명": law.get('법령명한글'),
                        "법령ID": law.get('법령ID')
                    })
            if result:
                return result
            else:
                return [{"법령명": law_name, "error": "일치하는 법령이 없습니다."}]
        except Exception as e:
            return [{"법령명": law_name, "error": f"JSON 파싱 오류: {str(e)}"}]
    else:
        return [{"법령명": law_name, "error": "법령 정보를 가져올 수 없습니다."}]


# 법령 본문(조문제목) 가져오기
def get_law_text(law_id: str) -> List[str]:
    OC = os.getenv("OC")
    url = f"http://www.law.go.kr/DRF/lawService.do?OC={OC}&target=law&ID={law_id}&type=JSON"

    response = requests.get(url)
    if response.status_code == 200:
        try:
            data = response.json()
            content = data['법령']['조문']['조문단위']
            law_titles = [item['조문제목'] for item in content if '조문제목' in item]
            return law_titles
        except Exception as e:
            return [f"JSON 파싱 오류: {str(e)}"]
    else:
        return [f"법령 본문을 가져올 수 없습니다. (응답 코드: {response.status_code})"]


# 실행 함수
def find_relevant_laws(laws: List[str]) -> List[dict]:
    data = extract_laws_from_precedents(laws)
    result = []
    found_laws = []

    for law in data:
        get_law_info(result, law)

    for law in result:
        조문제목들 = get_law_text(law['법령ID'])
        law['조문제목'] = 조문제목들
        found_laws.append(law)

    return found_laws


# 테스트
# matches = ["주택임대차보호법", "가등기담보 등에 관한 법률"]
# print(json.dumps(find_relevant_laws(matches), indent=2, ensure_ascii=False))
