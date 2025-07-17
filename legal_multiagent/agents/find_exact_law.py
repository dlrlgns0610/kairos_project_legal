#관련 조문 reasoning 에이전트

import json
from typing import List
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#openai
client = OpenAI(api_key = OPENAI_API_KEY)


def find_exact_law(
    legal_issue: str,
    facts: List[str],
    case_categories: List[str],
    law_texts: List[dict]
) -> List[str]:
    system_prompt = """
**역할**
너는 대한민국 법률 전문가로서, 아래에 주어진 '법적 쟁점', '기초 사실', '사건 분야', 그리고 '법령제목'을 바탕으로 사건에 **정확히 관련된 조문만을 선별하여 추천**하는 역할을 맡고 있어.

**지침**
- 반드시 아래에 주어진 '법령제목' 리스트 안에서만 조문을 선택해야 해.
- 각 법령에서 **관련성이 높은 조항을 가능한 한 고르게 추천**해. 특정 법률(예: 민법)에만 과도하게 의존하지 말 것.
- 조문과 사건의 **직접적인 관련성**이 있는 경우에만 포함할 것.
- **포괄적이거나 의미가 불명확한 일반 조항은 제외**할 것.
- 각 조문은 **'법령명 제조문번호 (조문 제목)'** 형식으로 출력해야 하며, '조문 제목'은 반드시 '조문제목' 필드에서 가져올 것.
- 결과는 반드시 **배열(JSON 배열)** 형식으로 출력하고, 각 항목은 한 줄에 하나의 조문만 포함할 것.

**예시 출력**
[
  "주택임대차보호법 제3조 (대항력 등)",
  "주택임대차보호법 제4조 (계약갱신과 보증금 반환)",
  "민사집행법 제5조 (채권압류의 절차)"
]


"""
    facts_summary = "\n".join(f"- {fact}" for fact in facts[:10])
    domain_text = ", ".join(case_categories)

    user_prompt = f"""
## 사건 분야 ##
{domain_text}

## 법적 쟁점 ##
{legal_issue}

## 기초 사실 ##
{facts_summary}

## 법령 본문 ##
{json.dumps(law_texts, ensure_ascii=False, indent=2)}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt.strip()}
            ],
            temperature=0.3
        )
        raw = response.choices[0].message.content
        data = json.loads(raw)

        if isinstance(data, dict) and "laws" in data:
            return data["laws"]
        if isinstance(data, list):
            return data
    except Exception as e:
        print("❌ 법령 추천 실패:", e)

    return []