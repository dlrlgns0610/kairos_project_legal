import json
import sys
from typing import List
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#openai
client = OpenAI(api_key = OPENAI_API_KEY)


def recommend_relevant_laws(
    legal_issue: str,
    facts: List[str],
    case_categories: List[str]
) -> List[str]:
    system_prompt = """
**역할**
너는 법률 전문가로서, 사용자가 제공한 '법적 쟁점', '기초 사실들', '사건 분야'를 종합 분석하여 **정확하고 관련성 높은 법령 조항**을 **가능한 한 많이** 추천하는 역할을 맡았어.

**지침**
- 반드시 해당 사건의 쟁점과 사실에 관련된 조문만 추천할 것.
- 각 조문은 아래 형식으로 작성:
  "<법령명> 제X조 (조문 제목)"
- 한 조문당 한 줄씩 리스트로 작성하고, 관련이 낮거나 유사 조항은 제외할 것.
- 최대한 다양한 법령(민법, 형법, 약관법 등 포함)을 반영해도 좋지만, 쟁점과의 관련성이 가장 중요함.

**예시 출력**
[
  "민법 제618조 (임대차의 정의)",
  "민법 제623조 (임대인의 의무)",
  "민법 제750조 (불법행위)",
  "주택임대차보호법 제3조 (대항력 등)",
  "주택임대차보호법 제4조 (계약갱신과 보증금 반환)"
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
        raw = response.choices[0].message.content.strip()
        
        if not raw: # 빈 문자열인 경우
            return []

        data = json.loads(raw)

        if isinstance(data, dict) and "laws" in data:
            return data["laws"]
        if isinstance(data, list):
            return data
    except json.JSONDecodeError as e:
        print("❌ 법령 추천 실패 (JSON 파싱 오류):", e, file=sys.stderr)
        return []
    except Exception as e:
        print("❌ 법령 추천 실패 (기타 오류):", e, file=sys.stderr)
        return []