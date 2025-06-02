#판례 검색용 질의 생성 에이전트

import json
from typing import List
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

#openai
client = OpenAI(api_key = OPENAI_API_KEY)

# ✅ Pydantic 모델 정의

def generate_precedent_queries(
    legal_issue: str,
    basic_facts: list[str],
    case_categories: list[str]
) -> list[str]:
    system_prompt = """
너는 변호사이자 리걸 엔지니어로서 사용자가 제공한 '법적 쟁점', '기초 사실들', '사건 분야(민사, 형사, 행정)'을 분석하고,
그에 기반해 판례 검색 시스템에서 사용할 수 있는 **다양한 문장 형태의 판례 검색 질의문들을 생성**하는 역할이야.

**생성 원칙**
- 각 문장은 검색 시스템의 질의로 사용할 수 있는 자연어 문장이어야 한다.
- 반드시 현실의 사건을 검색하듯, **검색 키워드가 포함된 문장 형태**로 쓸 것
- 질의마다 관점(행위자, 쟁점, 청구 취지 등)을 달리해 **다양한 검색 경로**를 제시할 것
- 사건의 분야(민사/형사/행정)를 고려해 해당 법 분야 특성에 맞는 키워드와 문맥을 반영할 것

**출력 예시**
[
  "보증금 반환을 거절하는 임대인에 대한 임차인의 반환청구 관련 판례",
  "임대차 계약 종료 후 보증금을 반환하지 않은 경우의 분쟁 사례",
  "임대차 계약 만료 후 집주인의 반환 지연에 대한 손해배상 관련 판례"
]

**출력 형식**
- 각 문장 끝은 '~ 판례'로 마무리
- 반드시 ["문장1", "문장2", ...] 형식의 JSON 리스트로 출력할 것.
"""

    facts_summary = "\n".join(f"- {fact}" for fact in basic_facts[:10])
    category_summary = ", ".join(case_categories)

    user_prompt = f"""
## 사건 분야 ##
{category_summary}

## 법적 쟁점 ##
{legal_issue}

## 기초 사실 ##
{facts_summary}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt.strip()}
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )

    try:
        parsed = json.loads(response.choices[0].message.content.strip())
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            return parsed
    except:
        lines = response.choices[0].message.content.strip().splitlines()
        return [
            line.strip("-• ").strip()
            for line in lines
            if line.strip() and line.strip().endswith("판례")
        ]