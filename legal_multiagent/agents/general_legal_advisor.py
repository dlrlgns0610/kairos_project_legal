from __future__ import annotations

import os
from typing import List, Tuple
from dotenv import load_dotenv
from openai import OpenAI

# ──────────────────── 환경 설정 ────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

oai = OpenAI(api_key=OPENAI_API_KEY)

def general_legal_advice(
    user_input: str,
    # basic_facts: List[str],
    # legal_issue: str,
    # law_recommendation: str,
    # precedent_summary: str,
    # case_categories: List[str],
) -> str:
    """주어진 사용자 질문을 바탕으로 일반적인 법률 자문을 생성"""

    prompt = f"""
당신은 경험 많은 법률 자문가입니다. 아래 제공된 사용자 질문을 분석하여, 다음 JSON 형식에 맞춰 답변을 생성하세요.
절대 다른 설명 없이 JSON 객체만 반환해야 합니다.

## 사용자 질문
{user_input}

---

## 출력 JSON 형식:
{{
  "reconstructed_facts": "사용자의 질문을 바탕으로 핵심 사실관계를 요약 및 재구성합니다.",
  "legal_conclusion": "재구성된 사실관계와 법률 지식에 근거한 최종 법률 자문 결론을 작성합니다."
}}
"""

    chat = oai.chat.completions.create(
        model="gpt-4o",
        temperature=0.5,
        response_format={"type": "json_object"}, # JSON 출력 모드 활성화
        messages=[
            {"role": "system", "content": "당신은 법률 자문가로서, 주어진 질문에 대해 지정된 JSON 형식으로만 답변해야 합니다."},
            {"role": "user", "content": prompt.strip()},
        ],
    )

    return chat.choices[0].message.content.strip()