from __future__ import annotations

import os
from typing import List, Tuple
from dotenv import load_dotenv
from openai import OpenAI

# ──────────────────── 환경 설정 ────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

oai = OpenAI(api_key=OPENAI_API_KEY)

def generate_conclusion_and_sentencing(
    basic_facts: List[str],
    legal_issue: str,
    law_recommendation: str,
    precedent_summary: str,
    case_categories: List[str],
) -> Tuple[str, str]:
    """사실관계, 쟁점, 법률, 판례를 종합하여 양형사유와 최종 결론을 생성"""

    is_criminal = "형사" in case_categories

    prompt = f"""
당신은 법률 전문가입니다. 아래 제공된 정보를 바탕으로 사건의 양형사유(형사사건의 경우)와 최종 결론을 상세하게 작성하세요.

## 기초 사실
{basic_facts}

## 법적 쟁점
{legal_issue}

## 관련 법률 조항
{law_recommendation}

## 유사 판례 요약
{precedent_summary}

--- 

## 지시사항:
1. 제공된 모든 정보를 종합적으로 고려하여 분석하세요.
2. 사건이 형사사건인 경우, 양형에 영향을 미칠 수 있는 사유(예: 범행 동기, 수단과 결과, 피해 회복 노력, 전과 유무 등)를 구체적으로 명시하세요.
3. 최종 결론은 법률 전문가의 관점에서 명확하고 간결하게 제시하세요.
4. 양형사유와 최종 결론을 구분하여 작성하세요.

## 양형사유 (형사사건의 경우에만 작성):
{'' if not is_criminal else '여기에 양형사유를 작성하세요.'}

## 최종 결론:
여기에 최종 결론을 작성하세요.
"""

    chat = oai.chat.completions.create(
        model="gpt-4o",
        temperature=0.3,
        messages=[
            {"role": "system", "content": "당신은 법률 전문가로서, 제공된 정보를 바탕으로 사건의 양형사유와 최종 결론을 상세하게 작성합니다."},
            {"role": "user",    "content": prompt.strip()},
        ],
    )

    response_text = chat.choices[0].message.content.strip()

    # 양형사유와 결론 분리 (프롬프트 형식에 따라 파싱)
    sentencing_factors = ""
    final_conclusion = ""

    if is_criminal:
        # 형사사건일 경우 양형사유와 결론을 분리
        if "## 양형사유" in response_text and "## 최종 결론" in response_text:
            parts = response_text.split("## 최종 결론:", 1)
            sentencing_factors_part = parts[0].replace("## 양형사유 (형사사건의 경우에만 작성):", "").strip()
            final_conclusion_part = parts[1].strip()
            sentencing_factors = sentencing_factors_part
            final_conclusion = final_conclusion_part
        else:
            # 파싱 실패 시 전체를 결론으로 간주
            final_conclusion = response_text
    else:
        # 형사사건이 아닐 경우 전체를 결론으로 간주
        if "## 최종 결론:" in response_text:
            final_conclusion = response_text.split("## 최종 결론:", 1)[1].strip()
        else:
            final_conclusion = response_text

    return sentencing_factors, final_conclusion
