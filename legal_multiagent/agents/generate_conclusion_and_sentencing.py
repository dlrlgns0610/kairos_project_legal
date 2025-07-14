from __future__ import annotations

import os
from typing import List, Tuple
from dotenv import load_dotenv
from openai import OpenAI

# ──────────────────── 환경 설정 ────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

oai = OpenAI(api_key=OPENAI_API_KEY)

def _create_chat_completion(prompt: str, model: str = "gpt-4o", temperature: float = 0.3) -> str:
    """OpenAI Chat Completion을 호출하는 헬퍼 함수"""
    response = oai.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": "당신은 매우 유능하고 논리적인 법률 AI입니다. 주어진 지시사항을 엄격히 따라서 결과물을 생성해야 합니다."},
            {"role": "user", "content": prompt.strip()},
        ],
    )
    return response.choices[0].message.content.strip()

def generate_conclusion_and_sentencing(
    basic_facts: List[str],
    legal_issue: str,
    law_recommendation: str,
    precedent_summary: str,
    case_categories: List[str],
) -> Tuple[str, str]:
    """
    논리적 추론 및 자체 검증/교정 메커니즘을 통해 양형사유와 최종 결론을 생성합니다.
    3단계 프로세스: 초안 생성 -> 비평 -> 최종본 생성
    """
    is_criminal = "형사" in case_categories
    full_context = f"""
## 기초 사실
{basic_facts}

## 법적 쟁점
{legal_issue}

## 관련 법률 조항
{law_recommendation}

## 유사 판례 요약
{precedent_summary}
"""

    # --- 1단계: 논리적 추론(Chain-of-Thought)을 통한 초안 생성 ---
    draft_prompt = f"""
당신은 판사처럼 사고하는 AI입니다. 아래 제공된 사건 정보를 바탕으로, 다음 논리적 단계를 엄격히 따라서 결론의 '초안'을 작성하세요.

{full_context}

---
## 지시사항 (초안 작성):
1.  **사실관계 확정**: 주어진 '기초 사실'을 명확하게 재확인하고 정리하여 서술하라.
2.  **법적 쟁점 식별**: 이 사건의 핵심 법적 쟁점이 무엇인지 명시하라.
3.  **법리 적용**: '관련 법률 조항'과 '유사 판례 요약'의 법리가 이 사건의 쟁점에 어떻게 적용되는지 분석하고 설명하라. ("A 판례에 따르면...", "B 법률에 의거하여...")
4.  **결론 도출**: 위 1, 2, 3단계의 분석에 근거하여, 최종적인 법률적 판단과 결론을 내려라.
5.  **양형사유(형사사건의 경우)**: 만약 형사 사건이라면, 양형에 영향을 미칠 수 있는 사유를 별도로 구분하여 서술하라.

## 초안:
"""
    draft = _create_chat_completion(draft_prompt, temperature=0.2)

    # --- 2단계: 생성된 초안에 대한 자체 비평 ---
    critique_prompt = f"""
당신은 매우 꼼꼼한 법률 검토관입니다. 아래에 '사건 정보'와 동료가 작성한 '결론 초안'이 있습니다. 
'결론 초안'을 비판적으로 검토하고, 개선점을 찾아내세요.

{full_context}

---
## 결론 초안:
{draft}

---
## 지시사항 (비평 작성):
1.  **논리적 오류**: 초안의 논리적 비약이나 근거가 부족한 부분은 없는가?
2.  **정보 누락**: '사건 정보'에 있는 중요한 내용(사실, 판례 등) 중 초안에 빠진 것은 없는가?
3.  **명확성 및 간결성**: 더 명확하거나 간결하게 표현할 수 있는 부분은 없는가?
4.  **의미적 일치성**: 초안의 내용과 표현이 원본 '사건 정보'(특히 '유사 판례 요약')의 핵심 의미와 뉘앙스를 잘 반영하고 있는가? 의미적으로 더 가깝게 만들 수 있는 표현이 있다면 제안하라.
5.  **개선 제안**: 위의 분석을 바탕으로, 초안을 어떻게 개선하면 좋을지 구체적인 제안 목록을 작성하라.

## 비평 및 개선 제안:
"""
    critique = _create_chat_completion(critique_prompt, temperature=0.5)

    # --- 3단계: 비평을 바탕으로 최종본 생성 ---
    final_prompt = f"""
당신은 최종 판결문을 작성하는 판사 AI입니다. 아래의 '사건 정보', '결론 초안', 그리고 '비평 및 개선 제안'을 모두 종합하여, 완결성 높은 최종 결론을 작성하세요.

{full_context}

---
## 결론 초안:
{draft}

---
## 비평 및 개선 제안:
{critique}

---
## 지시사항 (최종본 작성):
1.  '결론 초안'을 기반으로 하되, '비평 및 개선 제안'의 내용을 **반드시 반영**하여 글을 수정하고 완성하라.
2.  최종 결과물은 '양형사유'와 '최종 결론' 두 부분으로 명확히 구분하여 작성하라. (형사사건이 아닐 경우 '양형사유'는 비워둔다.)
3.  **문체 모방**: 최종 결론은 실제 판결문과 같이, **객관적이고, 간결하며, 감정이 배제된 형식적인 문체**로 작성하라. 구어체나 비유적인 표현을 삼가라.
4.  독자가 이해하기 쉽도록 명확하고 논리적인 구조를 갖춰야 한다.

## 최종 양형사유 (형사사건의 경우에만 작성):

## 최종 결론:
"""
    final_output = _create_chat_completion(final_prompt, temperature=0.3)

    # --- 최종 결과물 파싱 ---
    sentencing_factors = ""
    final_conclusion = ""

    try:
        if "## 최종 양형사유" in final_output and "## 최종 결론" in final_output:
            parts = final_output.split("## 최종 결론:", 1)
            sentencing_part = parts[0].replace("## 최종 양형사유 (형사사건의 경우에만 작성):", "").strip()
            conclusion_part = parts[1].strip()
            
            if is_criminal:
                sentencing_factors = sentencing_part
            final_conclusion = conclusion_part
        elif "## 최종 결론:" in final_output:
             final_conclusion = final_output.split("## 최종 결론:", 1)[1].strip()
        else:
            final_conclusion = final_output # Fallback
    except Exception:
        final_conclusion = final_output # Parsing error fallback

    return sentencing_factors, final_conclusion
