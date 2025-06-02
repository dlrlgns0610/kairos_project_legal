# 법원 판단 시뮬레이션 에이전트

from typing import List
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

def simulate_judgment(
    facts: List[str],
    precedents_summary: str,
    law_articles: List[str],
    case_type: str,
) -> str:
    """사실관계·판례·법령을 종합해 예상 판결문(판단 이유)을 생성한다."""
    facts_text = "\n".join(facts)
    laws_text = ", ".join(law_articles)
    messages = [
        {
            "role": "system",
            "content": (
                """
                **역할**
                너는 실제 법원 판결문을 작성하는 판사야. 사건의 사실관계, 법적 쟁점, 사건 분야, 적용 법령, 그리고 유사 판례들을 모두 고려하여 최종적인 법원의 판단을 내리는 역할이야.

                **목표**
                주어진 정보만으로 민사/형사/행정 사건에 대해 판결문 중 '판단 이유' 부분을 작성하듯, 법원이 내릴 법적 판단을 구체적이고 체계적이며 매우 전문적인 논증으로 정리해.

                **출력 기준**
                - 단 하나의 결론이 아니라, 논증 전개를 포함해 실제 판결문처럼 길고 깊이 있게 작성해
                - 법적 쟁점별로 판단 근거를 구조적으로 제시해
                - 유사 판례와 법 조항의 근거를 결합해 논리적으로 설명하고, 왜 해당 사건에도 동일하게 판단해야 하는지 밝혀줘
                - 판례와의 유사성과 차이점, 법 조항의 해석 적용 과정도 자세히 설명해
                - 끝부분에 결론을 명시하되, 단순히 "책임이 인정된다" 수준이 아니라 왜 그렇게 판단하는지를 정리해
                - 문장 하나하나가 실제 법관의 언어처럼 설득력 있고 신중하게 구성되어야 함
                - 형식은 반드시 '예상 판단 요지:' 없이 판결문 판단 부분처럼 자연스럽게 시작할 것
                """
            ),
        },
        {
            "role": "user",
            "content": f"""### 사건 분야
{case_type}

### 사실관계
{facts_text}

### 유사 판례 요약
{precedents_summary}

### 적용 법령
{laws_text}""",
        },
    ]

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,
            max_tokens=800,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print("❌ 판단 요지 생성 실패:", e)
        return ""