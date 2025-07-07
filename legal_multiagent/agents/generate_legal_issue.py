#쟁점 분석 에이전트

from typing import List
from openai import OpenAI
import json
import os
from dotenv import load_dotenv

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

#openai
client = OpenAI(api_key = OPENAI_API_KEY)

def generate_legal_issue(description: str, basic_facts: list[str], case_categories: list[str]) -> str:
    category_str = ', '.join(case_categories)
    facts_str = "\n".join(basic_facts)
    messages = [
        {
            "role": "system",
            "content": f"""
            **역할**
            너는 주어진 ##기초 사실##과 ##사건 설명##을 바탕으로, 복수의 법적 쟁점을 분석하는 법률 전문가야.

            **분석 대상 사건 분야: {category_str}**

            **지침**
            - 반드시 아래에 제공된 **기초 사실**을 기반으로 법적 쟁점을 도출해야 해.
            - 사건 설명을 참고하여 추가적인 맥락을 파악하되, 쟁점은 기초 사실에 근거해야 한다.
            - 각 쟁점은 구체적인 법률적 표현을 포함하여 명확하게 작성해.
            - 쟁점은 실제 소송에서 다뤄질 수 있는 수준으로, 책임, 권리, 절차, 지연손해금 등도 빠짐없이 포함해.
            - 출력은 반드시 다음 형식의 JSON으로 반환:
            {{
              "legal_issue": "1. ~에 관한 쟁점\n2. ~에 관한 쟁점\n..."
            }}
            **예시**
            {{
              "legal_issue": "1. 임대차 계약에서 임차인의 보증금 반환 청구권에 관한 쟁점\n2. 계약 해지에 따른 손해배상 책임\n3. 임대인의 계약상 의무 위반 여부"
            }}
            """
        },
        {
            "role": "user",
            "content": f"""## 기초 사실
{facts_str}

## 사건 설명
{description}"""
        }
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        raw = response.choices[0].message.content
        parsed = json.loads(raw)
        return parsed.get("legal_issue", "")
    except Exception as e:
        print("❌ 오류 발생:", e)
        return ""