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

def generate_legal_issue(description):
    messages = [
        {
            "role": "system",
            "content": """
            **역할**
            너는 민사, 형사, 행정 사건에서 발생하는 복수의 쟁점을 분석하는 법률 전문가야.  
            아래 ##사건 설명##을 보고, 관련된 **모든 법적 쟁점**을 빠짐없이, 명확하고 전문적으로 도출해줘.

            **지침**
            - 반드시 한 줄 요약이 아닌, 다양한 쟁점들을 항목별로 나열해줘.
            - 각 쟁점은 구체적인 법률적 표현을 포함하여 명확하게 작성해.
            - 쟁점은 실제 민사소송에서 다뤄질 수 있는 수준으로, 책임, 권리, 절차, 지연손해금 등도 빠짐없이 포함해.
            - 출력은 반드시 다음 형식의 JSON으로 반환:
            {
              "legal_issue": "1. ~에 관한 쟁점\\n2. ~에 관한 쟁점\\n..."
            }
            **예시**
            {
              "legal_issue": "1. 임대차 계약에서 임차인의 보증금 반환 청구권에 관한 쟁점\\n2. 계약 해지에 따른 손해배상 책임\\n3. 임대인의 계약상 의무 위반 여부"
            }
            """
        },
        {
            "role": "user",
            "content": description
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