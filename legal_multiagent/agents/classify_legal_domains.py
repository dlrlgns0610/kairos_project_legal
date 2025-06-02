# 사건 분야 분류 에이전트

from typing import List
from openai import OpenAI
import os
from dotenv import load_dotenv
import json

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

#openai
client = OpenAI(api_key = OPENAI_API_KEY)

# ✅ 분류 함수 정의
def classify_legal_domains(client_input: str) -> List[str]:
    system_prompt = """
**역할**
너는 클라이언트가 서술한 사건 설명을 읽고, 이 사건이 어느 법률 영역(민사, 형사, 행정)에 해당하는지 판단하는 법률 전문가야.

**판단 기준**
- 민사: 개인 간의 재산, 계약, 손해배상, 임대차, 부당이득, 불법행위 등
- 형사: 범죄 행위 (절도, 폭행, 사기 등)에 따른 형사처벌 또는 형사소송
- 행정: 공무원 징계, 허가취소, 세금, 행정청의 처분에 대한 불복 등 행정기관 상대 사건

**출력 형식**
다음 형식의 JSON으로 출력할 것:
{
  "domains": ["민사"]
}

예시:
{
  "domains": ["형사", "민사"]
}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": client_input}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        output = response.choices[0].message.content
        parsed = json.loads(output)
        return parsed.get("domains", [])
    except Exception as e:
        print("❌ 오류 발생:", e)
        return []