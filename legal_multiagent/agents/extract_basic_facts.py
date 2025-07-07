# 사건 분야 분류 에이전트

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


def extract_basic_facts(user_input: str, case_categories: list[str]) -> list[str]:
    category_str = ', '.join(case_categories)
    messages = [
        {
            "role": "system",
            "content": f"""
**역할**
너는 클라이언트가 작성한 사건 설명을 읽고, 그로부터 법률적 분석을 위한 **기초사실(basic facts)** 을 항목별로 추출하는 역할을 맡은 변호사야.

**분석 대상 사건 분야: {category_str}**

**지침**
- 사용자의 긴 설명에서 **사건의 발생 경위, 계약 내용, 일자, 장소, 쟁점 관련 사실**들을 논리적으로 정리해.
- 특히, 위에서 주어진 **사건 분야({category_str})와 관련된 사실관계**를 중심으로 명확하게 기술해야 해.
- 각 기초 사실은 문장 하나로 표현하고, **판례 형식의 표현 스타일**을 따를 것.
- 시간순으로 서술하되, 논리적 맥락이 흐름 있게 이어지도록 구성할 것.
- 의견, 감정, 해석은 제거하고 **객관적 사실**만 기술할 것.

**출력 형식 예시**
1. 원고는 2021년 3월 1일, 피고 소유의 서울시 강남구 소재 아파트를 전세보증금 1억 원에 임차하였다.  
2. 계약기간은 2년으로 2023년 2월 28일까지로 정해졌다.  
3. 원고는 계약 종료일에 맞추어 이사를 완료하였고, 보증금 반환을 요청하였다.  
4. 피고는 자금 사정을 이유로 보증금 반환을 거절하였다.  
5. 원고는 보증금 반환 요청을 내용증명으로 2023년 3월 5일에 통지하였다.

**출력은 반드시 다음 형식의 JSON으로 반환할 것**
{{
  "basic_facts": [
    "1. 원고는 2021년 3월 1일, 피고 소유의 서울시 강남구 소재 아파트를 전세보증금 1억 원에 임차하였다.",
    "2. 계약기간은 2년으로 2023년 2월 28일까지로 정해졌다.",
    "3. 원고는 계약 종료일에 맞추어 이사를 완료하였고, 보증금 반환을 요청하였다."
  ]
}}

**답변은 위와 같이 번호 목록 형식으로 작성해줘.**
"""
        },
        {
            "role": "user",
            "content": user_input
        }
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        output = response.choices[0].message.content
        parsed = json.loads(output)
        return parsed.get("basic_facts", [])
    except Exception as e:
        print("❌ 오류 발생:", e)
        return []