#양형 예측 에이전트
import json
from typing import List
from openai import OpenAI
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

#openai
client = OpenAI(api_key = OPENAI_API_KEY)

# ✅ Pydantic 모델 정의
class CaseDomainResponse(BaseModel):
    domains: List[str]

class SentencePredictionResponse(BaseModel):
    predicted_sentence: str

def predict_sentence(facts, law_articles, precedent_summary):
    messages = [
        {
            "role": "system",
            "content": """
**역할**
너는 형사 사건에 대해 선고될 수 있는 **가장 현실적이고 개연성 높은 형량**을 예측하는 판사 역할이야.

**입력 정보**는 아래와 같아:
- 사건 분야 (형사)
- 기초 사실 (basic facts)
- 법적 쟁점 (legal issue)
- 판례 검색용 질의 (precedent queries)
- 사건 분야 분류 결과 (분류된 카테고리)
- 관련 판례 요약 (precedent summary)
- 적용 법령 목록 (relevant laws)
- 법원의 예상 판단 요지 (legal judgment)

너는 이 모든 정보를 통합적으로 고려해서, **선고 형량**을 구체적으로 예측해야 해.

**형식 요건**
- 반드시 형식은 아래와 같아야 함:
    - 징역 ○년 ○월
    - 징역 ○년 ○월, 집행유예 ○년
    - 벌금 ○○만원
- 형사소송 관행에 부합하도록 실제 선고문처럼 **사실관계의 중대성**, **반성 여부**, **피해 회복**, **전과 유무**, **양형기준**, **참작 사유** 등을 종합 판단해.
- 반드시 실형/집행유예/벌금 여부와 기간 또는 액수를 정확히 제시할 것
- 변호사나 법조인에게 제공할 수준의 실무적, 논리적 타당성을 갖출 것
"""
        },
        {
            "role": "user",
            "content": f"""##사실관계##\n{facts}\n\n##적용 법령##\n{law_articles}\n\n##유사 판례 요약##\n{precedent_summary}"""
        }
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.3,
        )
        output = response.choices[0].message.content
        parsed = SentencePredictionResponse.model_validate_json(output)
        return parsed.predicted_sentence
    except Exception as e:
        print("❌ 예측 실패:", e)
        return "예측 실패: 오류 발생"