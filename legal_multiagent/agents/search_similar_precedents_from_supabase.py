# ──────────────────────────────────────────────
# Supabase 기반 판례 검색 에이전트 (vector-based re-ranking)
# ──────────────────────────────────────────────
"""
Prerequisite
------------
1. Supabase Postgres에 pgvector 확장 설치
   CREATE EXTENSION IF NOT EXISTS vector;
2. precedents_full 테이블에 다음 벡터 컬럼 존재
   - bsisfacts_vector    vector(1536)
   - courtdcss_vector    vector(1536)
   - relatelaword_vector vector(1536)
3. Python 패키지: openai, supabase-py, numpy, python-dotenv
"""
from __future__ import annotations

import os, json, numpy as np
from typing import List, Tuple
from functools import lru_cache
from dotenv import load_dotenv
from supabase import create_client, Client
from openai import OpenAI
from postgrest.exceptions import APIError

# ──────────────────── 환경 설정 ────────────────────
load_dotenv()
SUPABASE_URL         = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
oai              = OpenAI(api_key=OPENAI_API_KEY)

# ──────────────────── util: any → str ───────────────
def _to_text(x) -> str:
    """
    embed() 에 안전하게 넘길 수 있도록 타입을 문자열로 정규화
    • list/tuple  → 줄바꿈 join
    • dict       → pretty JSON
    • None       → ""
    • 기타       → str(x)
    """
    if x is None:
        return ""
    if isinstance(x, (list, tuple)):
        return "\n".join(map(str, x))
    if isinstance(x, dict):
        return json.dumps(x, ensure_ascii=False, indent=2)
    return str(x)

# ──────────────────── util: 임베딩 ──────────────────
@lru_cache(maxsize=1024)
def embed(text: str, model: str = "text-embedding-3-small") -> List[float]:
    """OpenAI 임베딩 호출 결과를 LRU 캐시"""
    resp = oai.embeddings.create(model=model, input=text)
    return resp.data[0].embedding    # 1536-D (3-small 기준)

# ──────────────────── main ─────────────────────────
def search_similar_precedents_from_supabase(
    *,                               # keyword-only
    basic_facts:  str | list | dict | None = None,
    legal_issue:  str | list | dict | None = None,
    related_laws: str | list | dict | None = None,
    legal_domains:     str | None = None,
    case_type:         str | None = None,
    # search params
    top_k:   int   = 5,
    w_facts: float = 0.4,
    w_issue: float = 0.4,
    w_law:   float = 0.2,
) -> Tuple[str, List[dict]]:
    """pgvector + GPT-4o 요약 기반 유사 판례 검색"""

    # 0️⃣ alias 매핑 -------------------------------------------------
    if case_type is None and legal_domains is not None:
        case_type = legal_domains
    case_type = case_type or "civil"          # 기본값

    # 👉 리스트·튜플이면 첫 번째 원소(또는 join)만 사용
    if isinstance(case_type, (list, tuple)):
        case_type = case_type[0] if case_type else "civil"
    case_type = str(case_type)                # 방어적 캐스팅

    # 🆕 영문 → 국문 매핑
    CASE_TYPE_MAP = {
        "civil"   : "민사",
        "criminal": "형사",
        "admin"   : "행정",
    }
    case_type = CASE_TYPE_MAP.get(case_type, case_type)   # 변환

    # 1️⃣ 문자열 정규화 ---------------------------------------------
    basic_facts  = _to_text(basic_facts)
    legal_issue  = _to_text(legal_issue)
    related_laws = _to_text(related_laws)

    # 2️⃣ 임베딩 ------------------------------------------------------
    fact_vec  = embed(basic_facts)
    issue_vec = embed(legal_issue)
    law_vec   = embed(related_laws)

    # 3️⃣ Supabase RPC 호출 -----------------------------------------
    payload = {
        "case_type": case_type,
        "fact_vec":  fact_vec,
        "issue_vec": issue_vec,
        "law_vec":   law_vec,
        "w_f":       w_facts,
        "w_i":       w_issue,
        "w_l":       w_law,
        "k":         top_k,
    }
    try:
        res = supabase.rpc("top_k_precedents", payload).execute()
    except APIError as e:
        # Handle Postgres statement timeout (code 57014) and other DB errors gracefully
        if getattr(e, "message", "") and "statement timeout" in str(e.message):
            return "⏰ Supabase 쿼리가 시간 초과되었습니다. 입력을 요약하거나 top_k 값을 줄여 다시 시도해 주세요.", []
        else:
            raise
    matches: List[dict] = res.data or []
    if not matches:
        return "🔍 유사한 판례를 찾을 수 없습니다.", []

    # 4️⃣ GPT-4o 판례 요약 ------------------------------------------
    cases_list = "\n".join(
        f"- 사건번호: {c['caseno']} / 사건명: {c.get('casenm','')}"
        for c in matches
    )
    summary_prompt = f"""
당신은 판례 분석 전문가입니다. 아래 사건의 사실관계·쟁점에 유사한 판례 요약을 작성하세요.

## 기초 사실
{basic_facts}

## 법적 쟁점
{legal_issue}

## 관련 조문
{related_laws}

## 유사 판례 목록
{cases_list}
"""
    chat = oai.chat.completions.create(
        model="gpt-4o",
        temperature=0.3,
        messages=[
            {"role": "system", "content": "당신은 법률 판례 요약 전문가입니다."},
            {"role": "user",    "content": summary_prompt.strip()},
        ],
    )
    summary_text = chat.choices[0].message.content.strip()
    return summary_text, matches