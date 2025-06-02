"""
precedents_full 3-way 벡터 컬럼 일괄 생성 스크립트
-------------------------------------------------
 * 대상 컬럼 : bsisfacts, courtdcss, relatelaword (TEXT/JSON string)
 * 벡터 컬럼 : bsisfacts_vector, courtdcss_vector, relatelaword_vector (vector(1536))
 * 임베딩   : OpenAI text-embedding-3-small (1536-d)
 * 실행 방법 :
     $ python bulk_embed_precedents.py
"""

import os, json, time, sys
import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv
from supabase import create_client, Client
from openai import OpenAI
from tqdm import tqdm
from httpx import RemoteProtocolError

# ──────────────────── 설정 ────────────────────
load_dotenv()
SUPABASE_URL         = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY")
BATCH_SIZE           = 5               # Supabase fetch size
EMBED_MODEL          = "text-embedding-3-small"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
oai                = OpenAI(api_key=OPENAI_API_KEY)

# ──────────────────── 유틸 함수 ─────────────────
def to_text(obj: Any, max_chars: int = 3000) -> str:
    """Json string → list → joined text (fallback) + 길이 제한"""
    if obj is None:
        return "[EMPTY]"
    try:
        parsed = json.loads(obj)
        text = ""
        if isinstance(parsed, list):
            text = " ".join(map(str, parsed))
        elif isinstance(parsed, dict):
            text = json.dumps(parsed, ensure_ascii=False)
        else:
            text = str(parsed)
    except Exception:
        text = str(obj)

    if not text.strip():
        return "[EMPTY]"

    # 최대 길이 제한
    return text[:max_chars]

def embed_batch(texts: List[str]) -> List[List[float]]:
    """OpenAI 임베딩 배치 호출"""
    # OpenAI 3-small: 8192 tokens, 1500r/m 제한. 필요 시 time.sleep 로 rate-limit 조절
    cleaned = [t if t.strip() else "[EMPTY]" for t in texts]
    resp = oai.embeddings.create(model=EMBED_MODEL, input=cleaned)
    return [d.embedding for d in resp.data]

# ──────────────────── 메인 루프 ─────────────────
def fetch_ids_without_vectors(limit=BATCH_SIZE) -> List[Dict]:
    """vector 컬럼이 NULL 인 row만 가져오기"""
    sel = ("id,bsisfacts,courtdcss,relatelaword")
    return (
        supabase.table("precedents_full")
        .select(sel)
        .is_("bsisfacts_vector", "null")
        .limit(limit)
        .execute()
        .data
    )

def process_batch(rows: List[Dict]):
    if not rows:
        return 0
    # ❶ 텍스트 변환
    facts_txt  = [to_text(r["bsisfacts"], max_chars=3000)      for r in rows]
    issue_txt  = [to_text(r["courtdcss"], max_chars=3000)      for r in rows]
    law_txt    = [to_text(r["relatelaword"], max_chars=3000)   for r in rows]

    # ❷ 임베딩 (3*batch → OpenAI 호출 3회)
    facts_vec  = embed_batch(facts_txt)
    issue_vec  = embed_batch(issue_txt)
    law_vec    = embed_batch(law_txt)

    # ❸ Supabase 업데이트
    updates = []
    for idx, row in enumerate(rows):
        vec_rec = {
            "id":                   row["id"],
            "bsisfacts_vector":     facts_vec[idx],
            "courtdcss_vector":     issue_vec[idx],
            "relatelaword_vector":  law_vec[idx],
        }
        updates.append(vec_rec)

    #   upsert 가 없으므로 반복 update
    for rec in tqdm(updates, desc="📤 Uploading to Supabase"):
        for attempt in range(3):  # 최대 3회 재시도
            try:
                supabase.table("precedents_full").update(rec).eq("id", rec["id"]).execute()
                break
            except RemoteProtocolError as e:
                tqdm.write(f"⚠️ 재시도 {attempt+1}회 (ID: {rec['id']}) - {str(e)}")
                time.sleep(1)
    return len(rows)

# ──────────────────── 실행 ─────────────────────
def main():
    total_done = 0
    start_time = time.time()

    while True:
        rows = fetch_ids_without_vectors(limit=BATCH_SIZE)
        if not rows:
            break
        done = process_batch(rows)
        total_done += done

        elapsed = time.time() - start_time
        avg_time = elapsed / total_done if total_done else 0
        est_total = total_done + BATCH_SIZE  # rough estimate
        eta = datetime.timedelta(seconds=round(avg_time * est_total - elapsed))
        tqdm.write(f"🔄 Embedded {done} rows (acc: {total_done}) | ETA: {eta}")

    print(f"\n✅ 전체 완료! 총 {total_done}건 벡터 생성 및 저장됨. ⏱️ 총 소요 시간: {datetime.timedelta(seconds=round(time.time() - start_time))}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️ 사용자 중단")
        sys.exit(0)