from __future__ import annotations

from langgraph.graph import StateGraph

from typing import TypedDict, Optional
from agents.extract_basic_facts import extract_basic_facts
from agents.classify_legal_domains import classify_legal_domains
from agents.generate_legal_issue import generate_legal_issue
from agents.generate_precedent_queries import generate_precedent_queries
from agents.search_similar_precedents_from_supabase import search_similar_precedents_from_supabase
from agents.recommend_relevant_laws import recommend_relevant_laws
from agents.simulate_judgment import simulate_judgment
from agents.predict_sentence import predict_sentence

# Langfuse CallbackHandler for Langgraph

#openai
from openai import OpenAI
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import json

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#openai
client = OpenAI(api_key = OPENAI_API_KEY)

class LegalCaseState(TypedDict):
    # ───── 읽기 전용 ─────
    user_input: str                       # 최초 사용자 질문
    case_type: Optional[str]            # 사건 유형 (민사/형사/행정)

    # ───── 각 노드에서 업데이트할 필드 ─────
    legal_issue: Optional[str]            # ✦ 이슈 추출 노드
    precedent_queries: Optional[list[str]]# ✦ 판례 검색 쿼리 생성 노드
    precedent_summary: Optional[str]      # ✦ 판례 요약 노드
    law_recommendation: Optional[str]     # ✦ 관련 조문 추천 노드
    legal_judgment_prediction: Optional[str]  # ✦ 판결 결과 예측 노드
    sentence_prediction: Optional[str]    # ✦ 형량 예측 노드
    basic_facts: Optional[list[str]]      # ✦ 사실관계 추출 노드
    case_categories: Optional[list[str]]  # ✦ 사건 분류 노드
    precedent_cases: Optional[list[str]]  # ✦ 사건번호/사건명 추출 노드
    final_answer: Optional[str]           # ✦ 최종 답변 조립 노드

workflow = StateGraph(LegalCaseState)

def classify_legal_domains_node(state: LegalCaseState) -> LegalCaseState:
    print("▶️ 사건 분야 분류 중...")
    categories = classify_legal_domains(state["user_input"])
    return {"case_categories": categories}                 # ✅

def extract_basic_facts_node(state: LegalCaseState) -> LegalCaseState:
    print("▶️ 사실관계 추출 중...")
    facts = extract_basic_facts(state["user_input"])
    return {"basic_facts": facts}                          # ✅

def extract_legal_issue_node(state: LegalCaseState) -> LegalCaseState:
    print("▶️ 법적 쟁점 추출 중...")
    issue = generate_legal_issue(state["user_input"])
    return {"legal_issue": issue}                          # ✅

def generate_precedent_queries_node(state: LegalCaseState) -> LegalCaseState:
    print("▶️ 판례 검색 쿼리 생성 중...")
    queries = generate_precedent_queries(
        legal_issue=state["legal_issue"],
        basic_facts=state["basic_facts"],
        case_categories=state["case_categories"]
    )
    return {"precedent_queries": queries}                  # ✅

def summarize_precedents_node(state: LegalCaseState) -> LegalCaseState:
    print("▶️ 유사 판례 요약 중...")
    summary, matches = search_similar_precedents_from_supabase(
        precedent_queries=state["precedent_queries"],
        legal_issue=state["legal_issue"],
        basic_facts=state["basic_facts"],
        legal_domains=state["case_categories"]
    )

    # 🆕 사건번호 + 사건명 10개 추출
    cases = [
        f"{m.get('caseno','')} / {m.get('casenm','')}"
        for m in matches[:10]
    ]
    return {
        "precedent_summary": summary,
        "precedent_matches": matches,
        "precedent_cases":   cases,   # <- 추가
    }

def recommend_law_node(state: LegalCaseState):
    print("▶️ 관련 법령 추천 중...")
    print("🧪 입력 상태 확인:", state)
    laws_dict = recommend_relevant_laws(
        legal_issue     = state["legal_issue"],
        facts           = state["basic_facts"],
        case_categories = state["case_categories"]
    )
    print("📤 GPT 응답 원문:", laws_dict)
    import json
    if isinstance(laws_dict, str):
        try:
            laws_dict = json.loads(laws_dict)
        except Exception as e:
            print("❌ JSON 파싱 실패:", e)
            raise
    if isinstance(laws_dict, dict) and "laws" in laws_dict:
        laws = laws_dict["laws"]
    else:
        laws = laws_dict  # already a list
    return {"law_recommendation": laws}

def simulate_judgment_node(state: LegalCaseState) -> LegalCaseState:
    print("▶️ 판결 결과 예측 중...")
    judgment_text = simulate_judgment(
        facts=state.get("basic_facts", []),
        precedents_summary=state.get("precedent_summary", ""),
        law_articles=state.get("law_recommendation", []),
        case_type=state.get("case_type", ""),
    ).strip()

    if not judgment_text:
        judgment_text = "판결 결과 예측을 생성하지 못했습니다."

    return {"legal_judgment_prediction": judgment_text}

def predict_sentence_node(state: LegalCaseState) -> LegalCaseState:
    print("▶️ 형량 예측 중...")
    sentence = predict_sentence(
        facts=state["basic_facts"],
        law_articles=state["law_recommendation"],
        precedent_summary=state["precedent_summary"]
    )
    return {"sentence_prediction": sentence}

def generate_final_answer_node(state: LegalCaseState) -> LegalCaseState:
    print("▶️ 최종 답변 조립 중...")
    parts = [
        f"\n\n✅ 사건 분야: {', '.join(state['case_categories'])}" if state.get("case_categories") else "",

        f"🔎 기초 사실:\n" + "\n".join(state["basic_facts"])
        if state.get("basic_facts") else "",

        f"⚖️ 법적 쟁점: {state['legal_issue']}"             if state.get("legal_issue") else "",

        f"📖 적용 법령:\n" + ("\n".join(state["law_recommendation"])
                             if isinstance(state["law_recommendation"], list)
                             else state["law_recommendation"])
        if state.get("law_recommendation") else "",

        f"📝 참고 판례(사건번호 / 사건명):\n" + "\n".join(state["precedent_cases"])
        if state.get("precedent_cases") else "",
        f"📚 유사 판례: {state['precedent_summary']}"  if state.get("precedent_summary") else "",
        f"🧑‍⚖️ 예상 판결 요지: {state['legal_judgment_prediction'].strip()}" if state.get("legal_judgment_prediction") and state["legal_judgment_prediction"].strip() else "",
    ]
    if state.get("sentence_prediction") and state["sentence_prediction"].strip():
        parts.append(f"🔐 예상 형량: {state['sentence_prediction']}")
    final_output = "\n\n\n".join([p for p in parts if p])
    return {"final_answer": final_output}                   # ✅

from langchain_core.runnables import RunnableLambda
from langfuse.callback import CallbackHandler as LangfuseCallbackHandler
import os

# Langfuse 핸들러 설정
langfuse_handler = LangfuseCallbackHandler(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
)

def create_workflow() -> StateGraph:
    workflow = StateGraph(LegalCaseState)

    # ── 1️⃣ 공통 노드 ───────────────────────────
    workflow.add_node("ExtractLegalIssue",  RunnableLambda(extract_legal_issue_node))
    workflow.add_node("ExtractBasicFacts",  RunnableLambda(extract_basic_facts_node))
    workflow.add_node("ClassifyCaseType",   RunnableLambda(classify_legal_domains_node))
    workflow.add_node("GenerateFinalAnswer", RunnableLambda(generate_final_answer_node))

    # ── 2️⃣ 브랜치 노드 ────────────────────────
    branch_labels = ["민사", "형사", "행정"]
    for label in branch_labels:
        gq   = f"GeneratePrecedentQuery_{label}"
        sum_ = f"SummarizePrecedents_{label}"
        law  = f"RecommendLaw_{label}"
        sim  = f"SimulateJudgment_{label}"
        sen  = f"PredictSentence_{label}"

        workflow.add_node(gq,  RunnableLambda(lambda s, l=label: generate_precedent_queries_node({**s, "case_type": l})))
        workflow.add_node(sum_, RunnableLambda(lambda s, l=label: summarize_precedents_node({**s, "case_type": l})))
        workflow.add_node(law, RunnableLambda(lambda s, l=label: recommend_law_node({**s, "case_type": l})))
        workflow.add_node(sim, RunnableLambda(lambda s, l=label: simulate_judgment_node({**s, "case_type": l})))
        if label == "형사":
            workflow.add_node(sen, RunnableLambda(lambda s, l=label: predict_sentence_node({**s, "case_type": l})))

        workflow.add_edge(gq,  sum_)
        workflow.add_edge(sum_, law)
        workflow.add_edge(law, sim)
        if label == "형사":
            workflow.add_edge(sim, sen)
            workflow.add_edge(sen, "GenerateFinalAnswer")
        else:
            workflow.add_edge(sim, "GenerateFinalAnswer")

    # ── 3️⃣ 사건 분류 조건부 흐름 ────────────────
    def case_router(state: LegalCaseState) -> str:
        cats = state.get("case_categories") or []
        for l in ("형사", "민사", "행정"):
            if l in cats:
                return l
        return "기타"

    workflow.add_conditional_edges(
        "ClassifyCaseType",
        case_router,
        {
            "민사":  "GeneratePrecedentQuery_민사",
            "형사":  "GeneratePrecedentQuery_형사",
            "행정":  "GeneratePrecedentQuery_행정",
            "기타":  "GenerateFinalAnswer",
        }
    )

    # ── 4️⃣ 기본 직렬 연결 ──────────────────────
    workflow.set_entry_point("ExtractLegalIssue")
    workflow.add_edge("ExtractLegalIssue", "ExtractBasicFacts")
    workflow.add_edge("ExtractBasicFacts", "ClassifyCaseType")
    workflow.set_finish_point("GenerateFinalAnswer")

    # ✅ Langfuse 트래킹 적용
    return workflow.compile().with_config({
        "callbacks": [langfuse_handler]
    })

# 그래프 인스턴스 생성
graph = create_workflow()

# 초기 상태 정의 (LegalCaseState 기준)
state = {
    "user_input": """
저는 2022년 3월 1일부터 서울시 마포구에 있는 다세대주택의 2층을 전세로 임차하여 거주해 왔습니다.  
계약 당시 보증금은 1억 5천만 원이었고, 계약기간은 2년으로 설정되어 있었으며, 2024년 2월 29일에 종료되었습니다.  
집주인과는 표준임대차계약서를 작성하였고, 보증금 반환과 관련하여 특약사항은 따로 명시하지 않았습니다.  
계약 갱신은 하지 않기로 하고, 저는 계약 종료일에 맞춰 이사를 준비하고 새로운 거처도 마련하였습니다.  

하지만 이사 하루 전인 2024년 2월 28일, 집주인이 갑자기 보증금이 당장 마련되지 않았다며 반환을 미룰 수밖에 없다고 통보해 왔습니다.  
이에 따라 저는 일단 새 집으로 이사를 완료한 후, 보증금 반환을 요청하는 내용증명을 보냈지만,  
집주인은 계속해서 “자금 사정이 어렵다”는 이유로 반환을 미루고 있습니다.  

현재 해당 주택에는 새로운 세입자도 들어오지 않은 상태이며,  
집주인은 전세보증금을 반환할 계획도, 일정도 명확하게 제시하지 않고 있습니다.  
저는 해당 보증금으로 새 집 전세자금을 충당해야 하는 상황이었기에  
현재 은행 대출을 받아 이사비용을 충당한 상태이며, 이로 인해 경제적 손해와 정신적 스트레스가 상당합니다.  

또한 집주인은 연락을 회피하고 있으며, 전화나 메시지에도 제대로 응답하지 않고 있어 자력으로 보증금을 돌려받기 어려운 상황입니다.  
이러한 상황에서 제가 취할 수 있는 법적 조치에는 어떤 것들이 있으며,  
실제로 소송을 진행하게 된다면 어떤 절차와 증거가 필요한지,  
보증금을 돌려받기까지 얼마나 걸릴 수 있는지 등 구체적인 조언을 받고 싶습니다.
"""
}

import time
print("✅ LangGraph 실행 준비 완료")
start_time = time.time()

# LangGraph 실행
result = graph.invoke(state)

# 최종 답변 출력
print(result["final_answer"])
print("\n\n⏱️ 실행 시간: {:.2f}초".format(time.time() - start_time))