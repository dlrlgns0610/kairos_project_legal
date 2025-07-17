import json
import sys
from .state import LegalCaseState
from ..agents.classify_legal_domains import classify_legal_domains
from ..agents.extract_civil_case_facts import extract_civil_case_facts
from ..agents.extract_criminal_case_facts import extract_criminal_case_facts
from ..agents.extract_administrative_case_facts import extract_administrative_case_facts
from ..agents.generate_civil_legal_issue import generate_legal_issue as generate_civil_legal_issue_agent
from ..agents.generate_criminal_legal_issue import generate_legal_issue as generate_criminal_legal_issue_agent
from ..agents.generate_administrative_legal_issue import generate_legal_issue as generate_administrative_legal_issue_agent
from ..agents.search_similar_precedents_from_supabase import search_similar_precedents_from_supabase
from ..agents.recommend_relevant_laws import recommend_relevant_laws
from ..agents.generate_conclusion_and_sentencing import generate_conclusion_and_sentencing
from ..agents.find_relevant_laws import find_relevant_laws
from ..agents.find_exact_law import find_exact_law

def classify_legal_domains_node(state: LegalCaseState) -> LegalCaseState:
    print("▶️ 사건 분야 분류 중...", file=sys.stderr)
    categories = classify_legal_domains(state["user_input"])
    return {"case_categories": categories}

def extract_civil_facts_node(state: LegalCaseState) -> LegalCaseState:
    print("▶️ [민사] 사실관계 추출 중...", file=sys.stderr)
    facts = extract_civil_case_facts(state["user_input"], state["case_categories"])
    return {"basic_facts": facts}

def extract_criminal_facts_node(state: LegalCaseState) -> LegalCaseState:
    print("▶️ [형사] 사실관계 추출 중...", file=sys.stderr)
    facts = extract_criminal_case_facts(state["user_input"], state["case_categories"])
    return {"basic_facts": facts}

def extract_administrative_facts_node(state: LegalCaseState) -> LegalCaseState:
    print("▶️ [행정] 사실관계 추출 중...", file=sys.stderr)
    facts = extract_administrative_case_facts(state["user_input"], state["case_categories"])
    return {"basic_facts": facts}

def generate_civil_issue_node(state: LegalCaseState) -> LegalCaseState:
    print("▶️ [민사] 법적 쟁점 생성 중...", file=sys.stderr)
    issue = generate_civil_legal_issue_agent(state["user_input"], state["basic_facts"], state["case_categories"])
    return {"legal_issue": issue}

def generate_criminal_issue_node(state: LegalCaseState) -> LegalCaseState:
    print("▶️ [형사] 법적 쟁점 생성 중...", file=sys.stderr)
    issue = generate_criminal_legal_issue_agent(state["user_input"], state["basic_facts"], state["case_categories"])
    return {"legal_issue": issue}

def generate_administrative_issue_node(state: LegalCaseState) -> LegalCaseState:
    print("▶️ [행정] 법적 쟁점 생성 중...", file=sys.stderr)
    issue = generate_administrative_legal_issue_agent(state["user_input"], state["basic_facts"], state["case_categories"])
    return {"legal_issue": issue}

def summarize_precedents_node(state: LegalCaseState) -> LegalCaseState:
    print("▶️ 유사 판례 요약 중...", file=sys.stderr)
    summary, matches = search_similar_precedents_from_supabase(
        basic_facts=state["basic_facts"],
        legal_issue=state["legal_issue"],
        related_laws=state["law_recommendation"],
        legal_domains=state["case_categories"]
    )

    # 🆕 사건번호 + 사건명 10개 추출
    cases = [
        f"{m.get('caseno','')}" + " / " + f"{m.get('casenm','')}"
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
            print("❌ JSON 파싱 실패:", e, file=sys.stderr)
            raise
    if isinstance(laws_dict, dict) and "laws" in laws_dict:
        laws = laws_dict["laws"]
    else:
        laws = laws_dict  # already a list
    return {"law_recommendation": laws}

def generate_conclusion_and_sentencing_node(state: LegalCaseState) -> LegalCaseState:
    print("▶️ 양형사유 및 결론 생성 중...", file=sys.stderr)
    sentencing_factors, final_conclusion = generate_conclusion_and_sentencing(
        basic_facts=state["basic_facts"],
        legal_issue=state["legal_issue"],
        law_recommendation=state["law_recommendation"],
        precedent_summary=state["precedent_summary"],
        case_categories=state["case_categories"],
    )
    return {
        "sentencing_factors": sentencing_factors,
        "final_conclusion": final_conclusion,
    } # ✅

def generate_final_answer_node(state: LegalCaseState) -> LegalCaseState:
    print("▶️ 최종 답변 조립 중...")
    parts = [
        f"\n\n✅ 사건 분야: {', '.join(state['case_categories'])}" if state.get("case_categories") else "",
        f"🔎 기초 사실:\n" + "\n".join(state["basic_facts"])
        if state.get("basic_facts") else "",
        f"⚖️ 법적 쟁점: {state['legal_issue']}" if state.get("legal_issue") else "",
        f"📖 적용 법령:\n" +
        ("\n".join(state["law_recommendation"])
        if isinstance(state["law_recommendation"], list)
        else state["law_recommendation"])
        if state.get("law_recommendation") else "",
        f"📝 참고 판례(사건번호 / 사건명):\n" + "\n".join(state["precedent_cases"])
        if state.get("precedent_cases") else "",
        f"📚 유사 판례: {state['precedent_summary']}"  if state.get("precedent_summary") else "",
        f"🧑‍⚖️ 최종 결론: {state['final_conclusion'].strip()}" if state.get("final_conclusion") and state["final_conclusion"].strip() else "",
    ]
    if state.get("sentencing_factors") and state["sentencing_factors"].strip():
        parts.append(f"🔐 양형사유: {state['sentencing_factors']}")
    final_output = "\n\n\n".join([p for p in parts if p])
    return {"final_answer": final_output}                   # ✅