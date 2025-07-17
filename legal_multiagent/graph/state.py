from typing import TypedDict, Optional, List, Dict

class LegalCaseState(TypedDict):
    # ───── 읽기 전용 ─────
    user_input: str                       # 최초 사용자 질문

    # ───── 각 노드에서 업데이트할 필드 ─────
    basic_facts: Optional[List[str]]             # ✦ 사실관계 추출 노드
    legal_issue: Optional[str]                   # ✦ 쟁점 추출 노드
    case_categories: Optional[List[str]]         # ✦ 사건 분류 노드
    law_recommendation: Optional[List[str]]      # ✦ 관련 조문 추천 노드

    relevant_laws: Optional[List[Dict]]             # ✅ 법령 ID 및 조문 전문
    exact_laws: Optional[List[str]]# ✅ GPT가 골라낸 정확한 조문 리스트

    precedent_summary: Optional[str]             # ✦ 판례 요약 노드
    precedent_cases: Optional[List[str]]         # ✦ 사건번호/사건명 추출 노드

    # 🆕 새로운 최종 결과 필드
    sentencing_factors: Optional[str]            # ✦ 양형사유
    final_conclusion: Optional[str]              # ✦ 최종 결론
    final_answer: Optional[str]                  # ✦ 최종 답변 조립 노드
