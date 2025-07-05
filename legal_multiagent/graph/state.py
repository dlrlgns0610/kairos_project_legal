from typing import TypedDict, Optional, List

class LegalCaseState(TypedDict):
    # ───── 읽기 전용 ─────
    user_input: str                       # 최초 사용자 질문

    # ───── 각 노드에서 업데이트할 필드 ─────
    basic_facts: Optional[List[str]]      # ✦ 사실관계 추출 노드
    legal_issue: Optional[str]            # ✦ 쟁점 추출 노드
    case_categories: Optional[List[str]]  # ✦ 사건 분류 노드 (내부 사용)
    law_recommendation: Optional[str]     # ✦ 관련 조문 추천 노드
    precedent_summary: Optional[str]      # ✦ 판례 요약 노드
    precedent_cases: Optional[List[str]]  # ✦ 사건번호/사건명 추출 노드

    # 🆕 새로운 최종 결과 필드
    sentencing_factors: Optional[str]     # ✦ 양형사유 (형사사건)
    final_conclusion: Optional[str]       # ✦ 최종 결론

    final_answer: Optional[str]           # ✦ 최종 답변 조립 노드