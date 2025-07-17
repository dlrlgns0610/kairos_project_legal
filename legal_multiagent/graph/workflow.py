from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda
from .state import LegalCaseState
from . import nodes
from langfuse.callback import CallbackHandler as LangfuseCallbackHandler
import os

def get_langfuse_handler():
    return LangfuseCallbackHandler(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST"),
    )

# 사건 유형에 따라 분기하는 함수
def route_by_case_category(state: LegalCaseState):
    if "민사" in state["case_categories"]:
        return "ExtractCivilFacts"
    elif "형사" in state["case_categories"]:
        return "ExtractCriminalFacts"
    elif "행정" in state["case_categories"]:
        return "ExtractAdministrativeFacts"
    else:
        return "ExtractCivilFacts"  # 기본값

def create_workflow() -> StateGraph:
    workflow = StateGraph(LegalCaseState)
    langfuse_handler = get_langfuse_handler()

    # ── 노드 정의 ───────────────────────────
    workflow.add_node("ClassifyCaseType", RunnableLambda(nodes.classify_legal_domains_node))

    workflow.add_node("ExtractCivilFacts", RunnableLambda(nodes.extract_civil_facts_node))
    workflow.add_node("ExtractCriminalFacts", RunnableLambda(nodes.extract_criminal_facts_node))
    workflow.add_node("ExtractAdministrativeFacts", RunnableLambda(nodes.extract_administrative_facts_node))

    workflow.add_node("GenerateCivilLegalIssue", RunnableLambda(nodes.generate_civil_issue_node))
    workflow.add_node("GenerateCriminalLegalIssue", RunnableLambda(nodes.generate_criminal_issue_node))
    workflow.add_node("GenerateAdministrativeLegalIssue", RunnableLambda(nodes.generate_administrative_issue_node))

    workflow.add_node("RecommendLaw", RunnableLambda(nodes.recommend_law_node))
    workflow.add_node("FindRelevantLaw", RunnableLambda(nodes.find_relevant_law_node))
    workflow.add_node("FindExactLaw", RunnableLambda(nodes.find_exact_law_node))
    workflow.add_node("SummarizePrecedents", RunnableLambda(nodes.summarize_precedents_node))
    workflow.add_node("GenerateConclusionAndSentencing", RunnableLambda(nodes.generate_conclusion_and_sentencing_node))
    workflow.add_node("GenerateFinalAnswer", RunnableLambda(nodes.generate_final_answer_node))

    # ── 시작점 설정 ─────────────────────────
    workflow.set_entry_point("ClassifyCaseType")

    # ── 사건 유형에 따른 분기 ───────────────
    workflow.add_conditional_edges(
        "ClassifyCaseType",
        route_by_case_category
    )

    # ── 각 사건별 법적 쟁점 생성 ─────────────
    workflow.add_edge("ExtractCivilFacts", "GenerateCivilLegalIssue")
    workflow.add_edge("ExtractCriminalFacts", "GenerateCriminalLegalIssue")
    workflow.add_edge("ExtractAdministrativeFacts", "GenerateAdministrativeLegalIssue")

    # ── 쟁점 생성 후 RecommendLaw로 합류 ─────
    workflow.add_edge("GenerateCivilLegalIssue", "RecommendLaw")
    workflow.add_edge("GenerateCriminalLegalIssue", "RecommendLaw")
    workflow.add_edge("GenerateAdministrativeLegalIssue", "RecommendLaw")

    # ── 공통 후속 흐름 ─────────────────────
    workflow.add_edge("RecommendLaw", "FindRelevantLaw")
    workflow.add_edge("FindRelevantLaw", "FindExactLaw")
    workflow.add_edge("FindExactLaw", "SummarizePrecedents")
    workflow.add_edge("SummarizePrecedents", "GenerateConclusionAndSentencing")
    workflow.add_edge("GenerateConclusionAndSentencing", "GenerateFinalAnswer")

    # ── 종료점 설정 ────────────────────────
    workflow.set_finish_point("GenerateFinalAnswer")

    return workflow.compile().with_config({
        "callbacks": [langfuse_handler]
    })
