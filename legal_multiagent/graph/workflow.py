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
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )

def create_workflow() -> StateGraph:
    workflow = StateGraph(LegalCaseState)
    langfuse_handler = get_langfuse_handler()

    # ── 노드 정의 ───────────────────────────
    workflow.add_node("ExtractBasicFacts", RunnableLambda(nodes.extract_basic_facts_node))
    workflow.add_node("ExtractLegalIssue", RunnableLambda(nodes.extract_legal_issue_node))
    workflow.add_node("ClassifyCaseType", RunnableLambda(nodes.classify_legal_domains_node))
    workflow.add_node("RecommendLaw", RunnableLambda(nodes.recommend_law_node))
    workflow.add_node("SummarizePrecedents", RunnableLambda(nodes.summarize_precedents_node))
    workflow.add_node("GenerateConclusionAndSentencing", RunnableLambda(nodes.generate_conclusion_and_sentencing_node)) # 🆕
    workflow.add_node("GenerateFinalAnswer", RunnableLambda(nodes.generate_final_answer_node))

    # ── 엣지 연결 (선형 흐름) ──────────────────
    workflow.set_entry_point("ClassifyCaseType")
    workflow.add_edge("ClassifyCaseType", "ExtractBasicFacts")
    workflow.add_edge("ExtractBasicFacts", "ExtractLegalIssue")
    workflow.add_edge("ExtractLegalIssue", "RecommendLaw")
    workflow.add_edge("RecommendLaw", "SummarizePrecedents")
    workflow.add_edge("SummarizePrecedents", "GenerateConclusionAndSentencing")
    workflow.add_edge("GenerateConclusionAndSentencing", "GenerateFinalAnswer")
    workflow.set_finish_point("GenerateFinalAnswer")

    # ✅ Langfuse 트래킹 적용
    return workflow.compile().with_config({
        "callbacks": [langfuse_handler]
    })