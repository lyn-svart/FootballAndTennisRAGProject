from ingestion import retriever
from graph.state import GraphState
from typing import Any, Dict

def retrieve(state: GraphState) -> Dict[str, Any]:
    print("----RETRIEVE----")

    question = state["question"]
    documents = retriever.invoke(question)

    return {"question": question, "documents": documents}