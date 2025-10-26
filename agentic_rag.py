import os
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

from tools import RAGTools

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# All sensitive keys and config details should be in the .env file
# This makes the framework generic and reusable
RAG_CONFIG = {
    "MILVUS_HOST": os.getenv("MILVUS_HOST"),
    "MILVUS_PORT": os.getenv("MILVUS_PORT"),
    "MILVUS_USER": os.getenv("MILVUS_USER"),
    "MILVUS_PASSWORD": os.getenv("MILVUS_PASSWORD"),
    "MILVUS_CHUNKS_COLLECTION_NAME": os.getenv("MILVUS_CHUNKS_COLLECTION_NAME"),
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "COHERE_API_KEY": os.getenv("COHERE_API_KEY"),
}
MAX_SELF_CORRECTION_ATTEMPTS = 3

# --- State Definition ---
class AgentState(TypedDict):
    original_query: str
    chat_history: List[Dict]
    rewritten_query: str
    retrieved_docs: List[Dict]
    reranked_docs: List[Dict]
    generated_answer: Dict
    evaluation_feedback: str
    is_final: bool
    final_response: Any
    num_steps: int

# --- Agent (Node) Implementations ---
class RAGAgent:
    def __init__(self, tools: RAGTools):
        self.tools = tools

    def triage_node(self, state: AgentState) -> AgentState:
        print("--- 1. Triage Agent ---")
        query = state['original_query']
        response = self.tools.handle_small_talk(query)
        if response:
            print("Triage: Small talk or invalid query detected.")
            return {**state, "is_final": True, "final_response": {"answer": response, "sources": []}}
        print("Triage: Valid RAG query.")
        return {**state, "is_final": False, "num_steps": 1}

    def rewrite_query_node(self, state: AgentState) -> AgentState:
        print(f"--- 2. Query Rewriter Agent (Attempt {state['num_steps']}) ---")
        query = state['original_query']
        history = state['chat_history']
        feedback = state.get('evaluation_feedback')
        
        rewritten_query = self.tools.rewrite_query(query, history, feedback)
        return {**state, "rewritten_query": rewritten_query}

    def retrieve_node(self, state: AgentState) -> AgentState:
        print("--- 3. Retriever Agent ---")
        query = state['rewritten_query']
        docs = self.tools.retrieve_documents(query)
        return {**state, "retrieved_docs": docs}

    def grade_and_rank_node(self, state: AgentState) -> AgentState:
        print("--- 4. Grader & Ranker Agent ---")
        query = state['rewritten_query']
        docs = state['retrieved_docs']
        reranked_docs = self.tools.rerank_documents(query, docs)
        return {**state, "reranked_docs": reranked_docs}

    def generate_node(self, state: AgentState) -> AgentState:
        print("--- 5. Generator Agent ---")
        query = state['rewritten_query']
        context_docs = state['reranked_docs']
        answer_obj = self.tools.generate_answer(query, context_docs)
        return {**state, "generated_answer": answer_obj}
        
    def evaluate_answer_node(self, state: AgentState) -> AgentState:
        print("--- 6. Answer Evaluator Agent ---")
        query = state['original_query']
        answer_obj = state['generated_answer']
        evaluation = self.tools.evaluate_answer(query, answer_obj)
        
        # Check for loop limit
        if state['num_steps'] >= MAX_SELF_CORRECTION_ATTEMPTS:
            print("Max self-correction attempts reached. Finalizing.")
            return {**state, "evaluation_feedback": "satisfied - max attempts reached", "is_final": True}

        return {**state, "evaluation_feedback": evaluation, "num_steps": state['num_steps'] + 1}

# --- Conditional Edges ---
def decide_after_triage(state: AgentState) -> str:
    return END if state.get('is_final') else "rewrite_query"

def decide_after_grading(state: AgentState) -> str:
    if state.get('reranked_docs'):
        print("Decision: Relevant docs found. Proceeding to generation.")
        return "generate"
    else:
        print("Decision: No relevant docs found. Looping back to rewrite query.")
        return "rewrite_query"

def decide_after_evaluation(state: AgentState) -> str:
    feedback = state.get('evaluation_feedback', '')
    if 'satisfied' in feedback.lower():
        print("Decision: Answer is satisfactory. Ending workflow.")
        return END
    else:
        print("Decision: Answer is unsatisfactory. Triggering self-correction loop.")
        return "rewrite_query"

# --- Build the Graph ---
def build_agentic_graph():
    # Initialize the tools
    rag_tools = RAGTools(config=RAG_CONFIG)
    agent = RAGAgent(tools=rag_tools)
    
    # Define the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("triage", agent.triage_node)
    workflow.add_node("rewrite_query", agent.rewrite_query_node)
    workflow.add_node("retrieve", agent.retrieve_node)
    workflow.add_node("grade_and_rank", agent.grade_and_rank_node)
    workflow.add_node("generate", agent.generate_node)
    workflow.add_node("evaluate_answer", agent.evaluate_answer_node)
    
    # Set entry point
    workflow.set_entry_point("triage")
    
    # Add edges
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("retrieve", "grade_and_rank")
    workflow.add_edge("generate", "evaluate_answer")
    
    # Add conditional edges
    workflow.add_conditional_edges("triage", decide_after_triage)
    workflow.add_conditional_edges("grade_and_rank", decide_after_grading)
    workflow.add_conditional_edges("evaluate_answer", decide_after_evaluation, {
        END: END,
        "rewrite_query": "rewrite_query"
    })
    
    # Compile the graph
    return workflow.compile()

# --- Main Execution ---
if __name__ == "__main__":
    app = build_agentic_graph()
    
    # Example Invocation
    query = "What is the set point temperature for an apartment in summer?"
    chat_history = [
        {"role": "user", "content": "Tell me about MEP design"},
        {"role": "assistant", "content": "The MEP design guidelines cover HVAC, electrical, and plumbing systems. What specifically are you interested in?"}
    ]
    
    initial_state = {
        "original_query": query,
        "chat_history": chat_history,
        "evaluation_feedback": None # Start with no feedback
    }

    print(f"\n🚀 Starting Agentic RAG for query: '{query}'")
    
    final_state = app.invoke(initial_state)
    
    print("\n--- ✅ Workflow Complete ---")
    if final_state.get('final_response'):
        final_answer = final_state['final_response']
    else:
        final_answer = final_state.get('generated_answer', {"answer": "No answer could be generated."})

    import json
    print(json.dumps(final_answer, indent=2))
