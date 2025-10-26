---

### 2. Agentic Workflow Documentation

```markdown
# Agentic RAG Workflow Technical Details

This document provides a technical overview of the LangGraph-based agentic workflow, including the state management, node definitions, and conditional logic that enables its decision-making capabilities.

## 1. Agent State (`AgentState`)

The workflow is orchestrated through a central state object. This `TypedDict` ensures that data is passed between agents in a structured and predictable manner.

-   `original_query: str`: The initial query from the user.
-   `chat_history: list`: The preceding conversation turns.
-   `rewritten_query: str`: The query after being processed by the Query Rewriter.
-   `retrieved_docs: list`: A list of documents fetched by the Retriever Agent.
-   `reranked_docs: list`: The filtered and sorted list of documents from the Grader & Ranker.
-   `generated_answer: dict`: The structured answer from the Generator Agent.
-   `evaluation_feedback: str`: Critique from the Answer Evaluator to guide self-correction.
-   `is_final: bool`: A flag to indicate if the process should terminate (e.g., small talk handled).
-   `final_response: Any`: The final payload to be returned to the user.
-   `num_steps: int`: A counter to prevent infinite self-correction loops.

## 2. Nodes (Agents)

Each agent is implemented as a node in the LangGraph `StateGraph`. A node is a function that takes the current `AgentState` as input, performs its designated task, and returns a dictionary with the updated state fields.

-   `triage_node`: Calls the `handle_small_talk` tool.
-   `rewrite_query_node`: Calls the `rewrite_query` tool.
-   `retrieve_node`: Calls the `retrieve_documents` tool.
-   `grade_and_rank_node`: Calls the `rerank_documents` tool.
-   `generate_node`: Calls the `generate_answer` tool.
-   `evaluate_answer_node`: Calls the `evaluate_answer` tool.

## 3. Edges (Workflow Logic)

The power of the agentic framework comes from its conditional edges, which direct the flow based on the current state.

-   **Entry Point**: The workflow begins at the `triage_node`.

-   **Triage Logic (`decide_after_triage`)**:
    -   **Condition**: Checks the `is_final` flag in the state.
    -   **Path 1 (`__end__`)**: If `True` (small talk was handled), the graph terminates.
    -   **Path 2 (`rewrite_query`)**: If `False`, the workflow proceeds to the `rewrite_query_node`.

-   **Grader Logic (`decide_after_grading`)**:
    -   **Condition**: Checks if `reranked_docs` is empty.
    -   **Path 1 (`generate`)**: If documents exist, the workflow proceeds to the `generate_node`.
    -   **Path 2 (`rewrite_query`)**: If no relevant documents were found, the graph routes back to the `rewrite_query_node` to attempt a reformulation. This is a key self-correction path.

-   **Evaluation Logic (`decide_after_evaluation`)**:
    -   **Condition**: Checks the `evaluation_feedback` from the `evaluate_answer_node`.
    -   **Path 1 (`__end__`)**: If the feedback indicates the answer is "satisfied", the graph terminates successfully.
    -   **Path 2 (`rewrite_query`)**: If the feedback indicates dissatisfaction, the graph routes back to the `rewrite_query_node`, passing the critique to guide the next attempt. This forms the primary self-correction loop.

-   **Loop Prevention**: The `num_steps` counter in the state is incremented with each cycle. The evaluation logic will force an exit after a predefined number of attempts (e.g., 3) to prevent infinite loops, ensuring the process always terminates.
