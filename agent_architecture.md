# Agentic RAG Framework Documentation

This document outlines the architecture of our generic and reusable Agentic RAG framework, built using LangGraph. The framework is designed as a multi-agent system where each agent has a specific role, contributing to a robust, decision-driven pipeline that can reason, retrieve, generate, and self-correct.

## Core Concepts

-   **State-Driven:** The entire workflow is managed by a shared `AgentState` object. Each agent reads from and writes to this state, passing information seamlessly.
-   **Decision Pipelines:** Instead of a fixed linear flow, the framework uses conditional edges to route the process based on the quality of data at each step (e.g., query validity, document relevance, answer satisfaction).
-   **Self-Correction:** The framework can loop back to earlier stages (like query rewriting) if it determines that the retrieved context or generated answer is insufficient, mimicking a reasoning process.

## Agent Roles

Here is a breakdown of each agent's role within the framework:

### 1. Triage Agent
-   **Role:** The initial gatekeeper. It inspects the user's input to decide the appropriate course of action.
-   **Responsibilities:**
    -   Handles small talk and greetings (e.g., "hello", "how are you?").
    -   Filters out gibberish or irrelevant queries.
    -   Validates if a query is meaningful enough for the RAG pipeline.
-   **Decision:**
    -   If **Small Talk/Invalid**: Responds directly and terminates the workflow.
    -   If **Valid RAG Query**: Passes the query to the Query Rewriter to begin the RAG process.

### 2. Query Rewriter Agent
-   **Role:** The context expert. It refines the user's query to make it optimal for vector retrieval.
-   **Responsibilities:**
    -   Analyzes the conversation history to understand the context.
    -   If the current query is conversational (e.g., "what about for villas?"), it incorporates historical context to create a standalone, explicit query.
    -   If the workflow loops back for self-correction, it incorporates feedback (e.g., "the previous context was irrelevant") to reformulate the query.
-   **Output:** A `rewritten_query` that is clear, context-rich, and ready for retrieval.

### 3. Retriever Agent
-   **Role:** The librarian. It fetches relevant documents from the multi-modal vector database (Milvus).
-   **Responsibilities:**
    -   Takes the rewritten query and generates a vector embedding.
    -   Performs a unified search across all configured collections (text, tables, images).
    -   Collects all candidate documents from different modalities into a single pool.
-   **Output:** A list of `retrieved_docs` with their content and metadata.

### 4. Grader & Ranker Agent
-   **Role:** The quality control specialist. It assesses the relevance of the retrieved documents and prioritizes the best ones.
-   **Responsibilities:**
    -   Uses a powerful reranker model (like Cohere Rerank) to score the relevance of each retrieved document against the query.
    -   Filters out documents that fall below a relevance threshold.
    -   Sorts the remaining documents from most to least relevant.
-   **Decision:**
    -   If **Relevant Docs Found**: Passes the high-quality `reranked_docs` to the Generator.
    -   If **No Relevant Docs Found**: Triggers a self-correction loop by sending feedback back to the Query Rewriter to try a different approach.

### 5. Generator Agent
-   **Role:** The synthesizer. It crafts a comprehensive answer based on the high-quality context provided.
-   **Responsibilities:**
    -   Constructs a detailed prompt for a generator LLM (e.g., GPT-4o) containing the user's query and the reranked context.
    -   Generates a final answer, identifies the source documents used, and suggests follow-up questions.
-   **Output:** A structured `generated_answer` object containing the answer text, sources, and follow-ups.

### 6. Answer Evaluator Agent
-   **Role:** The final critic. It judges whether the generated answer satisfactorily addresses the user's query.
-   **Responsibilities:**
    -   Compares the `generated_answer` against the `original_query`.
    -   Uses an LLM to check for correctness, completeness, and hallucination.
-   **Decision (The Self-Correction Loop):**
    -   If **Answer is Satisfactory**: The workflow terminates successfully, returning the answer to the user.
    -   If **Answer is Unsatisfactory**: Provides `evaluation_feedback` (e.g., "The answer missed key details") and routes the workflow back to the Query Rewriter to start a new attempt.

## Agentic Flow Diagram

```mermaid
graph TD
    A[Start] --> B(Triage Agent);
    B -- Valid RAG Query --> C(Query Rewriter);
    B -- Small Talk/Invalid --> G(End);
    C --> D(Retriever Agent);
    D --> E(Grader & Ranker Agent);
    E -- Relevant Docs --> F(Generator Agent);
    F --> H(Answer Evaluator Agent);
    H -- Answer Satisfactory --> G;
    H -- Answer Unsatisfactory --> C;
    E -- No Relevant Docs --> C;
