import os
import json
import re
from typing import Dict, Any, List, Optional, Tuple

# Milvus
from pymilvus import connections, utility, Collection
# OpenAI
from openai import OpenAI
import cohere
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Configuration Constants ---
MAX_RERUN_ATTEMPTS = 3
TEXT_EMBEDDING_MODEL = "text-embedding-3-large"
GENERATOR_LLM = "gpt-4o"
EVALUATOR_LLM = "gpt-4o-mini"
REWRITER_LLM = "gpt-4o-mini"

class RAGTools:
    """A collection of tools for the agentic RAG framework."""

    def __init__(self, config: dict):
        self.config = config
        self.openai_client = OpenAI(api_key=config.get("OPENAI_API_KEY"))
        self.cohere_client = cohere.Client(api_key=config.get("COHERE_API_KEY"))
        self.collections: Dict[str, Collection] = {}
        self._connect_and_load_milvus()

    def _connect_and_load_milvus(self):
        """Connects to Milvus and loads all necessary collections."""
        try:
            connections.connect(
                "default",
                host=self.config['MILVUS_HOST'],
                port=self.config['MILVUS_PORT'],
                user=self.config.get('MILVUS_USER'),
                password=self.config.get('MILVUS_PASSWORD')
            )
            print(f"Connected to Milvus at {self.config['MILVUS_HOST']}:{self.config['MILVUS_PORT']}")

            collection_names = {
                "text": self.config['MILVUS_CHUNKS_COLLECTION_NAME']
                # Add other collections like 'tables', 'images' if needed
            }

            for key, name in collection_names.items():
                if name and utility.has_collection(name):
                    col_obj = Collection(name)
                    col_obj.load()
                    self.collections[key] = col_obj
                    print(f"Collection '{name}' loaded for key '{key}'.")
                else:
                    print(f"Warning: Collection '{name}' for key '{key}' not found or not configured.")
        except Exception as e:
            print(f"Error connecting to Milvus or loading collections: {e}")
            raise

    def handle_small_talk(self, query: str) -> Optional[str]:
        """Handles greetings, simple questions, or invalid input."""
        prompt = """You are a professional AI Assistant specialized in building codes. First, determine if the user query is a greeting, small talk, gibberish, or an invalid question. If it is, provide a polite, professional response. If it is a valid, meaningful question for a RAG system, respond with the exact string "rag_query".
        Examples:
        - User: "hello there" -> Response: "Hello! How can I assist you with building codes and design guidelines today?"
        - User: "asdfghjkl" -> Response: "I'm sorry, I didn't understand that. Please ask a valid question."
        - User: "what is the minimum fire rating for a wall?" -> Response: "rag_query"
        """
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.1,
            max_tokens=100
        )
        answer = response.choices[0].message.content.strip()
        return None if "rag_query" in answer.lower() else answer
    
    def rewrite_query(self, query: str, chat_history: List[Dict], feedback: Optional[str] = None) -> str:
        """Rewrites the query based on chat history and feedback for better retrieval."""
        history_str = "\n".join([f"{turn['role']}: {turn['content']}" for turn in chat_history])
        
        prompt_parts = [
            "You are a query rewriting expert for a RAG system.",
            "Analyze the 'Original Query' and the 'Conversation History'.",
            "Your task is to rewrite the query to be a standalone, clear, and specific question, optimized for vector search.",
            "If the original query is already good, you can use it as is."
        ]
        if feedback:
            prompt_parts.append(f"\nIMPORTANT: A previous attempt failed. Incorporate this feedback: '{feedback}'. Reformulate the query to address this issue.")

        prompt_parts.append(f"\nConversation History:\n---\n{history_str}\n---\n\nOriginal Query: '{query}'")
        prompt_parts.append("\n\nRewritten Query:")
        
        system_prompt = "\n".join(prompt_parts)
        
        response = self.openai_client.chat.completions.create(
            model=REWRITER_LLM,
            messages=[{"role": "system", "content": system_prompt}],
            temperature=0,
        )
        rewritten = response.choices[0].message.content.strip()
        print(f"Original Query: '{query}' -> Rewritten Query: '{rewritten}'")
        return rewritten

    def _get_embedding(self, text: str) -> List[float]:
        """Generates an embedding for the given text."""
        response = self.openai_client.embeddings.create(input=[text], model=TEXT_EMBEDDING_MODEL)
        return response.data[0].embedding

    def retrieve_documents(self, query: str) -> List[Dict]:
        """Retrieves documents from all available Milvus collections."""
        if not self.collections:
            print("No Milvus collections loaded. Cannot retrieve.")
            return []

        query_vector = self._get_embedding(query)
        all_docs = []

        # Retrieve from text collection
        if "text" in self.collections:
            text_collection = self.collections["text"]
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 32}}
            results = text_collection.search(
                data=[query_vector],
                anns_field="embedding",
                param=search_params,
                limit=15, # Retrieve more candidates for reranking
                output_fields=["chunk_id", "content", "document_name", "page_number"]
            )
            for hit in results[0]:
                doc = hit.entity.to_dict()
                doc['score'] = 1 - hit.distance
                doc['type'] = 'text'
                all_docs.append(doc)

        # Add logic for table/image retrieval here if needed

        print(f"Retrieved {len(all_docs)} total candidate documents.")
        return all_docs
        
    def rerank_documents(self, query: str, documents: List[Dict]) -> List[Dict]:
        """Reranks documents using Cohere API and filters for relevance."""
        if not documents:
            return []
        
        doc_texts = [doc.get('content', '') for doc in documents]
        
        try:
            rerank_results = self.cohere_client.rerank(
                model="rerank-english-v3.0", # or v2.0
                query=query,
                documents=doc_texts,
                top_n=5 # Keep top 5 most relevant
            )
            
            reranked_docs = []
            for result in rerank_results.results:
                if result.relevance_score > 0.5: # Relevance threshold
                    original_doc = documents[result.index]
                    original_doc['rerank_score'] = result.relevance_score
                    reranked_docs.append(original_doc)
            
            print(f"Reranked documents. Kept {len(reranked_docs)} docs above threshold.")
            return reranked_docs
        except Exception as e:
            print(f"Error during Cohere reranking: {e}. Returning original top 5 docs.")
            return sorted(documents, key=lambda x: x.get('score', 0), reverse=True)[:5]

    def generate_answer(self, query: str, context_docs: List[Dict]) -> Dict:
        """Generates a final answer using the provided context."""
        context_str = "\n\n---\n\n".join([
            f"Source (Document: {doc.get('document_name', 'N/A')}, Page: {doc.get('page_number', 'N/A')}):\n{doc.get('content')}"
            for doc in context_docs
        ])
        
        system_prompt = f"""You are an expert Q&A system. Your task is to answer the user's query based ONLY on the provided context.
        Synthesize a comprehensive answer from the information given in the sources.
        Cite the sources used in your answer by referencing the document name and page number.
        If the context does not contain the answer, state that clearly. Do not use external knowledge.
        After the answer, suggest 3 relevant follow-up questions based on the context.
        
        Respond in the following JSON format:
        {{
          "answer": "...",
          "sources": [{{ "document_name": "...", "page_number": "..." }}],
          "follow_up_questions": ["...", "...", "..."]
        }}
        """

        user_prompt = f"Context:\n---\n{context_str}\n---\n\nUser Query: {query}"
        
        response = self.openai_client.chat.completions.create(
            model=GENERATOR_LLM,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            return result
        except json.JSONDecodeError:
            return {"answer": "Error: Failed to generate a valid JSON response.", "sources": [], "follow_up_questions": []}

    def evaluate_answer(self, query: str, generated_answer: Dict) -> str:
        """Evaluates if the generated answer is satisfactory."""
        answer_text = generated_answer.get("answer", "")
        
        system_prompt = f"""You are an answer evaluation agent. Your task is to determine if the 'Generated Answer' is a satisfactory response to the 'Original Query'.
        A satisfactory answer is complete, accurate, and directly addresses the query without hallucinating information not present in the context.
        
        Respond with a single word:
        - 'satisfied' if the answer is good.
        - 'unsatisfied' if the answer is poor, incomplete, or irrelevant.
        
        After the single word, provide a brief critique explaining your decision.
        
        Example:
        satisfied - The answer correctly extracts the numerical value and cites the source.
        unsatisfied - The answer is too vague and does not address the main point of the query.
        """
        user_prompt = f"Original Query: {query}\n\nGenerated Answer: {answer_text}"
        
        response = self.openai_client.chat.completions.create(
            model=EVALUATOR_LLM,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
        )
        evaluation = response.choices[0].message.content.strip()
        print(f"Evaluation Result: {evaluation}")
        return evaluation
