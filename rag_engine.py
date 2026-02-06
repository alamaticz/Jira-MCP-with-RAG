import os
import chromadb
from chromadb.utils import embedding_functions
from collections import defaultdict
import openai
from dotenv import load_dotenv

load_dotenv()

# Setup Chroma
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")
CHROMA_COLLECTION = "jira_issues_rag"

client = chromadb.CloudClient(
    api_key=CHROMA_API_KEY,
    tenant=CHROMA_TENANT,
    database=CHROMA_DATABASE,
)
embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-large"
)

collection = client.get_collection(
    name=CHROMA_COLLECTION,
    embedding_function=embedding_fn
)

# Setup OpenAI
client_openai = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))



import re

def retrieve(query, n_results=10):
    """
    Retrieve relevant chunks from ChromaDB.
    Implements Hybrid Search:
    1. Semantic Search (vector similarity)
    2. Direct Key Lookup (if query contains "HRLIF-123")
    """
    
    # 1. Semantic Search
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    chunks = []
    seen_ids = set()

    # Process Semantic Results
    if results["documents"]:
        for i, doc in enumerate(results["documents"][0]):
            uid = results["ids"][0][i]
            if uid not in seen_ids:
                meta = results["metadatas"][0][i]
                chunks.append({
                    "id": uid,
                    "document": doc,
                    "metadata": meta
                })
                seen_ids.add(uid)

    # 2. Direct Key Lookup (Regex)
    # Extract strings like "HRLIF-1234"
    issue_keys = re.findall(r"HRLIF-\d+", query.upper())
    
    if issue_keys:
        print(f"🔎 Detected Issue Keys: {issue_keys}")
        for key in issue_keys:
            # Fetch ALL chunks for this key
            direct_results = collection.get(
                where={"issue_key": key}
            )
            
            if direct_results["ids"]:
                for i, uid in enumerate(direct_results["ids"]):
                    if uid not in seen_ids:
                        chunks.append({
                            "id": uid,
                            "document": direct_results["documents"][i],
                            "metadata": direct_results["metadatas"][i]
                        })
                        seen_ids.add(uid)

    return chunks

def group_by_issue(chunks):
    """
    Group chunks by issue_key to keep context clean.
    """
    grouped = defaultdict(list)
    for c in chunks:
        grouped[c["metadata"]["issue_key"]].append(c)
    return grouped

def generate_answer(query, chunks):
    """
    Generate an answer using OpenAI based on retrieved chunks.
    """
    # Group chunks to avoid cross-contamination
    grouped_chunks = group_by_issue(chunks)
    
    context_parts = []
    
    for issue_key, issue_chunks in grouped_chunks.items():
        # Header for the Issue
        context_parts.append(f"=== ISSUE: {issue_key} ===")
        
        for c in issue_chunks:
            meta = c['metadata']
            chunk_type = meta.get('chunk_type', 'unknown')
            
            # Format Source Header
            header = f"--- Type: {chunk_type} ---"
            content = c['document']
            context_parts.append(f"{header}\n{content}")
            
        context_parts.append(f"=== END ISSUE {issue_key} ===\n")

    context = "\n".join(context_parts)

    system_prompt = """
    You are a Jira Data Analysis Agent.

    You MUST answer strictly using the provided context and its METADATA.
    Do NOT infer, guess, or generalize beyond what is explicitly present.

    The context consists of multiple chunks per Jira issue.
    Each chunk includes both:
    - Text content
    - Metadata fields (issue_key, issue_type, status, epic_key, chunk_type)

    ----------------------------------
    DEFINITIONS
    ----------------------------------
    An issue is considered COMPLETED if:
    - status ∈ {Closed, Done, Resolved, Retired}

    Issue Types:
    - Epic
    - Story
    - Task
    - Bug
    (Use ONLY the value from metadata.issue_type)

    ----------------------------------
    AGGREGATION RULES (CRITICAL)
    ----------------------------------
    1. Group chunks by metadata.issue_key
    2. Treat all chunks with the same issue_key as ONE issue
    3. Read metadata FIRST, text SECOND
    4. Never assume issue type or status from text alone
    5. If metadata is missing, explicitly say so

    ----------------------------------
    WHAT TO RETURN
    ----------------------------------
    When asked for completed issues:
    1. Filter ONLY completed issues (using metadata.status)
    2. Categorize them by issue_type
    3. Under each category, list:
    - Issue Key
    - Summary (from Business chunk if available)
    - Status
    - Epic Key (if present)

    ----------------------------------
    STRICT RULES
    ----------------------------------
    - Do NOT mention chunk names, file names, or source labels
    - Do NOT claim all issues are of one type unless metadata confirms it
    - Do NOT include issues with non-completed status
    - If data is insufficient, say exactly what is missing

    ----------------------------------
    OUTPUT FORMAT (MANDATORY)
    ----------------------------------

    Completed Epics:
    - EPIC-KEY: Summary | Status

    Completed Stories:
    - STORY-KEY: Summary | Status | Epic: EPIC-KEY

    (Include other issue types only if present)
    """

    user_prompt = f"""
    Context:
    {context}

    Task:
    Using ONLY the context and metadata provided,
    list all COMPLETED Jira issues and categorize them by Issue Type.
    """

    response = client_openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content

def get_issue_memory(issue_key):
    """
    Retrieve all memory chunks for a specific issue.
    Used for focused analysis of a single ticket.
    """
    try:
        results = collection.get(where={"issue_key": issue_key})
        
        if not results["ids"]:
            return []
            
        return [
            {"document": results["documents"][i], "metadata": results["metadatas"][i]}
            for i in range(len(results["ids"]))
        ]
    except Exception as e:
        return [f"Error retrieving memory: {str(e)}"]
