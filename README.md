# Jira MCP Hybrid Agent (RAG + Transactional)

A next-generation Jira AI Assistant that combines **Semantic Search (RAG)** with **Transactional Authority (MCP)** to provide accurate, verified, and context-aware answers.

![Architecture](https://img.shields.io/badge/Architecture-Hybrid%20(MCP%20%2B%20RAG)-blueviolet)
![Status](https://img.shields.io/badge/Status-Active-success)

## 🚀 The Core Innovation: "Trust but Verify"

Most AI agents are either:
1.  **RAG Bots**: Good at searching history, but hallucinations on status (e.g., claiming a ticket is "In Progress" when it was closed yesterday).
2.  **MCP Tools**: Good at looking up specific keys (e.g., `SCRUM-12`), but fail at fuzzy questions like *"What is the status of payment?"*.

**This project solves it by using a Hybrid Engine:**

| Component | Role | What it Does |
| :--- | :--- | :--- |
| **🧠 Semantic Memory (RAG)** | *The Librarian* | Searches vector database for concepts, summaries, and history. Finds "Payment" related issues. |
| **⚡ Live Jira Data (MCP)** | *The Auditor* | Connects directly to Jira API. Verifies the **live status** of any issue found by RAG. |

### The Workflow
1.  **User Asks**: *"What is the status of the 'Payment' feature?"*
2.  **RAG Layer**: Searches embeddings → Finds `SCRUM-12`, `SCRUM-45`.
3.  **Authentication Layer**: LLM sees keys in RAG, but knows **it is forbidden** to report status from memory.
4.  **Verification Layer**: LLM calls `get_issue(SCRUM-12)` via MCP.
5.  **Synthesis**: returns *"RAG context says this is for the new gateway, and Live Data confirms it is currently DONE."*

---

## ✨ Features

-   **Hybrid "Sources Used" Panel**: Visual confirmation of where data came from (Memory vs. Live).
-   **Structure-Aware Chunking**: Vector embeds distinct "views" of issues (Business Requirements vs. Technical Dependencies vs. Timeline).
-   **Token Budgeting**: Real-time display of token usage (Prompt vs. Completion) for cost visibility.
-   **Automatic Attachment Handling**: Securely downloads and caches attachments for analysis.
-   **Lazy-Loading**: Only fetches heavy issue details when specifically relevant.

## 🛠️ Tech Stack

-   **Frontend**: Streamlit
-   **Orchestration**: OpenAI Swarm / Function Calling
-   **Protocol**: Model Context Protocol (MCP) `mcp-atlassian`
-   **Vector DB**: ChromaDB (Cloud/Local)
-   **LLM**: GPT-4o

## 📦 Installation

1.  **Clone the Repository**
    ```bash
    git clone <repo-url>
    cd jira_MCP_RAG
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment**
    Create a `.env` file:
    ```ini
    OPENAI_API_KEY="sk-..."
    ATLASSIAN_BASE_URL="https://your-domain.atlassian.net"
    ATLASSIAN_EMAIL="user@example.com"
    ATLASSIAN_API_TOKEN="<jira-api-token>"
    
    # ChromaDB (Optional for Cloud)
    CHROMA_API_KEY="..."
    CHROMA_TENANT="..."
    CHROMA_DATABASE="..."
    ```

4.  **Ingest Data (Build RAG)**
    ```bash
    python ingest.py
    ```

5.  **Run the Agent**
    ```bash
    streamlit run app.py
    ```

## 🧠 Smart Ingestion

The `ingest.py` script is not a simple text splitter. It understands Jira architecture:
-   **Deployment Inference**: Infers if code is "in production" based on `fixVersion` release status.
-   **Content Signals**: Tags issues as "Epic-like" or "Story-like" based on linguistic patterns in descriptions.
-   **Dependency Mapping**: Explicitly indexes blocking relationships.

## License
MIT
