import streamlit as st
import asyncio
import os
import sys
import json
import base64
import uuid
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI
import docx
from rag_engine import retrieve as semantic_search_fn

# Load environment variables
load_dotenv()

# Page Config
st.set_page_config(page_title="Jira AI Assistant", layout="wide")

# Session State Initialization
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Sidebar Configuration
with st.sidebar:
    st.header("Configuration")
    
    # Check Credentials
    jira_url = os.getenv("ATLASSIAN_BASE_URL") or os.getenv("JIRA_URL")
    jira_email = os.getenv("ATLASSIAN_EMAIL") or os.getenv("JIRA_EMAIL")
    jira_api = os.getenv("ATLASSIAN_API_TOKEN") or os.getenv("JIRA_API_TOKEN")
    openai_key = os.getenv("OPENAI_API_KEY")

    if jira_url:
        st.success(f"Connected to: {jira_url}")
    else:
        st.error("Missing JIRA_URL / ATLASSIAN_BASE_URL")
    
    if jira_email:
        st.success(f"User: {jira_email}")
    else:
        st.error("Missing JIRA_EMAIL")

    if jira_api:
        st.success("Jira API Token Detected")
    else:
        st.error("Missing JIRA_API_TOKEN")

    if openai_key:
        st.success("OpenAI API Key Detected")
    else:
        st.error("Missing OPENAI_API_KEY")

    st.markdown("---")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Main Layout
st.title("Jira AI Assistant")
st.caption("Powered by OpenAI & MCP")

# Helper: File Preview Dialog
@st.dialog("File Preview", width="large")
def preview_file(file_path):
    try:
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            with open(file_path, "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)
            
        elif ext in ['.docx', '.doc']:
            try:
                doc = docx.Document(file_path)
                st.markdown("### Document Content")
                for para in doc.paragraphs:
                    st.markdown(para.text)
            except Exception as e:
                st.error(f"Could not read DOCX structure: {e}")
                
        elif ext in ['.txt', '.log', '.md', '.py', '.json', '.xml', '.csv', '.ssh', '.yaml', '.yml']:
            with open(file_path, "r", encoding='utf-8', errors='ignore') as f:
                content = f.read()
                if len(content) > 20000:
                    st.warning("File too large, showing first 20k chars.")
                    content = content[:20000]
                st.code(content)
                
        elif ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']:
            st.image(file_path, use_container_width=True)
            
        else:
            st.info(f"No preview available for {ext} files.")
            
    except Exception as e:
        st.error(f"Error previewing file: {str(e)}")

# Helper: Render Tool Outputs
def render_tool_outputs(tools_used, key_prefix="tool"):
    if not tools_used:
        return
        
    attachment_dirs = set()
    
    # 1. Sources Used UI (Hybrid Display)
    rag_queries = []
    live_tools = []
    
    for t in tools_used:
        if t['name'] == 'semantic_search':
            query = t['args'].get('query', 'Unknown Query')
            rag_queries.append(query)
        elif t['name'] not in ['download_attachments', 'jira_download_attachments']: 
            # Exclude utility tools from "Live Data" source list to keep it clean
            live_tools.append(f"{t['name']}")

        # Collect directories for attachment tools
        if t['name'] in ['download_attachments', 'jira_download_attachments']:
            target_dir = t['args'].get('target_dir')
            if target_dir and os.path.exists(target_dir):
                attachment_dirs.add(target_dir)

    # Render Sources Block
    if rag_queries or live_tools:
        with st.expander("🔎 Context Sources (Hybrid)", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                if rag_queries:
                    st.markdown("### 🧠 Semantic Memory (RAG)")
                    for q in rag_queries:
                        st.info(f"Searched: *'{q}'*")
                else:
                    st.markdown("### 🧠 Semantic Memory")
                    st.caption("Not used in this turn")

            with col2:
                if live_tools:
                    st.markdown("### ⚡ Live Jira Data (MCP)")
                    for tool in live_tools:
                        st.success(f"Executed: `{tool}`")
                else:
                    st.markdown("### ⚡ Live Jira Data")
                    st.caption("Not used in this turn")

    # 2. Render Attachments (Unified & Deduplicated)
    if attachment_dirs:
        all_files = []
        processed_paths = set()
        
        for adir in attachment_dirs:
            try:
                files = os.listdir(adir)
                for fname in files:
                    fpath = os.path.join(adir, fname)
                    if os.path.isdir(fpath):
                        continue
                    
                    # Deduplicate by absolute path
                    abs_path = os.path.abspath(fpath)
                    if abs_path not in processed_paths:
                        processed_paths.add(abs_path)
                        all_files.append((fname, fpath))
            except Exception:
                pass # varied permissions or issues
        
        if all_files:
            st.markdown(f"### 📎 Attachments ({len(all_files)} files)")
            
            for idx, (file_name, file_path) in enumerate(all_files):
                # Unique key: prefix + global index + filename
                unique_key = f"{key_prefix}_file_{idx}_{file_name}"
                
                with st.container(border=True):
                    col1, col2 = st.columns([1, 4])
                    
                    ext = os.path.splitext(file_name)[1].lower()
                    
                    with col1:
                        if ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']:
                            st.image(file_path, use_container_width=True)
                        elif ext == '.pdf':
                            st.markdown("📄 **PDF**")
                        elif ext in ['.txt', '.log', '.md', '.py', '.json', '.xml', '.csv']:
                            st.markdown("📝 **Text**")
                        else:
                            st.markdown("📦 **File**")
                            
                        # Download Button
                        with open(file_path, "rb") as f:
                            st.download_button(
                                label="⬇️ Download",
                                data=f,
                                file_name=file_name,
                                key=f"dl_{unique_key}",
                                use_container_width=True
                            )
                    
                    with col2:
                        st.subheader(file_name)
                        
                        # Preview Button (Modal)
                        if st.button("👁️ Preview", key=f"btn_{unique_key}"):
                            preview_file(file_path)

# Display Messages
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "tool_calls" in msg:
            render_tool_outputs(msg["tool_calls"], key_prefix=f"hist_{idx}")

# Core Logic


# Core Logic
async def run_chat_turn(user_input, chat_history):
    # Setup OpenAI
    client = AsyncOpenAI(api_key=openai_key)
    
    # Get Session ID for prompt injection
    session_id = st.session_state.session_id
    
    # Setup MCP Server Params
    python_executable = sys.executable
    server_params = StdioServerParameters(
        command=python_executable,
        args=["-c", "from mcp_atlassian import main; main()"],
        env={
            **os.environ,
            "JIRA_URL": jira_url,
            "JIRA_USERNAME": jira_email,
            "JIRA_API_TOKEN": jira_api,
            "ATLASSIAN_BASE_URL": jira_url,
            "ATLASSIAN_EMAIL": jira_email,
            "ATLASSIAN_API_TOKEN": jira_api,
        },
    )

    full_response = ""
    tool_logs = []
    token_usage = {"prompt": 0, "completion": 0, "total": 0}

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # List tools
                tools = await session.list_tools()
                openai_tools = [{
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                } for tool in tools.tools]

                # Add Semantic Search Tool
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": "semantic_search",
                        "description": "Semantic search across Jira historical memory (RAG). Use this for fuzzy concepts, intent discovery, or when no specific issue key is provided.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "The semantic query to search for."},
                                "top_k": {"type": "integer", "default": 10}
                            },
                            "required": ["query"]
                        }
                    }
                })

                # Prepare messages for OpenAI
                # Convert streamlit history to OpenAI format
                api_messages = [
                    {"role": "system", 
                    "content": f"""
                    You are an expert Jira Program Intelligence Assistant operating on large, complex,
                    enterprise-scale Jira projects using Jira MCP (Model Context Protocol).

                    Your job is NOT to answer quickly.
                    Your job is to reason correctly, deeply, and semantically using Jira data.
                    
                    You have access to a Hybrid Architecture:
                    1. Jira MCP (Transactional Truth): For exact issue datalookups, modifications, and rigid queries.
                    2. Semantic Memory (RAG): For fuzzy concepts, historical context, and "what work exists regarding X" queries.
                    
                    You MUST apply semantic reasoning on top of Jira MCP results.

                    Accuracy and correctness are more important than speed.

                    ────────────────────────────────────────────
                    SEMANTIC INTENT UNDERSTANDING (CRITICAL)
                    ────────────────────────────────────────────

                    Before taking any action, you MUST semantically interpret the user’s intent.

                    Classify the query into ONE or more of the following intent types:

                    A. Issue-Key Intent
                    - Explicit Jira keys (e.g., SCRUM-9, HRLIF-2080)
                    → Perform direct issue retrieval.

                    B. Semantic Concept Intent
                    - Domain concepts, acronyms, process names, models, business terms
                        (e.g., MDM, TMP, TMP1, TMP2, IDEF03, Credentialing, Provider, Payment, Login)
                    → Treat as semantic concepts, NOT Jira issue keys.
                    → Translate them into Jira search intent.

                    C. Relationship / Why / How Intent
                    - Questions asking for reasoning, impact, relationships, or explanation
                    → Retrieve relevant issues first, then analyze relationships and intent.

                    If the term is NOT a valid Jira issue key:
                    - NEVER say “issue does not exist”
                    - ALWAYS say:
                    “This appears to be a domain or process concept. I will search for it semantically across Jira.”

                    ────────────────────────────────────────────
                    SEMANTIC SEARCH STRATEGY (MANDATORY)
                    ────────────────────────────────────────────

                    For semantic concept intent, you MUST search Jira issues by mapping concepts to
                    multiple Jira fields:

                    Search across:
                    - Summary
                    - Description
                    - Labels
                    - Components
                    - Fix versions
                    - Linked issue summaries
                    - Attachment names (if available)

                    Examples of semantic normalization:
                    - MDM → Master Data, Provider Data, Epic MDM
                    - IDEF03 → Process model, integration definition, workflow document
                    - Payment → Card, Credit, Debit, Payment mode, Transaction
                    - TMP → TMP1, TMP2, go-live, integration flow

                    These mappings are for **search reasoning only**.
                    You MUST NOT invent results.

                    ────────────────────────────────────────────
                    CORE DATA INGESTION RULES
                    ────────────────────────────────────────────

                    You MUST ALWAYS retrieve Jira data before answering.

                    For every relevant issue, analyze:
                    - Full description (not summary only)
                    - Issue type (Epic, Story, Task, Bug, Sub-task)
                    - Current status AND workflow meaning
                    - Created, updated, and resolution dates
                    - Labels, components, fix versions, sprints, priority
                    - Assignee, reporter, and ownership history (if visible)

                    NEVER infer completion or importance without verifying Jira fields.

                    ────────────────────────────────────────────
                    RELATIONSHIP & STRUCTURE ANALYSIS
                    ────────────────────────────────────────────

                    You MUST identify and validate:
                    - Epic → Story → Task → Sub-task hierarchy
                    - Parent–child relationships
                    - Issue links (blocks, blocked by, relates to, clones, duplicates)
                    - Cross-epic dependencies

                    NEVER assume relationships unless explicitly verified via Jira links.

                    ────────────────────────────────────────────
                    PROJECT & WORKFLOW INTELLIGENCE
                    ────────────────────────────────────────────

                    Before answering, build a mental project model:
                    - What epics exist in SCRUM
                    - What capability each epic represents
                    - How stories map to epics
                    - How tasks implement stories
                    - How bugs affect delivery

                    Analyze workflow behavior:
                    - Detect stalled or blocked work
                    - Compare created vs updated timestamps
                    - Identify scope creep and rework

                    ────────────────────────────────────────────
                    DEPENDENCY & IMPACT ANALYSIS
                    ────────────────────────────────────────────

                    When answering, ALWAYS evaluate:
                    - What depends on what
                    - What is blocked or at risk
                    - Impact of delay
                    - Whether downstream work started prematurely

                    Explain dependencies clearly in natural language.

                    ────────────────────────────────────────────
                    DOMAIN & FUNCTIONAL QUESTIONS (SEMANTIC)
                    ────────────────────────────────────────────

                    When the user asks about product or system functionality
                    (e.g., “payment methods”, “login”, “provider onboarding”):

                    - Search for Epics, Stories, and Tasks that DEFINE this feature
                    - Synthesize answers from:
                    - Descriptions
                    - Acceptance criteria
                    - Comments
                    - Cite Jira issue keys used in the explanation

                    STATUS DISTINCTION IS MANDATORY:
                    - Done / Completed / Released → Feature is AVAILABLE
                    - To Do / In Progress / Backlog → Feature is PLANNED or IN DEVELOPMENT

                    NEVER claim availability unless status confirms it.

                    ────────────────────────────────────────────
                    ATTACHMENTS HANDLING (MANDATORY)
                    ────────────────────────────────────────────

                    When retrieving an issue:
                    - ALWAYS check for attachments
                    - Automatically download attachments to:
                    "./.jira_cache/{session_id}/attachments"

                    When mentioning attachments:
                    - State their purpose
                    - DO NOT claim to read PDF contents unless supported

                    ────────────────────────────────────────────
                    HYBRID EXECUTION PROTOCOL (STRICT)
                    ────────────────────────────────────────────
                    
                    RAG Data = HISTORICAL MEMORY (Potentially Stale)
                    MCP Data = LIVE TRUTH (Authoritative)
                    
                    When `semantic_search` returns issues:
                    1. You MUST treat the status/details in RAG as "unverified".
                    2. You MUST extract the Issue Keys (e.g., SCRUM-12) from the RAG result.
                    3. You MUST immediately call `get_issue` (MCP) for those specific keys to check their REAL-TIME status.
                    
                    ❌ INCORRECT BEHAVIOR:
                    "I found SCRUM-12 in RAG, and it says the status is Done." (Lazy, Dangerous)
                    
                    ✅ CORRECT BEHAVIOR:
                    "I found SCRUM-12 in RAG. Now checking its live status..." -> Calls `get_issue(SCRUM-12)` -> "Live data confirms it is In Progress."

                    You are FORBIDDEN from answering status questions using RAG data alone.
                    You MUST "Trust but Verify".

                    ────────────────────────────────────────────
                    RESPONSE CONSTRUCTION RULES
                    ────────────────────────────────────────────

                    Responses MUST be:
                    - Structured
                    - Evidence-based
                    - Explicit about assumptions
                    - Clear about data gaps

                    Preferred response structure:
                    - What it is
                    - Why it was created
                    - Current status
                    - Relationships and dependencies
                    - Risks and impact
                    - Missing or unclear data (if any)

                    If no Jira data supports the question:
                    - Say explicitly what is missing
                    - Explain why a conclusion cannot be made
                    - Suggest what should be checked in Jira

                    ────────────────────────────────────────────
                    STRICT CONSTRAINTS
                    ────────────────────────────────────────────

                    - ALWAYS use project key SCRUM
                    - NEVER fabricate Jira data
                    - NEVER hallucinate
                    - NEVER overstate certainty
                    - Use cautious language for inferred relationships

                    ────────────────────────────────────────────
                    GOAL
                    ────────────────────────────────────────────

                    Behave like:
                    - A senior program manager
                    - A delivery lead
                    - A Jira architect
                    - A dependency and risk analyst

                    Assume answers are used for:
                    - Program reviews
                    - Release planning
                    - Executive updates
                    - Root cause analysis

                    If keyword not found in summary:
                    → check labels
                    → check intent of summary
                    → classify by business meaning


                    NEVER conclude that work does not exist solely because a keyword
                    is not present verbatim in summaries.

                    You MUST:
                    - Evaluate labels
                    - Evaluate summary intent
                    - Classify issues by business function
                    -If an Epic summary explicitly names a capability (e.g., API, Web Services,
                     Reporting, Notifications), the agent MUST treat this as authoritative evidence
                     and answer directly.

                    Do NOT downgrade explicit summary text into inferred or speculative reasoning.

                    If an Epic’s summary or labels explicitly describe a capability
                    (e.g., API, Web Services, Reporting, Activation, Deactivation),
                    the agent MUST answer decisively.

                    The agent must NOT downgrade explicit Jira evidence
                    into speculative or exploratory language.

                    Correctness and semantic grounding override speed.
                    """}
                ]
                for msg in chat_history:
                    role = msg["role"]
                    content = msg["content"]
                    # If it was a tool run provided by us previously, we just show the assistant text?
                    # For simple "reconnect" architecture, we only feed text history.
                    # This implies the context of *previous* tool results is lost unless we store it as text.
                    # Improving strategy: Store full state? or just summarize.
                    # For MVP: Just append text content.
                    api_messages.append({"role": role, "content": content})
                
                # Add current user input
                api_messages.append({"role": "user", "content": user_input})

                # Call OpenAI
                # Loop for multi-step tool execution
                MAX_TURNS = 10
                turn_count = 0
                
                while turn_count < MAX_TURNS:
                    turn_count += 1
                    
                    response = await client.chat.completions.create(
                        model="gpt-4o", 
                        messages=api_messages,
                        tools=openai_tools,
                        tool_choice="auto"
                    )

                    # Track Token Usage
                    if response.usage:
                        token_usage["prompt"] += response.usage.prompt_tokens
                        token_usage["completion"] += response.usage.completion_tokens
                        token_usage["total"] += response.usage.total_tokens

                    assistant_msg = response.choices[0].message
                    api_messages.append(assistant_msg)
                    
                    if not assistant_msg.tool_calls:
                        # No tool calls, this is the final text response
                        full_response = assistant_msg.content
                        break

                    # We have tool calls, execute them
                    for tool_call in assistant_msg.tool_calls:
                        name = tool_call.function.name
                        args = tool_call.function.arguments
                        
                        try:
                            args_dict = json.loads(args)
                            
                            # Execute Tool
                            if name == "semantic_search":
                                st.toast(f"🧠 Thinking semantically: {args_dict.get('query')}")
                                result_data = semantic_search_fn(
                                    query=args_dict.get("query"),
                                    n_results=args_dict.get("top_k", 10)
                                )
                                # Serialize
                                result_str = json.dumps(result_data, indent=2, default=str)
                                
                                # Mock an MCP result structure so the loop handles it uniformly
                                class MockContent:
                                    def __init__(self, text):
                                        self.type = "text"
                                        self.text = text
                                
                                class MockResult:
                                    def __init__(self, content):
                                        self.content = content
                                
                                result = MockResult([MockContent(result_str)])
                                
                            else:
                                result = await session.call_tool(name, arguments=args_dict)
                            
                            # Format output
                            tool_output = ""
                            if result.content:
                                for c in result.content:
                                    if c.type == "text":
                                        tool_output += c.text
                                    else:
                                        tool_output += f"[{c.type}]"
                        except Exception as e:
                            tool_output = f"Error executing tool {name}: {str(e)}"
                            args_dict = {"error": "Failed to parse args or execute"}

                        tool_logs.append({
                            "name": name,
                            "args": args_dict,
                            "result": tool_output
                        })
                        
                        api_messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_output
                        })
                        
                    # Loop continues to next iteration to give AI results and see if it wants to do more.

                if turn_count >= MAX_TURNS:
                    full_response = "Error: Maximum tool execution limit reached. The task was too complex to finish in one go."

    except Exception as e:
        full_response = f"Error: {str(e)}"
    
    return full_response, tool_logs, token_usage


# Chat Input
if prompt := st.chat_input("Ask about your Jira tickets..."):
    # Add User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking & Talking to Jira..."):
            response_text, tools_used, usage_stats = asyncio.run(run_chat_turn(prompt, st.session_state.messages[:-1]))
            
            # Show tools inside the message block
            if tools_used:
                render_tool_outputs(tools_used)

            st.markdown(response_text)
            
            # Display Token Usage
            if usage_stats["total"] > 0:
                st.caption(f"🪙 **Token Usage**: {usage_stats['total']} (Prompt: {usage_stats['prompt']} | Completion: {usage_stats['completion']})")
    
    # Append to history with tool context
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "tool_calls": tools_used
    })