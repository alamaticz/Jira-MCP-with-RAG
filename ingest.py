import json
from pathlib import Path
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")

CHROMA_COLLECTION = "jira_issues_rag"
chroma_client = chromadb.CloudClient(
    api_key=CHROMA_API_KEY,
    tenant=CHROMA_TENANT,
    database=CHROMA_DATABASE,
)

embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-large"
)

collection = chroma_client.get_or_create_collection(
    name=CHROMA_COLLECTION,
    embedding_function=embedding_fn
)

# -------------------------
# LOAD JIRA FILES
# -------------------------
def load_jira(path):
    issues = []
    for file in Path(path).glob("*.json"):
        with open(file, encoding="utf-8") as f:
            issues.append(json.load(f))
    return issues

# -------------------------
# ADF TO TEXT
# -------------------------
def adf_to_text(adf):
    """
    Convert Atlassian Document Format (ADF) to plain text.
    """
    if not adf or not isinstance(adf, dict):
        return ""

    texts = []

    def walk(node):
        if isinstance(node, dict):
            node_type = node.get("type")

            # Plain text
            if node_type == "text":
                texts.append(node.get("text", ""))

            # Mentions (@user)
            elif node_type == "mention":
                texts.append(node.get("attrs", {}).get("text", ""))

            # Recurse children
            for child in node.get("content", []):
                walk(child)

        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(adf)
    return " ".join(texts).strip()


# -------------------------
# CHUNK GENERATOR
# -------------------------
# -------------------------
# CHUNK GENERATOR (COMPREHENSIVE JIRA INTELLIGENCE)
# -------------------------
def generate_chunks(issue):
    """
    Generate semantic chunks for Jira Program & Delivery Intelligence.
    Extracts ALL required fields for content-based hierarchy reasoning,
    timeline analysis, deployment inference, and change tracking.
    """
    chunks = []
    key = issue["key"]
    fields = issue["fields"]
    
    # ========================================
    # EXTRACT ALL JIRA FIELDS (MANDATORY)
    # ========================================
    
    # CONTENT FIELDS
    summary = fields.get("summary", "")
    description = adf_to_text(fields.get("description"))
    labels = fields.get("labels", [])
    
    # STRUCTURE FIELDS
    issuetype = fields.get("issuetype") or {}
    issuetype_name = issuetype.get("name", "Unknown")
    parent = fields.get("parent") or {}
    parent_key = parent.get("key")
    
    # TIMELINE & STATUS FIELDS
    created = fields.get("created", "")
    updated = fields.get("updated", "")
    resolutiondate = fields.get("resolutiondate", "")
    status = fields.get("status") or {}
    status_name = status.get("name", "Unknown")
    status_category = status.get("statusCategory") or {}
    status_category_name = status_category.get("name", "Unknown")
    
    # DELIVERY & RELEASE FIELDS
    fix_versions = fields.get("fixVersions", [])
    fix_version_names = [v.get("name", "") for v in fix_versions]
    fix_version_details = []
    for v in fix_versions:
        fix_version_details.append({
            "name": v.get("name", ""),
            "releaseDate": v.get("releaseDate", ""),
            "released": v.get("released", False)
        })
    
    # OWNERSHIP & PRIORITY
    assignee = fields.get("assignee") or {}
    assignee_name = assignee.get("displayName", "Unassigned")
    reporter = fields.get("reporter") or {}
    reporter_name = reporter.get("displayName", "Unknown")
    priority = fields.get("priority") or {}
    priority_name = priority.get("name", "Unknown")
    
    # SPRINT (customfield_10006)
    sprint_field = fields.get("customfield_10006")
    sprint_names = []
    if sprint_field:
        if isinstance(sprint_field, list):
            for sprint in sprint_field:
                if isinstance(sprint, dict):
                    sprint_names.append(sprint.get("name", ""))
                elif isinstance(sprint, str):
                    # Some Jira instances return sprint as string
                    sprint_names.append(sprint)
        elif isinstance(sprint_field, str):
            sprint_names.append(sprint_field)
    
    # COMMENTS
    comments_data = fields.get("comment", {})
    comments_list = comments_data.get("comments", []) if isinstance(comments_data, dict) else []
    comments_text = []
    for comment in comments_list:
        author = comment.get("author", {}).get("displayName", "Unknown")
        created_date = comment.get("created", "")
        body = adf_to_text(comment.get("body"))
        if body:
            comments_text.append(f"[{author} on {created_date}]: {body}")
    
    # ATTACHMENTS
    attachments = fields.get("attachment", [])
    attachment_details = []
    for att in attachments:
        attachment_details.append({
            "filename": att.get("filename", ""),
            "created": att.get("created", ""),
            "mimeType": att.get("mimeType", "")
        })
    
    # ISSUE LINKS
    issuelinks = fields.get("issuelinks", [])
    
    # SUBTASKS
    subtasks = fields.get("subtasks", [])
    subtask_keys = [st.get("key", "") for st in subtasks]
    
    # CHANGELOG (if available)
    changelog = issue.get("changelog", {})
    histories = changelog.get("histories", [])
    changelog_summary = []
    for history in histories:
        created_date = history.get("created", "")
        author = history.get("author", {}).get("displayName", "Unknown")
        for item in history.get("items", []):
            field = item.get("field", "")
            from_val = item.get("fromString", "")
            to_val = item.get("toString", "")
            changelog_summary.append(f"{created_date} - {author} changed {field} from '{from_val}' to '{to_val}'")
    
    # ========================================
    # BASE METADATA (ALL FIELDS)
    # ========================================
    metadata_base = {
        "issue_key": key,
        "summary": summary,
        "issue_type": issuetype_name,
        "parent_epic": parent_key,
        
        # Status & Category
        "status": status_name,
        "status_category": status_category_name,
        
        # Priority & Ownership
        "priority": priority_name,
        "assignee": assignee_name,
        "reporter": reporter_name,
        
        # Labels
        "labels": labels,
        
        # Timeline
        "created": created,
        "updated": updated,
        "resolutiondate": resolutiondate,
        
        # Fix Versions
        "fix_versions": fix_version_names,
        "fix_versions_released": [v["name"] for v in fix_version_details if v["released"]],
        
        # Sprint
        "sprints": sprint_names,
        
        # Subtasks
        "subtask_count": len(subtask_keys),
        "has_subtasks": len(subtask_keys) > 0,
        
        # Comments & Attachments
        "comment_count": len(comments_text),
        "attachment_count": len(attachment_details),
        
        # Change History
        "changelog_count": len(changelog_summary)
    }
    
    # ========================================
    # DEPLOYMENT INFERENCE LOGIC
    # ========================================
    deployment_status = "Not Deployed"
    deployment_inference = ""
    
    if status_category_name in ["Done", "Complete"]:
        released_versions = [v for v in fix_version_details if v["released"]]
        if released_versions:
            deployment_status = "Deployed"
            version_list = ", ".join([v["name"] for v in released_versions])
            deployment_inference = f"This item was likely delivered as part of release {version_list}."
        else:
            deployment_inference = "Completed but not yet released to production."
    else:
        deployment_inference = f"Currently in {status_category_name} state."
    
    metadata_base["deployment_status"] = deployment_status
    
    # ========================================
    # CHUNK 1: IDENTITY & STRUCTURE
    # ========================================
    epic_text = f"under Epic {parent_key}" if parent_key else "with no parent Epic"
    
    identity_text = f"Issue {key} is a {issuetype_name} {epic_text}.\n"
    identity_text += f"Summary: {summary}\n"
    identity_text += f"Status: {status_name} ({status_category_name})\n"
    identity_text += f"Priority: {priority_name}\n"
    identity_text += f"Assignee: {assignee_name} | Reporter: {reporter_name}\n"
    
    if labels:
        identity_text += f"Labels: {', '.join(labels)}\n"
    
    if subtask_keys:
        identity_text += f"Subtasks ({len(subtask_keys)}): {', '.join(subtask_keys)}\n"
    
    chunks.append({
        "id": f"{key}::identity",
        "document": identity_text,
        "metadata": {**metadata_base, "chunk_type": "identity"}
    })
    
    # ========================================
    # CHUNK 2: BUSINESS CONTENT
    # ========================================
    # This chunk enables CONTENT-BASED HIERARCHY REASONING
    business_text = f"{key} ({issuetype_name})\n"
    business_text += f"Summary: {summary}\n\n"
    business_text += f"Description:\n{description}\n\n"
    
    # Add content signals for Epic/Story detection
    if description:
        content_signals = []
        
        # Epic-like signals
        if any(term in description.lower() for term in ["end-to-end", "workflow", "lifecycle", "all processes", "idef"]):
            content_signals.append("Contains end-to-end workflow language (Epic-like)")
        
        if any(term in description.lower() for term in ["tmp", "mdm", "epic", "credentialing"]) and \
           any(term in description.lower() for term in ["multiple", "all", "entire"]):
            content_signals.append("Mentions multiple systems (Epic-like)")
        
        # Story-like signals
        if any(term in description.lower() for term in ["create", "update", "activate", "deactivate", "notify"]) and \
           any(term in description.lower() for term in ["operator", "provider", "location", "message"]):
            content_signals.append("Describes single action on entity (Story-like)")
        
        if content_signals:
            business_text += f"Content Analysis: {'; '.join(content_signals)}\n"
    
    chunks.append({
        "id": f"{key}::business",
        "document": business_text,
        "metadata": {**metadata_base, "chunk_type": "business"}
    })
    
    # ========================================
    # CHUNK 3: TIMELINE & DELIVERY
    # ========================================
    timeline_text = f"{key} Timeline & Delivery Status:\n\n"
    timeline_text += f"Created: {created}\n"
    timeline_text += f"Last Updated: {updated}\n"
    
    if resolutiondate:
        timeline_text += f"Resolved: {resolutiondate}\n"
    
    timeline_text += f"\nCurrent Status: {status_name} ({status_category_name})\n"
    
    if fix_version_details:
        timeline_text += f"\nFix Versions:\n"
        for v in fix_version_details:
            release_status = "Released" if v["released"] else "Unreleased"
            release_date = v["releaseDate"] if v["releaseDate"] else "No date set"
            timeline_text += f"  - {v['name']}: {release_status} (Release Date: {release_date})\n"
    
    timeline_text += f"\nDeployment Inference: {deployment_inference}\n"
    
    if sprint_names:
        timeline_text += f"Sprints: {', '.join(sprint_names)}\n"
    
    chunks.append({
        "id": f"{key}::timeline",
        "document": timeline_text,
        "metadata": {**metadata_base, "chunk_type": "timeline"}
    })
    
    # ========================================
    # CHUNK 4: RELATIONSHIPS
    # ========================================
    relationships = []
    if issuelinks:
        for link in issuelinks:
            # Outward link (e.g. "blocks")
            if "outwardIssue" in link:
                rel_type = link["type"].get("outward", "relates to")
                target = link["outwardIssue"]["key"]
                target_summary = link["outwardIssue"]["fields"].get("summary", "")
                relationships.append(f"- {rel_type} {target}: {target_summary}")
            # Inward link (e.g. "is blocked by")
            elif "inwardIssue" in link:
                rel_type = link["type"].get("inward", "relates to")
                target = link["inwardIssue"]["key"]
                target_summary = link["inwardIssue"]["fields"].get("summary", "")
                relationships.append(f"- {rel_type} {target}: {target_summary}")
    
    if relationships:
        rel_text = f"{key} Relationships:\n" + "\n".join(relationships)
    else:
        rel_text = f"{key} has no documented issue links."
    
    chunks.append({
        "id": f"{key}::relationships",
        "document": rel_text,
        "metadata": {**metadata_base, "chunk_type": "relationships"}
    })
    
    # ========================================
    # CHUNK 5: COMMENTS & COLLABORATION
    # ========================================
    if comments_text:
        comments_chunk = f"{key} Comments ({len(comments_text)}):\n\n"
        comments_chunk += "\n\n".join(comments_text[:10])  # Limit to first 10 comments
        
        if len(comments_text) > 10:
            comments_chunk += f"\n\n... and {len(comments_text) - 10} more comments"
        
        chunks.append({
            "id": f"{key}::comments",
            "document": comments_chunk,
            "metadata": {**metadata_base, "chunk_type": "comments"}
        })
    
    # ========================================
    # CHUNK 6: ATTACHMENTS
    # ========================================
    if attachment_details:
        attachments_text = f"{key} Attachments ({len(attachment_details)}):\n\n"
        for att in attachment_details:
            attachments_text += f"- {att['filename']} ({att['mimeType']}) - Created: {att['created']}\n"
        
        chunks.append({
            "id": f"{key}::attachments",
            "document": attachments_text,
            "metadata": {**metadata_base, "chunk_type": "attachments"}
        })
    
    # ========================================
    # CHUNK 7: CHANGE HISTORY
    # ========================================
    if changelog_summary:
        changelog_text = f"{key} Change History ({len(changelog_summary)} changes):\n\n"
        changelog_text += "\n".join(changelog_summary[:20])  # Limit to first 20 changes
        
        if len(changelog_summary) > 20:
            changelog_text += f"\n\n... and {len(changelog_summary) - 20} more changes"
        
        chunks.append({
            "id": f"{key}::changelog",
            "document": changelog_text,
            "metadata": {**metadata_base, "chunk_type": "changelog"}
        })
    
    # ========================================
    # CHUNK 8: EPIC SUMMARY (Epics only)
    # ========================================
    if issuetype_name == "Epic":
        epic_text = f"Epic {key}: {summary}\n\n"
        epic_text += f"This Epic represents a high-level initiative or workflow.\n"
        epic_text += f"Description scope: {description[:200]}...\n" if len(description) > 200 else f"Description: {description}\n"
        
        if subtask_keys:
            epic_text += f"\nContains {len(subtask_keys)} subtasks/stories.\n"
        
        chunks.append({
            "id": f"{key}::epic_summary",
            "document": epic_text,
            "metadata": {**metadata_base, "chunk_type": "epic_summary"}
        })
    
    return chunks

# -------------------------
# SANITIZE METADATA
# -------------------------
def normalize_metadata(meta):
    """
    Flatten metadata for consistent storage.
    Lists -> " | " joined strings.
    """
    out = {}
    for k, v in meta.items():
        if isinstance(v, list):
            out[k] = " | ".join(str(x) for x in v)
        else:
            out[k] = v
    return out

# -------------------------
# STORE IN CHROMA CLOUD
# -------------------------
def store_chroma(chunks):
    """
    Store chunks with normalized metadata.
    """
    collection.upsert(
        ids=[c["id"] for c in chunks],
        documents=[c["document"] for c in chunks],
        metadatas=[normalize_metadata(c["metadata"]) for c in chunks]
    )

# -------------------------
# MAIN
# -------------------------
def main():
    issues = load_jira("./data/jira")

    for issue in tqdm(issues):
        chunks = generate_chunks(issue)
        store_chroma(chunks)

    print("✅ Chroma Cloud ingestion complete for 'jira_issues_rag'.")

if __name__ == "__main__":
    main()