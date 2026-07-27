"""Small JSON-backed store for the public User Site.

The main project does not currently include a database or migration framework.
This store gives the User Site real backend-owned Projects, Conversations,
Messages, ContextItems, and DetectionResults without changing the admin stack.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STORE_PATH = PROJECT_ROOT / "data" / "user_site_store.json"


class StoreError(ValueError):
    """Raised when a requested user-site entity is invalid or unavailable."""


class NotFoundError(StoreError):
    """Raised when the current user cannot access the requested entity."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


def public_detection_summary(detection: Any) -> dict[str, Any]:
    """Return the small, non-diagnostic status contract exposed to User Site."""
    value = detection if isinstance(detection, dict) else {}
    security = value.get("security", {}) if isinstance(value.get("security"), dict) else {}
    model_scores = value.get("modelScores", {}) if isinstance(value.get("modelScores"), dict) else {}
    document = model_scores.get("document", {}) if isinstance(model_scores.get("document"), dict) else {}
    source = document.get("source", {}) if isinstance(document.get("source"), dict) else {}

    existing_document_status = value.get("documentStatus")
    document_status = deepcopy(existing_document_status) if isinstance(existing_document_status, dict) else None
    if document:
        unsafe_chunks = int(document.get("unsafeChunkCount", 0) or 0)
        safe_chunks = int(document.get("safeChunkCount", 0) or 0)
        document_status = {
            "fileName": source.get("fileName"),
            "scanStatus": "content_removed" if unsafe_chunks else "checked",
            "contentRemoved": unsafe_chunks > 0,
            "removedChunkCount": unsafe_chunks,
            "acceptedChunkCount": safe_chunks,
        }

    return {
        "requestId": value.get("requestId"),
        "decision": value.get("decision"),
        "label": value.get("label"),
        "warning": security.get("warning", value.get("warning")),
        "inputDecision": security.get("inputDecision", value.get("inputDecision")),
        "outputDecision": security.get("outputDecision", value.get("outputDecision")),
        "documentStatus": document_status,
    }


def _empty_store() -> dict[str, list[dict[str, Any]]]:
    return {
        "projects": [],
        "contextItems": [],
        "conversations": [],
        "messages": [],
        "detectionResults": [],
    }


class UserSiteStore:
    """Persist User Site data in one small JSON document."""

    def __init__(self, path: Path = DEFAULT_STORE_PATH) -> None:
        self.path = path
        self._lock = Lock()

    def _load_unlocked(self) -> dict[str, list[dict[str, Any]]]:
        if not self.path.exists():
            return _empty_store()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise StoreError(f"User Site store is not valid JSON: {self.path}") from exc

        store = _empty_store()
        for key in store:
            value = data.get(key, [])
            store[key] = value if isinstance(value, list) else []
        return store

    def _save_unlocked(self, data: dict[str, list[dict[str, Any]]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def _project_unlocked(self, data: dict[str, list[dict[str, Any]]], owner_id: str, project_id: str) -> dict[str, Any]:
        for project in data["projects"]:
            if project["id"] == project_id and project["ownerId"] == owner_id:
                return project
        raise NotFoundError("Project not found.")

    def _conversation_unlocked(
        self,
        data: dict[str, list[dict[str, Any]]],
        owner_id: str,
        conversation_id: str,
    ) -> dict[str, Any]:
        for conversation in data["conversations"]:
            if conversation["id"] == conversation_id and conversation["ownerId"] == owner_id:
                return conversation
        raise NotFoundError("Conversation not found.")

    def list_projects(self, owner_id: str) -> list[dict[str, Any]]:
        with self._lock:
            data = self._load_unlocked()
            projects = [deepcopy(item) for item in data["projects"] if item["ownerId"] == owner_id]
            projects.sort(key=lambda item: item["updatedAt"], reverse=True)
            return projects

    def create_project(
        self,
        owner_id: str,
        *,
        name: str,
        description: str = "",
        system_instruction: str = "",
        context_summary: str = "",
        context_text: str = "",
    ) -> dict[str, Any]:
        clean_name = name.strip()
        if not clean_name:
            raise StoreError("Project name is required.")

        now = utc_now()
        project = {
            "id": new_id("proj"),
            "name": clean_name,
            "description": description.strip(),
            "ownerId": owner_id,
            "createdAt": now,
            "updatedAt": now,
            "systemInstruction": system_instruction.strip(),
            "contextSummary": context_summary.strip(),
        }
        with self._lock:
            data = self._load_unlocked()
            data["projects"].append(project)
            if context_text.strip():
                data["contextItems"].append(
                    {
                        "id": new_id("ctx"),
                        "projectId": project["id"],
                        "ownerId": owner_id,
                        "title": "Initial context",
                        "content": context_text.strip(),
                        "type": "text",
                        "createdAt": now,
                        "updatedAt": now,
                    }
                )
            self._save_unlocked(data)
        return deepcopy(project)

    def get_project(self, owner_id: str, project_id: str) -> dict[str, Any]:
        with self._lock:
            data = self._load_unlocked()
            return deepcopy(self._project_unlocked(data, owner_id, project_id))

    def update_project(self, owner_id: str, project_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        allowed = {"name", "description", "systemInstruction", "contextSummary"}
        with self._lock:
            data = self._load_unlocked()
            project = self._project_unlocked(data, owner_id, project_id)
            for key in allowed:
                if key in updates and updates[key] is not None:
                    value = str(updates[key]).strip()
                    if key == "name" and not value:
                        raise StoreError("Project name is required.")
                    project[key] = value
            project["updatedAt"] = utc_now()
            self._save_unlocked(data)
            return deepcopy(project)

    def delete_project(self, owner_id: str, project_id: str) -> dict[str, Any]:
        with self._lock:
            data = self._load_unlocked()
            project = self._project_unlocked(data, owner_id, project_id)
            data["projects"] = [
                item for item in data["projects"] if not (item["id"] == project_id and item["ownerId"] == owner_id)
            ]
            data["contextItems"] = [
                item
                for item in data["contextItems"]
                if not (item["projectId"] == project_id and item["ownerId"] == owner_id)
            ]
            for conversation in data["conversations"]:
                if conversation["ownerId"] == owner_id and conversation.get("projectId") == project_id:
                    conversation["projectId"] = None
                    conversation["updatedAt"] = utc_now()
            self._save_unlocked(data)
            return deepcopy(project)

    def list_context_items(self, owner_id: str, project_id: str) -> list[dict[str, Any]]:
        with self._lock:
            data = self._load_unlocked()
            self._project_unlocked(data, owner_id, project_id)
            items = [
                deepcopy(item)
                for item in data["contextItems"]
                if item["ownerId"] == owner_id and item["projectId"] == project_id
            ]
            items.sort(key=lambda item: item["createdAt"])
            return items

    def create_context_item(
        self,
        owner_id: str,
        project_id: str,
        *,
        title: str,
        content: str,
        item_type: str = "text",
    ) -> dict[str, Any]:
        if not content.strip():
            raise StoreError("Context content is required.")
        now = utc_now()
        item = {
            "id": new_id("ctx"),
            "projectId": project_id,
            "ownerId": owner_id,
            "title": title.strip() or "Untitled context",
            "content": content.strip(),
            "type": item_type.strip() or "text",
            "createdAt": now,
            "updatedAt": now,
        }
        with self._lock:
            data = self._load_unlocked()
            project = self._project_unlocked(data, owner_id, project_id)
            data["contextItems"].append(item)
            project["updatedAt"] = now
            self._save_unlocked(data)
            return deepcopy(item)

    def update_context_item(
        self,
        owner_id: str,
        project_id: str,
        context_id: str,
        updates: dict[str, Any],
    ) -> dict[str, Any]:
        with self._lock:
            data = self._load_unlocked()
            project = self._project_unlocked(data, owner_id, project_id)
            for item in data["contextItems"]:
                if item["id"] == context_id and item["projectId"] == project_id and item["ownerId"] == owner_id:
                    if updates.get("title") is not None:
                        item["title"] = str(updates["title"]).strip() or item["title"]
                    if updates.get("content") is not None:
                        content = str(updates["content"]).strip()
                        if not content:
                            raise StoreError("Context content is required.")
                        item["content"] = content
                    if updates.get("type") is not None:
                        item["type"] = str(updates["type"]).strip() or "text"
                    item["updatedAt"] = utc_now()
                    project["updatedAt"] = item["updatedAt"]
                    self._save_unlocked(data)
                    return deepcopy(item)
            raise NotFoundError("Context item not found.")

    def delete_context_item(self, owner_id: str, project_id: str, context_id: str) -> dict[str, Any]:
        with self._lock:
            data = self._load_unlocked()
            project = self._project_unlocked(data, owner_id, project_id)
            deleted: dict[str, Any] | None = None
            remaining: list[dict[str, Any]] = []
            for item in data["contextItems"]:
                if item["id"] == context_id and item["projectId"] == project_id and item["ownerId"] == owner_id:
                    deleted = item
                else:
                    remaining.append(item)
            if deleted is None:
                raise NotFoundError("Context item not found.")
            data["contextItems"] = remaining
            project["updatedAt"] = utc_now()
            self._save_unlocked(data)
            return deepcopy(deleted)

    def build_project_context(self, owner_id: str, project_id: str | None) -> tuple[dict[str, Any] | None, str]:
        if not project_id:
            return None, ""
        with self._lock:
            data = self._load_unlocked()
            project = deepcopy(self._project_unlocked(data, owner_id, project_id))
            items = [
                deepcopy(item)
                for item in data["contextItems"]
                if item["ownerId"] == owner_id and item["projectId"] == project_id
            ]

        parts = [
            f"Project name: {project['name']}",
            f"Project description: {project.get('description', '')}",
            f"Project system instruction: {project.get('systemInstruction', '')}",
            f"Project context summary: {project.get('contextSummary', '')}",
        ]
        for item in items:
            parts.append(f"Context item ({item.get('title', 'Untitled')}): {item.get('content', '')}")

        context = {
            "projectId": project["id"],
            "projectName": project["name"],
            "projectDescription": project.get("description", ""),
            "systemInstruction": project.get("systemInstruction", ""),
            "contextSummary": project.get("contextSummary", ""),
            "documents": [
                {
                    "id": item["id"],
                    "title": item.get("title", ""),
                    "type": item.get("type", "text"),
                    "content": item.get("content", ""),
                }
                for item in items
            ],
        }
        return context, "\n\n".join(part for part in parts if part.strip())

    def list_conversations(
        self,
        owner_id: str,
        *,
        project_id: str | None = None,
        search: str | None = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            data = self._load_unlocked()
            if project_id:
                self._project_unlocked(data, owner_id, project_id)
            query = (search or "").strip().lower()
            conversations = []
            for conversation in data["conversations"]:
                if conversation["ownerId"] != owner_id:
                    continue
                if project_id and conversation.get("projectId") != project_id:
                    continue
                if query and query not in conversation.get("title", "").lower():
                    continue
                conversations.append(deepcopy(conversation))
            conversations.sort(key=lambda item: item["updatedAt"], reverse=True)
            return conversations

    def create_conversation(
        self,
        owner_id: str,
        *,
        title: str | None = None,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        now = utc_now()
        with self._lock:
            data = self._load_unlocked()
            if project_id:
                self._project_unlocked(data, owner_id, project_id)
            conversation = {
                "id": new_id("conv"),
                "ownerId": owner_id,
                "projectId": project_id,
                "title": (title or "New chat").strip() or "New chat",
                "createdAt": now,
                "updatedAt": now,
            }
            data["conversations"].append(conversation)
            self._save_unlocked(data)
            return deepcopy(conversation)

    def get_conversation(self, owner_id: str, conversation_id: str) -> dict[str, Any]:
        with self._lock:
            data = self._load_unlocked()
            conversation = deepcopy(self._conversation_unlocked(data, owner_id, conversation_id))
            messages = [
                deepcopy(item)
                for item in data["messages"]
                if item["ownerId"] == owner_id and item["conversationId"] == conversation_id
            ]
            messages.sort(key=lambda item: item["createdAt"])
            results_by_id = {
                item["id"]: {
                    **deepcopy(item),
                    "summary": public_detection_summary(item.get("summary")),
                }
                for item in data["detectionResults"]
                if item["ownerId"] == owner_id and item["conversationId"] == conversation_id
            }
        conversation["messages"] = [
            {
                **message,
                "detectionResult": results_by_id.get(message.get("detectionResultId"))
                if message.get("detectionResultId")
                else None,
            }
            for message in messages
        ]
        return conversation

    def update_conversation(
        self,
        owner_id: str,
        conversation_id: str,
        updates: dict[str, Any],
    ) -> dict[str, Any]:
        with self._lock:
            data = self._load_unlocked()
            conversation = self._conversation_unlocked(data, owner_id, conversation_id)
            if "title" in updates and updates["title"] is not None:
                title = str(updates["title"]).strip()
                if not title:
                    raise StoreError("Conversation title is required.")
                conversation["title"] = title
            if "projectId" in updates:
                project_id = updates["projectId"]
                if project_id:
                    self._project_unlocked(data, owner_id, str(project_id))
                conversation["projectId"] = project_id
            conversation["updatedAt"] = utc_now()
            self._save_unlocked(data)
            return deepcopy(conversation)

    def delete_conversation(self, owner_id: str, conversation_id: str) -> dict[str, Any]:
        with self._lock:
            data = self._load_unlocked()
            conversation = self._conversation_unlocked(data, owner_id, conversation_id)
            data["conversations"] = [
                item
                for item in data["conversations"]
                if not (item["id"] == conversation_id and item["ownerId"] == owner_id)
            ]
            data["messages"] = [
                item
                for item in data["messages"]
                if not (item["conversationId"] == conversation_id and item["ownerId"] == owner_id)
            ]
            data["detectionResults"] = [
                item
                for item in data["detectionResults"]
                if not (item["conversationId"] == conversation_id and item["ownerId"] == owner_id)
            ]
            self._save_unlocked(data)
            return deepcopy(conversation)

    def append_chat_exchange(
        self,
        owner_id: str,
        *,
        conversation_id: str,
        project_id: str | None,
        user_message: str,
        assistant_message: str,
        detection: dict[str, Any],
    ) -> dict[str, Any]:
        now = utc_now()
        with self._lock:
            data = self._load_unlocked()
            conversation = self._conversation_unlocked(data, owner_id, conversation_id)
            if conversation.get("projectId") != project_id:
                conversation["projectId"] = project_id
            if conversation.get("title") == "New chat":
                conversation["title"] = user_message.strip()[:48] or "New chat"
            conversation["updatedAt"] = now

            user_message_record = {
                "id": new_id("msg"),
                "conversationId": conversation_id,
                "ownerId": owner_id,
                "role": "user",
                "content": user_message,
                "createdAt": now,
            }
            detection_record = {
                "id": new_id("det"),
                "conversationId": conversation_id,
                "ownerId": owner_id,
                "projectId": project_id,
                "messageId": None,
                "createdAt": now,
                "summary": public_detection_summary(detection),
            }
            assistant_message_record = {
                "id": new_id("msg"),
                "conversationId": conversation_id,
                "ownerId": owner_id,
                "role": "assistant",
                "content": assistant_message,
                "createdAt": now,
                "detectionResultId": detection_record["id"],
            }
            detection_record["messageId"] = assistant_message_record["id"]

            data["messages"].extend([user_message_record, assistant_message_record])
            data["detectionResults"].append(detection_record)
            self._save_unlocked(data)

            return {
                "conversation": deepcopy(conversation),
                "userMessage": deepcopy(user_message_record),
                "assistantMessage": {
                    **deepcopy(assistant_message_record),
                    "detectionResult": deepcopy(detection_record),
                },
                "detectionResult": deepcopy(detection_record),
            }


store = UserSiteStore()
