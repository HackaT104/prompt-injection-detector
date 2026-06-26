"""FastAPI demo for the Prompt Injection Detector."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.advanced_detection import (
    compare_all_models,
    detect_prompt_advanced,
    get_project_statistics,
    simulate_chat_detection,
)
from src.batch_evaluation import evaluate_batch_items, validate_batch_items
from src.detection.pipeline import run_hybrid_detection
from src.detector import detect_prompt, detect_prompt_with_context, model_files_status
from src.indirect_pipeline import detect_indirect_content
from src.model_diagnostics import diagnose_model
from src.transformer_utils import (
    TRANSFORMER_MODELS_DIR,
    diagnose_transformer,
    is_finetuned_transformer_checkpoint,
    resolve_transformer_model_dir,
    safe_model_dir_name,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ADVANCED_DEMO_PAGE = PROJECT_ROOT / "static" / "advanced_demo.html"
CHAT_SIMULATION_PAGE = PROJECT_ROOT / "static" / "chat_simulation.html"
BATCH_EVALUATION_PAGE = PROJECT_ROOT / "static" / "batch_evaluation.html"

app = FastAPI(
    title="Prompt Injection Detector",
    description="Detector layer phát hiện prompt injection bằng rule-based và ML.",
    version="1.0.0",
)


class DetectRequest(BaseModel):
    text: str
    model_type: str = "hybrid"


class DetectContextRequest(BaseModel):
    user_prompt: str
    context: str = ""
    model_type: str = "hybrid"


class ContextAwareDetectRequest(BaseModel):
    user_task: str
    external_content: str
    source_type: str = "raw_text"
    source_name: str = "inline-content"
    model_name: str = "roberta"
    ml_model_type: str = "logistic_regression"
    safe_context_policy: str = "exclude"
    ensemble_config: dict[str, Any] | None = None
    use_transformer: bool = True
    use_cuda: bool = True


class AdvancedDetectRequest(BaseModel):
    input_type: str = "text"
    text: str = ""
    model: str = "hybrid"
    model_type: str | None = None
    model_name: str | None = None
    hybrid_config: dict[str, Any] | None = None
    file_name: str | None = None
    mime_type: str | None = None


class ModelCompareRequest(BaseModel):
    input_type: str = "text"
    text: str
    hybrid_config: dict[str, Any] | None = None


class ChatSimulationRequest(BaseModel):
    text: str
    model: str = "hybrid"
    model_type: str | None = None
    model_name: str | None = None
    hybrid_config: dict[str, Any] | None = None


class BatchDatasetRequest(BaseModel):
    items: list[dict[str, Any]]
    models: list[str] | None = None
    hybrid_config: dict[str, Any] | None = None
    dataset_name: str | None = None
    max_items: int = 1000


class ModelDiagnosticsRequest(BaseModel):
    text: str
    model: str


def _compact_rule_result(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if result is None:
        return None
    risk_score = result.get("risk_score", 0)
    return {
        "triggered": bool(result.get("label") == 1 or float(risk_score or 0) > 0),
        "score": risk_score,
        "action": result.get("action"),
        "matched_rules": result.get("matched_rules", []),
    }


def _compact_ml_result(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if not result:
        return None
    compact: dict[str, Any] = {
        "predicted_label": result.get("label"),
        "score": result.get("risk_score"),
        "model_action": result.get("action"),
        "model": result.get("method"),
    }
    for field in ["available", "decision_mode", "thresholds", "missing_files", "message"]:
        if field in result:
            compact[field] = result[field]
    return compact


def _compact_detect_response(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "input": {
            "original_text": result.get("original_text", result.get("input")),
            "detected_language": result.get("detected_language"),
            "canonical_text": result.get("canonical_text"),
        },
        "decision": {
            "label": result.get("label"),
            "risk_score": result.get("risk_score"),
            "action": result.get("action"),
            "method": result.get("method"),
        },
        "signals": {
            "rule_based": _compact_rule_result(result.get("rule_based_result")),
            "ml": _compact_ml_result(result.get("ml_result")),
        },
        "explanation": result.get("explanation"),
    }


def _compact_context_response(result: dict[str, Any]) -> dict[str, Any]:
    rule_result = result.get("rule_result", {})
    ml_result = result.get("ml_result", {})
    language_result = result.get("language_result", {})
    return {
        "input": {
            "user_prompt": {
                "detected_language": language_result.get("user_prompt"),
                "canonical_text": result.get("canonical_user_prompt"),
            },
            "context": {
                "detected_language": language_result.get("context"),
                "canonical_text": result.get("canonical_context"),
            },
        },
        "decision": {
            "label": 0 if result.get("action") == "allow" else 1,
            "risk_score": result.get("risk_score"),
            "action": result.get("action"),
            "attack_type": result.get("attack_type"),
        },
        "signals": {
            "matched_rules": result.get("matched_rules", []),
            "rule_based": {
                "direct": _compact_rule_result(rule_result.get("direct")),
                "indirect": _compact_rule_result(rule_result.get("indirect")),
            },
            "ml": {
                "direct": _compact_ml_result(ml_result.get("direct")),
                "indirect": _compact_ml_result(ml_result.get("indirect")),
            },
        },
        "explanation": result.get("explanation"),
    }


@app.get("/")
def root() -> dict[str, object]:
    return {
        "project": "Prompt Injection Detector",
        "description": "Hệ thống phát hiện Prompt Injection đa ngôn ngữ bằng canonical English normalization, Rule-based, Logistic Regression, Linear SVM và Random Forest.",
        "endpoints": [
            "/detect",
            "/detect-context",
            "/detect-context-aware",
            "/detect-context-aware/upload",
            "/detect/advanced",
            "/detect/compare",
            "/advanced-demo",
            "/chat-simulation",
            "/batch-evaluation",
            "/batch/validate",
            "/batch/evaluate",
            "/project/stats",
            "/diagnostics/model",
            "/diagnostics/transformer",
            "/health",
        ],
    }


@app.get("/advanced-demo")
def advanced_demo_page() -> FileResponse:
    if not ADVANCED_DEMO_PAGE.exists():
        raise HTTPException(status_code=404, detail="Advanced demo page not found.")
    return FileResponse(ADVANCED_DEMO_PAGE, media_type="text/html")


@app.get("/chat-simulation")
def chat_simulation_page() -> FileResponse:
    if not CHAT_SIMULATION_PAGE.exists():
        raise HTTPException(status_code=404, detail="Chat simulation page not found.")
    return FileResponse(CHAT_SIMULATION_PAGE, media_type="text/html")


@app.get("/batch-evaluation")
def batch_evaluation_page() -> FileResponse:
    if not BATCH_EVALUATION_PAGE.exists():
        raise HTTPException(status_code=404, detail="Batch evaluation page not found.")
    return FileResponse(BATCH_EVALUATION_PAGE, media_type="text/html")


@app.get("/health")
def health() -> dict[str, object]:
    status = model_files_status()
    direct_status = {
        key: value
        for key, value in status.items()
        if not key.startswith("indirect_")
    }
    indirect_status = {
        key: value
        for key, value in status.items()
        if key.startswith("indirect_")
    }
    direct_required_files_found = all(
        item["model_found"] and item["vectorizer_found"]
        for item in direct_status.values()
    )
    indirect_files_found = any(
        item["model_found"] and item["vectorizer_found"]
        for item in indirect_status.values()
    )
    transformer_status = {
        model_name: {
            "resolved_path": str(resolve_transformer_model_dir(model_name)),
            "fine_tuned_ready": is_finetuned_transformer_checkpoint(resolve_transformer_model_dir(model_name)),
        }
        for model_name in ["roberta", "xlm_roberta"]
    }
    transformer_status["distilbert_deprecated"] = {
        "default_runtime": False,
        "backup_found": (PROJECT_ROOT / "models" / "deprecated").exists(),
    }
    return {
        "api": "running",
        "model_files_found": direct_required_files_found,
        "direct_model_files_found": direct_required_files_found,
        "indirect_model_files_found": indirect_files_found,
        "transformer_models": transformer_status,
        "models": status,
    }


@app.post("/detect")
def detect(request: DetectRequest) -> dict[str, object]:
    if request.text is None or not request.text.strip():
        raise HTTPException(status_code=400, detail="Field 'text' không được rỗng.")

    result = detect_prompt(request.text, model_type=request.model_type)
    if result.get("error"):
        message = str(result.get("message", "Lỗi không xác định."))
        status_code = 503 if "chưa được train" in message else 400
        raise HTTPException(status_code=status_code, detail=result)
    return _compact_detect_response(result)


@app.post("/detect-context")
def detect_context(request: DetectContextRequest) -> dict[str, object]:
    if request.user_prompt is None or not request.user_prompt.strip():
        raise HTTPException(status_code=400, detail="Field 'user_prompt' không được rỗng.")

    result = detect_prompt_with_context(
        request.user_prompt,
        context=request.context,
        model_type=request.model_type,
    )
    if result.get("error"):
        raise HTTPException(status_code=400, detail=result)
    return _compact_context_response(result)


@app.post("/detect-context-aware")
def detect_context_aware(request: ContextAwareDetectRequest) -> dict[str, object]:
    """Analyze external text with explainable hybrid context-aware detection."""
    try:
        return run_hybrid_detection(
            user_prompt=request.user_task,
            user_task=request.user_task,
            external_content=request.external_content,
            ml_model_type=request.ml_model_type,
            transformer_model=request.model_name,
            use_ml=True,
            use_transformer=request.use_transformer,
            use_cuda=request.use_cuda,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/detect-context-aware/upload")
async def detect_context_aware_upload(
    request: Request,
    user_task: str = Query(..., min_length=1),
    source_type: str = Query("auto"),
    source_name: str = Query("uploaded-content"),
    model_name: str = Query("roberta"),
    safe_context_policy: str = Query("exclude"),
    use_cuda: bool = Query(True),
) -> dict[str, object]:
    """Analyze raw request-body bytes without requiring multipart form dependencies."""
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Uploaded content is empty.")
    if len(body) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Uploaded content exceeds the 20 MB limit.")
    try:
        return detect_indirect_content(
            user_task=user_task,
            content_bytes=body,
            source_type=source_type,
            source_name=source_name,
            model_name=model_name,
            safe_context_policy=safe_context_policy,
            use_cuda=use_cuda,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/detect/advanced")
def detect_advanced(request: AdvancedDetectRequest) -> dict[str, object]:
    selected_model = request.model_type or request.model_name or request.model
    try:
        return detect_prompt_advanced(
            text=request.text,
            input_type=request.input_type,
            model=selected_model,
            hybrid_config=request.hybrid_config,
            file_name=request.file_name,
            mime_type=request.mime_type,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/detect/compare")
def detect_compare(request: ModelCompareRequest) -> dict[str, object]:
    try:
        return compare_all_models(
            text=request.text,
            input_type=request.input_type,
            hybrid_config=request.hybrid_config,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/project/stats")
def project_stats() -> dict[str, object]:
    return get_project_statistics()


@app.post("/batch/validate")
def batch_validate(request: BatchDatasetRequest) -> dict[str, object]:
    try:
        validation = validate_batch_items(request.items, max_items=request.max_items, dataset_name=request.dataset_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {key: value for key, value in validation.items() if key != "items"}


@app.post("/batch/evaluate")
def batch_evaluate(request: BatchDatasetRequest) -> dict[str, object]:
    try:
        return evaluate_batch_items(
            items=request.items,
            models=request.models,
            hybrid_config=request.hybrid_config,
            dataset_name=request.dataset_name,
            max_items=request.max_items,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/llm/mock")
def llm_mock(request: ChatSimulationRequest) -> dict[str, object]:
    selected_model = request.model_type or request.model_name or request.model
    try:
        return simulate_chat_detection(
            user_prompt=request.text,
            model=selected_model,
            hybrid_config=request.hybrid_config,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/diagnostics/transformer")
def diagnostics_transformer(request: ModelDiagnosticsRequest) -> dict[str, object]:
    try:
        return diagnose_transformer(text=request.text, model_name=request.model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/diagnostics/model")
def diagnostics_model(request: ModelDiagnosticsRequest) -> dict[str, object]:
    try:
        return diagnose_model(text=request.text, model=request.model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
