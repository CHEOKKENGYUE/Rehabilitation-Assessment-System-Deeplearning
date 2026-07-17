"""FastAPI entrypoint for the rehabilitation assessment platform.

Three endpoints:
  POST /api/assess                       — accept multipart files + patient info, return session_id
  GET  /api/assess/{session_id}/stream   — SSE stream of progress events
  GET  /api/assess/{session_id}/result   — cached final result (reconnect fallback)

The full inference pipeline runs in a worker thread; events are pushed onto a
per-session queue.Queue that the SSE coroutine drains asynchronously.
"""
from __future__ import annotations

import asyncio
import json
import queue
import shutil
import tempfile
import threading
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

import db
from inference import SENTINEL, ModelRegistry, error_event, run_pipeline
from schemas import (
    AssessmentOverview,
    AssessmentResult,
    AssessSessionResponse,
    PatientDetail,
    PatientInfo,
    PatientSummary,
    PatientUpdate,
    PredictionResult,
    StatsSummary,
)

load_dotenv(Path(__file__).resolve().parent / ".env")

SESSION_ROOT = Path(tempfile.gettempdir()) / "rehab_sessions"
SESSION_ROOT.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# In-process session registry. Keys: session_id → SessionState.               #
# --------------------------------------------------------------------------- #
class SessionState:
    def __init__(self, session_id: str, patient: PatientInfo, eeg_paths: List[Path],
                 emg_paths: List[Path], institution: str = "hospital"):
        self.session_id = session_id
        self.patient = patient
        self.eeg_paths = eeg_paths
        self.emg_paths = emg_paths
        self.institution = institution
        self.queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self.result: Optional[AssessmentResult] = None
        self.started: bool = False
        self.lock = threading.Lock()


SESSIONS: Dict[str, SessionState] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    print(f"[startup] SQLite ready at {db.DB_PATH}")

    registry = ModelRegistry()
    print(f"[startup] loading CMK-AGN models onto {registry.device}...")
    registry.load_all()
    print(f"[startup] loaded {len(registry.models)} models: {list(registry.models.keys())}")
    app.state.registry = registry

    yield
    # No teardown needed; torch frees memory on process exit.


app = FastAPI(title="Rehabilitation Assessment Platform", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _save_uploads(files: List[UploadFile], destdir: Path, prefix: str) -> List[Path]:
    destdir.mkdir(parents=True, exist_ok=True)
    out: List[Path] = []
    for i, uf in enumerate(files):
        suffix = Path(uf.filename or f"{prefix}_{i}.csv").suffix or ".csv"
        target = destdir / f"{prefix}_{i:02d}{suffix}"
        with target.open("wb") as fh:
            shutil.copyfileobj(uf.file, fh)
        out.append(target)
    return out


def _worker(state: SessionState, registry: ModelRegistry) -> None:
    """Run the FMA inference pipeline on a worker thread."""
    try:
        predictions_raw = run_pipeline(
            state.eeg_paths, state.emg_paths, registry, state.queue,
            institution=state.institution,
        )

        predictions = PredictionResult(
            FMA_UE=float(predictions_raw["FMA_UE"]),
        )

        state.result = AssessmentResult(
            session_id=state.session_id,
            patient_info=state.patient,
            predictions=predictions,
        )

        # Persist to SQLite. DB errors are isolated so they never break the SSE
        # `done` event.
        try:
            pid = db.upsert_patient(state.patient)
            db.insert_assessment(pid, state.session_id, predictions)
        except Exception as exc:  # noqa: BLE001
            print(f"[persist][warn] failed to save assessment {state.session_id}: {exc}")

        state.queue.put({"type": "done"})
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        state.queue.put(error_event(f"会话 {state.session_id} 失败：{exc}"))
    finally:
        state.queue.put(SENTINEL)


# --------------------------------------------------------------------------- #
# Endpoints                                                                   #
# --------------------------------------------------------------------------- #
@app.post("/api/assess", response_model=AssessSessionResponse)
async def create_assessment(
    patient_id: str = Form(...),
    name: str = Form(...),
    sex: str = Form(...),
    age: Optional[int] = Form(None),
    diagnosis: str = Form(...),
    disease_days: Optional[int] = Form(None),
    paralysis_side: str = Form(...),
    eeg_files: List[UploadFile] = File(...),
    emg_files: List[UploadFile] = File(...),
):
    if len(eeg_files) == 0 or len(emg_files) == 0:
        raise HTTPException(status_code=422, detail="必须至少上传一对 EEG / EMG 文件")
    if len(eeg_files) != len(emg_files):
        raise HTTPException(
            status_code=422,
            detail=f"EEG 与 EMG 文件数量不匹配：{len(eeg_files)} vs {len(emg_files)}",
        )

    try:
        patient = PatientInfo(
            patient_id=patient_id,
            name=name,
            sex=sex,  # type: ignore[arg-type]
            age=age,
            diagnosis=diagnosis,
            disease_days=disease_days,
            paralysis_side=paralysis_side,  # type: ignore[arg-type]
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"患者信息无效：{exc}") from exc

    session_id = uuid.uuid4().hex[:12]
    destdir = SESSION_ROOT / session_id
    eeg_paths = _save_uploads(eeg_files, destdir / "eeg", "eeg")
    emg_paths = _save_uploads(emg_files, destdir / "emg", "emg")

    SESSIONS[session_id] = SessionState(session_id, patient, eeg_paths, emg_paths)
    return AssessSessionResponse(session_id=session_id, n_trials=len(eeg_paths))


@app.get("/api/assess/{session_id}/stream")
async def stream_assessment(session_id: str):
    state = SESSIONS.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="session 不存在或已过期")

    registry: ModelRegistry = app.state.registry

    # Kick off worker only once per session.
    with state.lock:
        if not state.started:
            state.started = True
            threading.Thread(
                target=_worker, args=(state, registry), daemon=True
            ).start()

    async def event_generator():
        loop = asyncio.get_event_loop()
        while True:
            try:
                item = await loop.run_in_executor(None, state.queue.get)
            except Exception as exc:  # noqa: BLE001
                yield f"data: {json.dumps({'type': 'error', 'message': str(exc)}, ensure_ascii=False)}\n\n"
                break
            if item is SENTINEL or item.get("__sentinel__"):
                break
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)


@app.get("/api/assess/{session_id}/result", response_model=AssessmentResult)
async def get_result(session_id: str):
    state = SESSIONS.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="session 不存在")
    if state.result is None:
        raise HTTPException(status_code=425, detail="评估尚未完成")
    return state.result


@app.get("/api/health")
async def health():
    return {"status": "ok", "models_loaded": list(app.state.registry.models.keys())}


# --------------------------------------------------------------------------- #
# Patient management / records / stats (SQLite-backed)                        #
# --------------------------------------------------------------------------- #
@app.get("/api/patients", response_model=List[PatientSummary])
async def list_patients():
    return db.list_patients()


@app.get("/api/patients/{patient_db_id}", response_model=PatientDetail)
async def get_patient(patient_db_id: int):
    patient = db.get_patient(patient_db_id)
    if patient is None:
        raise HTTPException(status_code=404, detail="患者不存在")
    return patient


@app.patch("/api/patients/{patient_db_id}", response_model=PatientDetail)
async def update_patient(patient_db_id: int, payload: PatientUpdate):
    fields = payload.model_dump(exclude_unset=True)
    patient = db.update_patient(patient_db_id, fields)
    if patient is None:
        raise HTTPException(status_code=404, detail="患者不存在")
    return patient


@app.get("/api/assessments", response_model=AssessmentOverview)
async def list_assessments(limit: int = 50, offset: int = 0):
    limit = max(1, min(limit, 500))
    offset = max(0, offset)
    return db.list_all_assessments(limit=limit, offset=offset)


@app.get("/api/stats/summary", response_model=StatsSummary)
async def stats_summary():
    return db.stats_summary()


# --------------------------------------------------------------------------- #
# CLI entry: `python -m backend.main` or `uvicorn backend.main:app --reload`.  #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
