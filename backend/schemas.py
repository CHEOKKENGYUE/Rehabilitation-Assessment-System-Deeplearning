"""Pydantic schemas for the rehabilitation assessment API."""
from __future__ import annotations

from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class PatientInfo(BaseModel):
    patient_id: str = Field(..., description="患者编号")
    name: str = Field(..., description="姓名")
    sex: Literal["男", "女"]
    age: Optional[int] = Field(None, ge=0, le=120)
    diagnosis: str = Field(..., description="诊断")
    disease_days: Optional[int] = Field(None, ge=0)
    paralysis_side: Literal["左", "右"]


class PredictionResult(BaseModel):
    FMA_UE: float = Field(..., ge=0.0, le=20.0, description="FMA手部分数")
    hand_tone: str = Field(..., description='手部肌张力："0"/"1"/"1+"/"2"/"3"/"4"')
    hand_function: int = Field(..., ge=1, le=6, description="Brunnstrom分期 1–6")


class AssessSessionResponse(BaseModel):
    session_id: str
    n_trials: int


class AssessmentResult(BaseModel):
    session_id: str
    patient_info: PatientInfo
    predictions: PredictionResult


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    message: str


# --------------------------------------------------------------------------- #
# Patient management + records + stats (SQLite-backed)                         #
# --------------------------------------------------------------------------- #
class PatientUpdate(BaseModel):
    """PATCH semantics — all fields optional; patient_id is NOT editable."""

    name: Optional[str] = None
    sex: Optional[Literal["男", "女"]] = None
    age: Optional[int] = Field(None, ge=0, le=120)
    diagnosis: Optional[str] = None
    disease_days: Optional[int] = Field(None, ge=0)
    paralysis_side: Optional[Literal["左", "右"]] = None
    birth_date: Optional[str] = None      # 出生年月日 yyyy-mm-dd
    id_number: Optional[str] = None       # 身份证号
    phone: Optional[str] = None           # 手机号
    onset_date: Optional[str] = None      # 发病日期 yyyy-mm-dd


class AssessmentRecord(BaseModel):
    id: int
    session_id: Optional[str] = None
    created_at: str
    fma_ue: float
    hand_tone: str
    hand_function: int


class PatientSummary(BaseModel):
    id: int
    patient_id: str
    name: str
    sex: str
    age: Optional[int] = None
    diagnosis: str
    disease_days: Optional[int] = None
    paralysis_side: str
    birth_date: Optional[str] = None
    id_number: Optional[str] = None
    phone: Optional[str] = None
    onset_date: Optional[str] = None
    created_at: str
    updated_at: str
    assessment_count: int = 0
    last_assessed_at: Optional[str] = None


class PatientDetail(PatientSummary):
    assessments: List[AssessmentRecord] = []


class AssessmentOverviewItem(BaseModel):
    id: int
    created_at: str
    patient_db_id: int
    patient_id: str
    name: str
    fma_ue: float
    hand_tone: str
    hand_function: int


class AssessmentOverview(BaseModel):
    total: int
    items: List[AssessmentOverviewItem]


class StatsSummary(BaseModel):
    patient_count: int
    assessment_count: int
    diagnosis_distribution: Dict[str, int]
    hand_function_distribution: Dict[str, int]
    avg_fma_ue: Optional[float] = None
    assessments_by_day: List[Dict[str, Union[str, int]]]
