export type Sex = '男' | '女'
export type ParalysisSide = '左' | '右'

// Age / disease_days use `number | ''` so the form fields can be genuinely
// empty (no forced 0, no leading-zero artifacts).
export interface PatientInfo {
  patient_id: string
  name: string
  sex: Sex
  age: number | ''
  diagnosis: string
  disease_days: number | ''
  paralysis_side: ParalysisSide
}

export const DIAGNOSIS_OPTIONS = ['脑外伤', '脑梗死', '脑出血', '其他'] as const

// Navigation routes (no react-router; route enum drives the AppShell). ------ //
export type Route =
  | 'dashboard'
  | 'patients'
  | 'assessment'
  | 'records'
  | 'stats'
  | 'system'

// Backend-mirrored persistence types --------------------------------------- //
export interface AssessmentRecord {
  id: number
  session_id: string | null
  created_at: string
  fma_ue: number
  hand_tone: string
  hand_function: number
}

export interface PatientSummary {
  id: number
  patient_id: string
  name: string
  sex: string
  age: number | null
  diagnosis: string
  disease_days: number | null
  paralysis_side: string
  birth_date: string | null
  id_number: string | null
  phone: string | null
  onset_date: string | null
  created_at: string
  updated_at: string
  assessment_count: number
  last_assessed_at: string | null
}

export interface PatientDetail extends PatientSummary {
  assessments: AssessmentRecord[]
}

export interface PatientUpdate {
  name?: string
  sex?: Sex
  age?: number | null
  diagnosis?: string
  disease_days?: number | null
  paralysis_side?: ParalysisSide
  birth_date?: string | null
  id_number?: string | null
  phone?: string | null
  onset_date?: string | null
}

export interface AssessmentOverviewItem {
  id: number
  created_at: string
  patient_db_id: number
  patient_id: string
  name: string
  fma_ue: number
  hand_tone: string
  hand_function: number
}

export interface AssessmentOverview {
  total: number
  items: AssessmentOverviewItem[]
}

export interface StatsSummary {
  patient_count: number
  assessment_count: number
  diagnosis_distribution: Record<string, number>
  hand_function_distribution: Record<string, number>
  avg_fma_ue: number | null
  assessments_by_day: { date: string; count: number }[]
}

export type StepKey =
  | 'parse'
  | 'preprocess'
  | 'alignment'
  | 'feature_extract'
  | 'graph_fusion'
  | 'inference'

export type StepStatus = 'pending' | 'running' | 'done'

export interface StepState {
  key: StepKey
  label: string
  status: StepStatus
  details: string[]
}

export type TaskKey = 'FMA_UE' | 'hand_tone' | 'hand_function'

export interface PredictionEntry {
  task: TaskKey
  label: string
  value: number | string
  range?: string
}

// SSE event union ------------------------------------------------------- //
export type SSEEvent =
  | { type: 'step_start'; step: StepKey; label: string }
  | { type: 'step_detail'; step: StepKey; detail: string }
  | { type: 'step_done'; step: StepKey }
  | {
      type: 'prediction'
      task: TaskKey
      value: number | string
      label: string
      range?: string
    }
  | { type: 'done' }
  | { type: 'error'; message: string }
