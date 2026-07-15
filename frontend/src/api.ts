import {
  AssessmentOverview,
  PatientDetail,
  PatientSummary,
  PatientUpdate,
  StatsSummary,
} from './types'

async function getJSON<T>(url: string): Promise<T> {
  const res = await fetch(url)
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(detail.detail || `HTTP ${res.status}`)
  }
  return res.json() as Promise<T>
}

export function fetchPatients(): Promise<PatientSummary[]> {
  return getJSON('/api/patients')
}

export function fetchPatient(id: number): Promise<PatientDetail> {
  return getJSON(`/api/patients/${id}`)
}

export async function updatePatient(
  id: number,
  payload: PatientUpdate,
): Promise<PatientDetail> {
  const res = await fetch(`/api/patients/${id}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(detail.detail || `HTTP ${res.status}`)
  }
  return res.json()
}

export function fetchAssessments(limit = 50, offset = 0): Promise<AssessmentOverview> {
  return getJSON(`/api/assessments?limit=${limit}&offset=${offset}`)
}

export function fetchStats(): Promise<StatsSummary> {
  return getJSON('/api/stats/summary')
}

export function fetchHealth(): Promise<{ status: string; models_loaded: string[] }> {
  return getJSON('/api/health')
}
