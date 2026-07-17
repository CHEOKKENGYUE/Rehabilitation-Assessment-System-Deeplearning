import { AssessmentRecord } from '../types'

// Renders the FMA indicator for one persisted assessment record — a compact,
// centered card (the assessment page uses the larger radial gauge).
export default function RecordDetail({ record }: { record: AssessmentRecord }) {
  const v = Math.round(record.fma_ue)
  const pct = Math.max(0, Math.min(100, (record.fma_ue / 20) * 100))
  return (
    <div className="record-detail">
      <div className="record-fma">
        <div className="record-fma-head">
          <span className="fma-label">FMA-UE 手部分数</span>
          <span className="record-fma-value">
            {v}
            <span className="fma-unit">/ 20 分</span>
          </span>
        </div>
        <div className="progress-bar">
          <div style={{ width: `${pct}%` }} />
        </div>
      </div>
    </div>
  )
}
