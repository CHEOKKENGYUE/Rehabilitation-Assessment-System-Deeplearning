import { PredictionEntry, TaskKey } from '../types'

interface Props {
  results: Partial<Record<TaskKey, PredictionEntry>>
}

// FMA-UE hand subscore (0–20) → upper-limb motor impairment band.
function fmaBand(v: number): string {
  if (v <= 5) return '重度上肢运动功能障碍'
  if (v <= 10) return '中重度上肢运动功能障碍'
  if (v <= 15) return '中度上肢运动功能障碍'
  return '轻度上肢运动功能障碍'
}

// Radial dial geometry — a 270° arc gauge (gap centered at the bottom).
const R = 64
const C = 2 * Math.PI * R
const ARC = 0.75 // 270° of the full circle
const TRACK = ARC * C

export default function ResultsPanel({ results }: Props) {
  const entry = results.FMA_UE
  if (!entry) return null

  const v = typeof entry.value === 'number' ? entry.value : parseFloat(String(entry.value))
  const pct = Math.max(0, Math.min(100, (v / 20) * 100))
  const progress = (pct / 100) * TRACK

  return (
    <div className="card">
      <h2>
        评估结果
        <span className="h2-suffix">Clinical · Score</span>
      </h2>
      <div className="fma-result">
        <div className="fma-gauge">
          <svg viewBox="0 0 160 160" role="img" aria-label={`FMA 手部分数 ${v.toFixed(0)} / 20 分`}>
            <defs>
              <linearGradient id="fmaArc" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stopColor="var(--teal-500)" />
                <stop offset="100%" stopColor="var(--cyan-400)" />
              </linearGradient>
            </defs>
            <g transform="rotate(135 80 80)">
              <circle
                className="fma-gauge-track"
                cx="80"
                cy="80"
                r={R}
                strokeDasharray={`${TRACK} ${C}`}
              />
              <circle
                className="fma-gauge-progress"
                cx="80"
                cy="80"
                r={R}
                strokeDasharray={`${progress} ${C}`}
              />
            </g>
          </svg>
          <div className="fma-gauge-center">
            <span className="fma-score">{v.toFixed(0)}</span>
            <span className="fma-unit">/ 20 分</span>
          </div>
        </div>
        <div className="fma-caption">
          <div className="fma-label">{entry.label || 'FMA 手部分数'}</div>
          <div className="fma-band">{fmaBand(v)}</div>
        </div>
      </div>
    </div>
  )
}
