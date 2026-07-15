import { useCallback, useEffect, useRef, useState } from 'react'
import {
  PredictionEntry,
  SSEEvent,
  StepKey,
  StepState,
  TaskKey,
} from '../types'

export type StreamPhase = 'idle' | 'processing' | 'done'

const STEP_DEFS: { key: StepKey; label: string }[] = [
  { key: 'parse', label: '文件解析与校验' },
  { key: 'preprocess', label: '信号预处理' },
  { key: 'alignment', label: '多模态时序对齐' },
  { key: 'feature_extract', label: '多尺度特征提取' },
  { key: 'graph_fusion', label: '跨模态图注意力融合' },
  { key: 'inference', label: '模型推理' },
]

export function freshSteps(): StepState[] {
  return STEP_DEFS.map((s) => ({ ...s, status: 'pending', details: [] }))
}

/**
 * Shared consumer for the `/api/assess/{id}/stream` SSE flow. Owns the steps /
 * predictions state and the EventSource lifecycle so the 康复评估 page drives
 * the backend pipeline without duplicating event handling.
 */
export function useAssessmentStream() {
  const [phase, setPhase] = useState<StreamPhase>('idle')
  const [steps, setSteps] = useState<StepState[]>(freshSteps)
  const [results, setResults] = useState<Partial<Record<TaskKey, PredictionEntry>>>({})
  const [error, setError] = useState<string | null>(null)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const esRef = useRef<EventSource | null>(null)

  useEffect(() => () => esRef.current?.close(), [])

  const updateStep = useCallback((key: StepKey, mutator: (s: StepState) => StepState) => {
    setSteps((prev) => prev.map((s) => (s.key === key ? mutator(s) : s)))
  }, [])

  const handleEvent = useCallback(
    (event: SSEEvent) => {
      switch (event.type) {
        case 'step_start':
          updateStep(event.step, (s) => ({ ...s, status: 'running', label: event.label || s.label }))
          break
        case 'step_detail':
          updateStep(event.step, (s) => ({ ...s, details: [...s.details, event.detail] }))
          break
        case 'step_done':
          updateStep(event.step, (s) => ({ ...s, status: 'done' }))
          break
        case 'prediction':
          setResults((prev) => ({
            ...prev,
            [event.task]: { task: event.task, label: event.label, value: event.value, range: event.range },
          }))
          break
        case 'done':
          setPhase('done')
          esRef.current?.close()
          break
        case 'error':
          setError(event.message)
          break
      }
    },
    [updateStep],
  )

  /** Reset transient state and begin consuming the stream for a session id. */
  const start = useCallback(
    (newSessionId: string) => {
      setSteps(freshSteps())
      setResults({})
      setError(null)
      setSessionId(newSessionId)
      setPhase('processing')

      const es = new EventSource(`/api/assess/${newSessionId}/stream`)
      esRef.current = es
      es.onmessage = (e) => {
        try {
          handleEvent(JSON.parse(e.data) as SSEEvent)
        } catch (parseErr) {
          console.error('SSE parse error', parseErr, e.data)
        }
      }
      es.onerror = () => {
        // EventSource auto-reconnects; leave UI as-is during processing.
      }
    },
    [handleEvent],
  )

  const reset = useCallback(() => {
    esRef.current?.close()
    esRef.current = null
    setPhase('idle')
    setSteps(freshSteps())
    setResults({})
    setError(null)
    setSessionId(null)
  }, [])

  return {
    phase,
    steps,
    results,
    error,
    sessionId,
    setError,
    start,
    reset,
  }
}
