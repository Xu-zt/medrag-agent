import { useCallback } from 'react'
import { AgentTimeline } from '../components/AgentTimeline'
import { AnswerPanel } from '../components/AnswerPanel'
import { EvidencePanel } from '../components/EvidencePanel'
import { QueryInput } from '../components/QueryInput'
import { useAgentStream } from '../hooks/useAgentStream'
import { useStore } from '../store'

const SUGGESTED_QUERIES = [
  'Efficacy of SGLT2 inhibitors for heart failure with preserved ejection fraction',
  'Optimal duration of dual antiplatelet therapy after drug-eluting stent placement',
  'Tirzepatide vs semaglutide for weight loss in non-diabetic adults',
  'CAR-T cell therapy outcomes in relapsed diffuse large B-cell lymphoma',
]

export function AnswerPage() {
  const { setQuery, setSelectedChunkId, activeQuery, result } = useStore()
  const { send } = useAgentStream()

  const handleCiteClick = useCallback((citation: string) => {
    const chunks = result?.chunks ?? []
    const idx = chunks.findIndex((c) => c.citation === citation)
    if (idx >= 0) {
      const id = chunks[idx].chunk_id
      setSelectedChunkId(id)
      document.getElementById(`chunk-${id}`)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }, [result, setSelectedChunkId])

  const handlePickQuery = useCallback((q: string) => {
    setQuery(q)
    send(q)
  }, [setQuery, send])

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <div style={{
        flex: 1,
        display: 'grid',
        gridTemplateColumns: 'minmax(260px, 1fr) minmax(0, 2fr) minmax(300px, 1.2fr)',
        overflow: 'hidden',
      }}>
        {/* Left: Reasoning trace */}
        <AgentTimeline />

        {/* Center: Answer */}
        <div style={{ overflow: 'hidden', borderRight: '1px solid var(--rule)' }}>
          <AnswerPanel
            query={activeQuery}
            suggestedQueries={SUGGESTED_QUERIES}
            onCiteClick={handleCiteClick}
            onPickQuery={handlePickQuery}
          />
        </div>

        {/* Right: Evidence */}
        <EvidencePanel />
      </div>

      {/* Bottom: Composer */}
      <QueryInput />
    </div>
  )
}
