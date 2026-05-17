import { AgentTimeline } from '../components/AgentTimeline'
import { AnswerPanel } from '../components/AnswerPanel'
import { EvidencePanel } from '../components/EvidencePanel'
import { QueryInput } from '../components/QueryInput'

export function AnswerPage() {
  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <div style={{
        flex: 1,
        display: 'grid',
        gridTemplateColumns: 'minmax(220px, 0.9fr) minmax(0, 2.5fr) minmax(300px, 1.1fr)',
        overflow: 'hidden',
      }}>
        {/* Left: Reasoning trace */}
        <AgentTimeline />

        {/* Center: Answer */}
        <div style={{ overflow: 'hidden', borderRight: '1px solid var(--rule)' }}>
          <AnswerPanel />
        </div>

        {/* Right: Evidence */}
        <EvidencePanel />
      </div>

      {/* Bottom: Composer */}
      <QueryInput />
    </div>
  )
}
