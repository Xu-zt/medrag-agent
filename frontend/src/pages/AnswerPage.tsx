import { AgentTimeline } from '../components/AgentTimeline'
import { AnswerPanel } from '../components/AnswerPanel'
import { EvidencePanel } from '../components/EvidencePanel'
import { QueryInput } from '../components/QueryInput'

export function AnswerPage() {
  return (
    <div className="h-full flex flex-col overflow-hidden">
      {/* Three-column layout */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left: Agent Timeline (280px) */}
        <div className="w-70 flex-shrink-0 border-r border-slate-200 overflow-hidden">
          <AgentTimeline />
        </div>

        {/* Center: Answer Panel (flex-1) */}
        <div className="flex-1 overflow-hidden">
          <AnswerPanel />
        </div>

        {/* Right: Evidence Panel (360px) */}
        <div className="w-90 flex-shrink-0 border-l border-slate-200 overflow-hidden">
          <EvidencePanel />
        </div>
      </div>

      {/* Footer: Query Input */}
      <QueryInput />
    </div>
  )
}
