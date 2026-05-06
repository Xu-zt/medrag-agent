import clsx from 'clsx'
import { CheckCircle, Circle, AlertCircle, RefreshCw, Loader2, XCircle } from 'lucide-react'
import { useStore } from '../store'
import type { TimelineNode } from '../types'

const STATUS_ICON: Record<string, React.ReactNode> = {
  waiting: <Circle size={16} className="text-slate-400" />,
  running: <Loader2 size={16} className="text-blue-500 animate-spin" />,
  done: <CheckCircle size={16} className="text-emerald-500" />,
  rewrite: <RefreshCw size={16} className="text-amber-500" />,
  error: <XCircle size={16} className="text-rose-500" />,
}

const STATUS_LINE: Record<string, string> = {
  waiting: 'bg-slate-300',
  running: 'bg-blue-400',
  done: 'bg-emerald-400',
  rewrite: 'bg-amber-400',
  error: 'bg-rose-400',
}

function NodeCard({ node, isLast }: { node: TimelineNode; isLast: boolean }) {
  const rewriteDetail = node.name === 'rewrite' && node.detail
  const gradeDetail = node.name === 'grade' && node.detail

  return (
    <div className="flex gap-3">
      {/* Vertical connector */}
      <div className="flex flex-col items-center">
        <div className="mt-1">{STATUS_ICON[node.status]}</div>
        {!isLast && (
          <div className={clsx('w-px flex-1 mt-1', STATUS_LINE[node.status])} />
        )}
      </div>

      {/* Card body */}
      <div className="pb-4 flex-1 min-w-0">
        <div className="flex items-baseline gap-2">
          <span className="text-sm font-semibold text-slate-700">{node.label}</span>
          {node.status === 'running' && (
            <span className="text-xs text-slate-400 animate-pulse">running…</span>
          )}
        </div>

        {node.summary && (
          <p className="text-xs text-slate-500 mt-0.5 truncate">{node.summary}</p>
        )}

        {/* Grade score bar */}
        {gradeDetail && node.status === 'done' && (
          <div className="mt-1.5">
            <div className="h-1.5 w-full bg-slate-100 rounded-full overflow-hidden">
              <div
                className={clsx(
                  'h-full rounded-full transition-all',
                  (gradeDetail as { relevance_score?: number }).relevance_score! >= 0.6
                    ? 'bg-emerald-400'
                    : 'bg-amber-400',
                )}
                style={{
                  width: `${((gradeDetail as { relevance_score?: number }).relevance_score ?? 0) * 100}%`,
                }}
              />
            </div>
          </div>
        )}

        {/* Rewrite diff */}
        {rewriteDetail && (
          <div className="mt-1.5 space-y-1 text-xs">
            <div className="bg-rose-50 border border-rose-200 rounded px-2 py-1 font-mono text-rose-700 truncate">
              − {String((rewriteDetail as Record<string, unknown>)['rewritten_queries'] instanceof Array
                ? ''
                : '')}
            </div>
            <div className="bg-emerald-50 border border-emerald-200 rounded px-2 py-1 font-mono text-emerald-700 truncate">
              + {String((rewriteDetail as { new_query?: unknown }).new_query ?? '')}
            </div>
          </div>
        )}

        {/* Check result */}
        {node.name === 'check' && node.status === 'done' && node.detail && (
          <div className={clsx(
            'mt-1.5 flex items-center gap-1.5 text-xs px-2 py-1 rounded',
            (node.detail as { faithful?: boolean }).faithful
              ? 'bg-emerald-50 text-emerald-700'
              : 'bg-amber-50 text-amber-700',
          )}>
            {(node.detail as { faithful?: boolean }).faithful
              ? <CheckCircle size={12} />
              : <AlertCircle size={12} />}
            <span>
              {(node.detail as { faithful?: boolean }).faithful
                ? 'All claims supported'
                : (node.detail as { issues?: string }).issues?.slice(0, 80) || 'Issues found'}
            </span>
          </div>
        )}
      </div>
    </div>
  )
}

export function AgentTimeline() {
  const { timeline, isStreaming } = useStore()

  if (timeline.length === 0 && !isStreaming) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-slate-400 p-4 text-center">
        <Circle size={24} className="mb-3 opacity-40" />
        <p className="text-sm">Reasoning steps will appear here</p>
      </div>
    )
  }

  return (
    <div className="h-full overflow-y-auto p-4 space-y-0">
      <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-4">
        Reasoning Process
      </h2>
      {timeline.map((node, i) => (
        <NodeCard key={`${node.name}-${i}`} node={node} isLast={i === timeline.length - 1} />
      ))}
    </div>
  )
}
