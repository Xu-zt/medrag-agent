import clsx from 'clsx'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { CheckCircle, AlertTriangle, Clock, RefreshCw } from 'lucide-react'
import { useStore, chunkColor } from '../store'

// ── Citation colour badges ─────────────────────────────────────────────────

const CITE_TEXT: Record<string, string> = {
  blue: 'text-blue-600',
  emerald: 'text-emerald-600',
  violet: 'text-violet-600',
  amber: 'text-amber-600',
  rose: 'text-rose-600',
  cyan: 'text-cyan-600',
  fuchsia: 'text-fuchsia-600',
  lime: 'text-lime-600',
}

// Build citation-to-index map
function citationIndex(citations: string[]): Record<string, number> {
  const map: Record<string, number> = {}
  citations.forEach((c, i) => { map[c] = i })
  return map
}

// Replace [PMID:xxx] and [PMC:xxx] in answer text with styled superscripts
function AnnotatedAnswer({
  text,
  citations,
  onCiteClick,
}: {
  text: string
  citations: string[]
  onCiteClick: (citation: string) => void
}) {
  const citeMap = citationIndex(citations)

  // Split on citation markers
  const CITE_RE = /\[(PMID:[^\]]+|PMC:[^\]]+)\]/g
  const parts: React.ReactNode[] = []
  let last = 0
  let match: RegExpExecArray | null

  while ((match = CITE_RE.exec(text)) !== null) {
    if (match.index > last) {
      parts.push(
        <ReactMarkdown key={last} remarkPlugins={[remarkGfm]} components={{ p: 'span' }}>
          {text.slice(last, match.index)}
        </ReactMarkdown>,
      )
    }
    const citeStr = match[1]
    const idx = citeMap[citeStr] ?? Object.keys(citeMap).length
    const color = chunkColor(idx)
    parts.push(
      <button
        key={match.index}
        className={clsx(
          'text-xs font-bold align-super cursor-pointer hover:underline',
          CITE_TEXT[color],
        )}
        onClick={() => onCiteClick(citeStr)}
        title={citeStr}
      >
        [{idx + 1}]
      </button>,
    )
    last = match.index + match[0].length
  }
  if (last < text.length) {
    parts.push(
      <ReactMarkdown key={last} remarkPlugins={[remarkGfm]} components={{ p: 'span' }}>
        {text.slice(last)}
      </ReactMarkdown>,
    )
  }

  return (
    <div className="prose prose-sm prose-slate max-w-none leading-relaxed">
      {parts}
    </div>
  )
}

// ── Faithfulness badge ────────────────────────────────────────────────────

function FaithfulnessBadge({
  faithful,
  issues,
  confidence,
  iterations,
  regenCount,
  latency,
}: {
  faithful: boolean
  issues: string
  confidence: number
  iterations: number
  regenCount: number
  latency: number
}) {
  return (
    <div
      className={clsx(
        'rounded-xl border px-4 py-3 mt-4 text-sm',
        faithful
          ? 'bg-emerald-50 border-emerald-200 text-emerald-800'
          : 'bg-amber-50 border-amber-200 text-amber-800',
      )}
    >
      <div className="flex items-center gap-2 font-semibold mb-1">
        {faithful ? <CheckCircle size={15} /> : <AlertTriangle size={15} />}
        {faithful ? 'All claims are literature-supported' : 'Unsupported claims detected'}
      </div>
      {!faithful && issues && (
        <p className="text-xs mt-1 opacity-80">{issues}</p>
      )}
      <div className="flex items-center gap-4 mt-2 text-xs opacity-70">
        <span>Confidence: {(confidence * 100).toFixed(0)}%</span>
        <span className="flex items-center gap-1">
          <RefreshCw size={11} />
          Rewrites: {iterations}
        </span>
        <span>Regen: {regenCount}</span>
        <span className="flex items-center gap-1">
          <Clock size={11} />
          {(latency / 1000).toFixed(1)}s
        </span>
      </div>
    </div>
  )
}

// ── AnswerPanel ────────────────────────────────────────────────────────────

export function AnswerPanel() {
  const { result, streamingAnswer, isStreaming, setSelectedChunkId } = useStore()

  const displayText = result?.answer ?? streamingAnswer

  function handleCiteClick(citation: string) {
    // Find the chunk by citation and scroll to / highlight it in EvidencePanel
    const chunks = result?.chunks ?? []
    const idx = chunks.findIndex((c) => c.citation === citation)
    if (idx >= 0) {
      const id = chunks[idx].chunk_id
      setSelectedChunkId(id)
      document.getElementById(`chunk-${id}`)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }

  if (!displayText && !isStreaming) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-slate-400 p-8 text-center">
        <div className="text-5xl mb-4 opacity-20">💊</div>
        <p className="text-base font-medium mb-2 text-slate-500">Ask a medical question</p>
        <p className="text-sm max-w-sm">
          VeritasMed retrieves evidence from PubMed and PMC, then generates a
          literature-grounded answer with automatic faithfulness verification.
        </p>
      </div>
    )
  }

  return (
    <div className="h-full overflow-y-auto p-6">
      <div className="max-w-prose">
        {displayText && (
          <AnnotatedAnswer
            text={displayText}
            citations={result?.citations ?? []}
            onCiteClick={handleCiteClick}
          />
        )}

        {isStreaming && !result && (
          <span className="inline-block w-2 h-4 bg-blue-500 animate-pulse ml-0.5" />
        )}

        {result && (
          <FaithfulnessBadge
            faithful={result.faithful}
            issues={result.faithfulness_issues}
            confidence={result.confidence}
            iterations={result.iterations}
            regenCount={result.regen_count}
            latency={result.latency_ms}
          />
        )}

        {(result?.rewritten_queries?.length ?? 0) > 0 && result && (
          <div className="mt-4 text-xs text-slate-400 space-y-1">
            <p className="font-semibold text-slate-500 flex items-center gap-1">
              <RefreshCw size={11} /> Query rewrites
            </p>
            {result.rewritten_queries.map((q, i) => (
              <p key={i} className="pl-3 border-l-2 border-slate-200 text-slate-500">{q}</p>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
