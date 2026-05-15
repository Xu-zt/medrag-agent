import { useState } from 'react'
import clsx from 'clsx'
import { ExternalLink, ChevronDown, ChevronUp } from 'lucide-react'
import { useStore, chunkColor } from '../store'
import { fetchChunk } from '../api/client'
import type { ChunkOut, ChunkContextResponse } from '../types'

// ── Score bar ─────────────────────────────────────────────────────────────

function ScoreBar({ score }: { score: number | null }) {
  if (score === null) return null
  const pct = Math.min(1, Math.max(0, score)) * 100
  const color = score >= 0.7 ? 'bg-emerald-400' : score >= 0.4 ? 'bg-amber-400' : 'bg-rose-400'
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs font-mono text-slate-500 w-10 text-right">
        {score.toFixed(2)}
      </span>
      <div className="flex-1 h-1.5 bg-slate-100 rounded-full overflow-hidden">
        <div className={clsx('h-full rounded-full', color)} style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}

// ── Highlighted text ──────────────────────────────────────────────────────

function HighlightedText({
  text,
  ranges,
}: {
  text: string
  ranges: [number, number][]
}) {
  if (!ranges.length) return <span>{text}</span>

  // Sort ranges
  const sorted = [...ranges].sort((a, b) => a[0] - b[0])
  const parts: React.ReactNode[] = []
  let cursor = 0
  for (const [start, end] of sorted) {
    if (start > cursor) parts.push(<span key={cursor}>{text.slice(cursor, start)}</span>)
    parts.push(
      <mark key={start} className="bg-yellow-200 text-yellow-900 rounded-sm px-0.5">
        {text.slice(start, end)}
      </mark>,
    )
    cursor = end
  }
  if (cursor < text.length) parts.push(<span key={cursor}>{text.slice(cursor)}</span>)
  return <>{parts}</>
}

// ── Context expansion ─────────────────────────────────────────────────────

function ContextExpander({ chunkId }: { chunkId: string }) {
  const [open, setOpen] = useState(false)
  const [ctx, setCtx] = useState<ChunkContextResponse | null>(null)
  const [loading, setLoading] = useState(false)

  async function toggle() {
    if (!open && !ctx) {
      setLoading(true)
      try {
        const data = await fetchChunk(chunkId, 1)
        setCtx(data)
      } catch {
        // ignore
      } finally {
        setLoading(false)
      }
    }
    setOpen((v) => !v)
  }

  return (
    <div>
      <button
        onClick={toggle}
        className="flex items-center gap-1 text-xs text-blue-500 hover:text-blue-700 transition-colors"
      >
        {open ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
        {loading ? 'Loading…' : open ? 'Hide context' : 'View context'}
      </button>
      {open && ctx && (
        <div className="mt-2 space-y-2">
          {ctx.prev_chunk && (
            <div className="text-xs text-slate-400 bg-slate-50 rounded p-2 border-l-2 border-slate-200">
              <span className="font-semibold block mb-1 text-slate-500">↑ Previous chunk</span>
              {ctx.prev_chunk.text.slice(0, 300)}…
            </div>
          )}
          {ctx.next_chunk && (
            <div className="text-xs text-slate-400 bg-slate-50 rounded p-2 border-l-2 border-slate-200">
              <span className="font-semibold block mb-1 text-slate-500">↓ Next chunk</span>
              {ctx.next_chunk.text.slice(0, 300)}…
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ── Color accent map ──────────────────────────────────────────────────────

const COLOR_ACCENT: Record<string, string> = {
  blue: 'border-blue-400 bg-blue-50',
  emerald: 'border-emerald-400 bg-emerald-50',
  violet: 'border-violet-400 bg-violet-50',
  amber: 'border-amber-400 bg-amber-50',
  rose: 'border-rose-400 bg-rose-50',
  cyan: 'border-cyan-400 bg-cyan-50',
  fuchsia: 'border-fuchsia-400 bg-fuchsia-50',
  lime: 'border-lime-400 bg-lime-50',
}

const BADGE_COLOR: Record<string, string> = {
  blue: 'bg-blue-500',
  emerald: 'bg-emerald-500',
  violet: 'bg-violet-500',
  amber: 'bg-amber-500',
  rose: 'bg-rose-500',
  cyan: 'bg-cyan-500',
  fuchsia: 'bg-fuchsia-500',
  lime: 'bg-lime-500',
}

// ── Chunk card ────────────────────────────────────────────────────────────

function ChunkCard({
  chunk,
  idx,
  isSelected,
  onSelect,
}: {
  chunk: ChunkOut
  idx: number
  isSelected: boolean
  onSelect: (id: string) => void
}) {
  const color = chunkColor(idx)
  const accent = COLOR_ACCENT[color]
  const badge = BADGE_COLOR[color]

  return (
    <div
      id={`chunk-${chunk.chunk_id}`}
      className={clsx(
        'rounded-xl border-l-4 p-4 cursor-pointer transition-all',
        accent,
        isSelected && 'ring-2 ring-offset-1 ring-blue-400',
      )}
      onClick={() => onSelect(chunk.chunk_id)}
    >
      {/* Header */}
      <div className="flex items-start justify-between gap-2 mb-2">
        <div className="flex items-center gap-2">
          <span
            className={clsx('text-white text-xs font-bold rounded-full w-5 h-5 flex items-center justify-center', badge)}
          >
            {idx + 1}
          </span>
          <span className="text-xs font-semibold text-slate-600">{chunk.citation}</span>
          <span className={clsx(
            'text-xs px-1.5 py-0.5 rounded-full font-medium',
            chunk.source === 'pubmed'
              ? 'bg-blue-100 text-blue-700'
              : 'bg-purple-100 text-purple-700',
          )}>
            {chunk.source === 'pubmed' ? 'PubMed' : 'PMC'}
          </span>
        </div>
      </div>

      {/* Score */}
      <div className="mb-2">
        <ScoreBar score={chunk.score ?? null} />
      </div>

      {/* Title */}
      <p className="text-xs font-medium text-slate-700 mb-2 line-clamp-2">{chunk.title}</p>
      {chunk.section && (
        <p className="text-xs text-slate-400 mb-1">{chunk.section}</p>
      )}

      {/* Text */}
      <p className="text-xs text-slate-600 leading-relaxed line-clamp-4">
        <HighlightedText text={chunk.text} ranges={chunk.highlight_ranges ?? []} />
      </p>

      {/* Actions */}
      <div className="mt-3 flex items-center gap-4">
        <ContextExpander chunkId={chunk.chunk_id} />
        {chunk.external_url && (
          <a
            href={chunk.external_url}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1 text-xs text-slate-400 hover:text-blue-600 transition-colors"
            onClick={(e) => e.stopPropagation()}
          >
            <ExternalLink size={11} />
            {chunk.source === 'pubmed' ? 'PubMed' : 'PMC'}
          </a>
        )}
      </div>
    </div>
  )
}

// ── EvidencePanel ─────────────────────────────────────────────────────────

export function EvidencePanel() {
  const { result, liveChunks, selectedChunkId, setSelectedChunkId, isStreaming } = useStore()

  const chunks = result?.chunks?.length ? result.chunks : liveChunks

  if (!chunks.length && !isStreaming) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-slate-400 p-4 text-center">
        <div className="text-3xl mb-3 opacity-30">📄</div>
        <p className="text-sm">Retrieved evidence will appear here</p>
      </div>
    )
  }

  function handleSelect(id: string) {
    setSelectedChunkId(selectedChunkId === id ? null : id)
  }

  return (
    <div className="h-full overflow-y-auto p-4">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
          Evidence
        </h2>
        <span className="text-xs text-slate-400">{chunks.length} chunk{chunks.length !== 1 ? 's' : ''}</span>
      </div>

      <div className="space-y-3">
        {chunks.map((chunk, i) => (
          <ChunkCard
            key={chunk.chunk_id}
            chunk={chunk}
            idx={i}
            isSelected={selectedChunkId === chunk.chunk_id}
            onSelect={handleSelect}
          />
        ))}
      </div>
    </div>
  )
}
