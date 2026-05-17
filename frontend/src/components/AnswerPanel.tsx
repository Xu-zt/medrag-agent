import React from 'react'
import { useStore } from '../store'
import type { AnswerOut } from '../types/ws'

// ── SVG icons ──────────────────────────────────────────────────────────────
function I({ size = 16, sw = 1.6, children, style }: {
  size?: number; sw?: number; children: React.ReactNode; style?: React.CSSProperties
}) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth={sw} strokeLinecap="round"
      strokeLinejoin="round" aria-hidden="true" style={style}>
      {children}
    </svg>
  )
}
const IconCheck   = (p: { size?: number; sw?: number }) => <I {...p}><path d="m5 12 4 4L19 7"/></I>
const IconAlert   = (p: { size?: number; sw?: number }) => <I {...p}><path d="M12 9v4M12 17h.01"/><path d="M10.3 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0Z"/></I>
const IconArrow   = (p: { size?: number; sw?: number }) => <I {...p}><path d="M5 12h14M12 5l7 7-7 7"/></I>
const IconX       = (p: { size?: number; sw?: number }) => <I {...p}><path d="M18 6 6 18M6 6l12 12"/></I>

const CITE_VARS = ['--c0','--c1','--c2','--c3','--c4','--c5','--c6','--c7']

// ── Inline citation annotation ─────────────────────────────────────────────
function AnnotatedParagraph({ text, citations, onCiteClick, isFirst }: {
  text: string
  citations: string[]
  onCiteClick: (c: string) => void
  isFirst: boolean
}) {
  const CITE_RE = /\[(PMID:[^\]]+|PMC:[^\]]+)\]/g
  const parts: React.ReactNode[] = []
  let last = 0
  let match: RegExpExecArray | null
  let key = 0

  while ((match = CITE_RE.exec(text)) !== null) {
    if (match.index > last) {
      parts.push(<span key={key++}>{text.slice(last, match.index)}</span>)
    }
    const citeStr = match[1]
    const idx = citations.indexOf(citeStr)
    const colorVar = CITE_VARS[(idx >= 0 ? idx : citations.length) % CITE_VARS.length]
    parts.push(
      <button
        key={key++}
        className="vm-cite"
        style={{ '--cite-color': `var(${colorVar})` } as React.CSSProperties}
        onClick={() => onCiteClick(citeStr)}
        title={citeStr}
      >
        {idx >= 0 ? idx + 1 : '?'}
      </button>
    )
    last = match.index + match[0].length
  }
  if (last < text.length) parts.push(<span key={key++}>{text.slice(last)}</span>)

  return (
    <p style={{
      margin: '0 0 1.1em', fontFamily: 'var(--serif)', fontSize: 16,
      lineHeight: 1.72, color: 'var(--ink)', letterSpacing: '-0.005em',
    }}
      className={isFirst ? 'vm-dropcap' : undefined}
    >
      {parts}
    </p>
  )
}

// ── Verification mark ──────────────────────────────────────────────────────
function VerificationMark({ result }: { result: AnswerOut }) {
  const ok = result.faithful
  return (
    <div style={{
      marginTop: 28,
      padding: '14px 18px',
      borderRadius: 10,
      border: `1px solid ${ok ? 'var(--verified)' : 'var(--warn)'}`,
      background: ok ? 'var(--verified-soft)' : 'var(--warn-soft)',
      position: 'relative',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
        <div style={{ color: ok ? 'var(--verified)' : 'var(--warn)' }}>
          {ok ? <IconCheck size={15} sw={2.2} /> : <IconAlert size={15} sw={2.2} />}
        </div>
        <span style={{
          fontFamily: 'var(--serif)', fontStyle: 'italic', fontSize: 14,
          color: ok ? 'var(--verified)' : 'var(--warn)', letterSpacing: '-0.005em',
        }}>
          {ok ? 'All claims literature-supported' : 'Unsupported claims detected'}
        </span>
      </div>
      {!ok && result.faithfulness_issues && (
        <p style={{ margin: '0 0 8px', fontSize: 12, color: 'var(--ink-soft)', lineHeight: 1.5 }}>
          {result.faithfulness_issues}
        </p>
      )}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14, fontSize: 11 }}>
        <MetricChip label="Confidence" value={`${(result.confidence * 100).toFixed(0)}%`} />
        <MetricChip label="Rewrites"   value={String(result.iterations)} />
        <MetricChip label="Regen"      value={String(result.regen_count)} />
        <MetricChip label="Latency"    value={`${(result.latency_ms / 1000).toFixed(1)}s`} />
        <MetricChip label="Verifier"   value="NLI" />
      </div>
    </div>
  )
}

function MetricChip({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
      <span className="vm-eyebrow" style={{ fontSize: 8 }}>{label}</span>
      <span className="vm-mono" style={{ fontSize: 11, color: 'var(--ink-soft)', fontWeight: 600 }}>{value}</span>
    </div>
  )
}

// ── Sources strip ──────────────────────────────────────────────────────────
function SourcesStrip({ result, onCiteClick }: { result: AnswerOut; onCiteClick: (c: string) => void }) {
  if (!result.citations.length) return null
  return (
    <div style={{ marginTop: 18 }}>
      <div className="vm-eyebrow" style={{ marginBottom: 8 }}>Sources</div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
        {result.citations.map((c, i) => {
          const colorVar = CITE_VARS[i % CITE_VARS.length]
          const chunk = result.chunks.find((ch: AnswerOut['chunks'][0]) => ch.citation === c)
          return (
            <button
              key={c}
              onClick={() => onCiteClick(c)}
              title={chunk?.title ?? c}
              style={{
                display: 'inline-flex', alignItems: 'center', gap: 6,
                padding: '5px 10px', borderRadius: 5,
                border: `1px solid var(${colorVar})`,
                background: 'transparent',
                color: `var(${colorVar})`,
                fontSize: 11, fontWeight: 600, fontFamily: 'var(--mono)',
                transition: 'background 120ms',
              }}
            >
              <span style={{ fontFamily: 'var(--serif)', fontStyle: 'italic', fontSize: 14 }}>{i + 1}</span>
              {c}
            </button>
          )
        })}
      </div>
    </div>
  )
}

// ── Empty state ────────────────────────────────────────────────────────────
const SUGGESTED = [
  'What is the mechanism of metformin in type 2 diabetes?',
  'Efficacy of mRNA vaccines against severe COVID-19',
  'Risk factors for acute kidney injury in ICU patients',
]

function EmptyState({ onSuggest }: { onSuggest: (q: string) => void }) {
  return (
    <div style={{
      height: '100%', display: 'flex', flexDirection: 'column',
      alignItems: 'center', justifyContent: 'center',
      padding: '40px 32px', textAlign: 'center',
    }}>
      <div style={{
        fontFamily: 'var(--serif)', fontSize: 38, letterSpacing: '-0.03em',
        color: 'var(--ink)', lineHeight: 1, marginBottom: 10,
      }}>
        VeritasMed
      </div>
      <p style={{
        fontFamily: 'var(--serif)', fontStyle: 'italic', fontSize: 15,
        color: 'var(--muted)', margin: '0 0 28px', maxWidth: 420,
        lineHeight: 1.55,
      }}>
        Self-verifying medical QA — evidence retrieved, claims checked.
      </p>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8, width: '100%', maxWidth: 480 }}>
        {SUGGESTED.map((q) => (
          <button
            key={q}
            onClick={() => onSuggest(q)}
            style={{
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              padding: '12px 16px', borderRadius: 8, border: '1px solid var(--rule)',
              background: 'var(--panel)', color: 'var(--ink-soft)',
              fontSize: 13, fontFamily: 'var(--serif)', textAlign: 'left',
              lineHeight: 1.4, transition: 'border-color 120ms, color 120ms',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.borderColor = 'var(--accent)'
              e.currentTarget.style.color = 'var(--ink)'
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.borderColor = 'var(--rule)'
              e.currentTarget.style.color = 'var(--ink-soft)'
            }}
          >
            <span>{q}</span>
            <IconArrow size={13} sw={2} />
          </button>
        ))}
      </div>
    </div>
  )
}

// ── AnswerPanel ────────────────────────────────────────────────────────────
export function AnswerPanel() {
  const { result, isStreaming, setSelectedChunkId, setQuery, errorMessage } = useStore()

  const displayText = result?.answer ?? ''

  function handleCiteClick(citation: string) {
    const chunks = result?.chunks ?? []
    const idx = chunks.findIndex((c) => c.citation === citation)
    if (idx >= 0) {
      const id = chunks[idx].chunk_id
      setSelectedChunkId(id)
      document.getElementById(`chunk-${id}`)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }

  if (errorMessage && !isStreaming) {
    return (
      <div style={{
        height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 32,
      }}>
        <div style={{
          display: 'flex', gap: 12, padding: '16px 20px', borderRadius: 10,
          border: '1px solid var(--error)', background: 'var(--error-soft)',
          maxWidth: 520,
        }}>
          <div style={{ color: 'var(--error)', flexShrink: 0, paddingTop: 1 }}>
            <IconX size={16} sw={2} />
          </div>
          <div>
            <p style={{ margin: '0 0 4px', fontSize: 13, fontWeight: 600, color: 'var(--error)' }}>Error</p>
            <p className="vm-mono" style={{ margin: 0, fontSize: 11, color: 'var(--error)' }}>{errorMessage}</p>
          </div>
        </div>
      </div>
    )
  }

  if (!displayText && !isStreaming) {
    return <EmptyState onSuggest={(q) => setQuery(q)} />
  }

  const paragraphs = displayText.split(/\n\n+/).filter(Boolean)

  return (
    <div style={{ height: '100%', overflowY: 'auto', padding: '28px 32px 48px' }}>
      <div style={{ maxWidth: 660 }}>
        {isStreaming && !displayText && (
          <div style={{
            fontFamily: 'var(--serif)', fontSize: 16, color: 'var(--muted)',
          }} className="vm-caret" />
        )}

        {paragraphs.map((para, i) => (
          <AnnotatedParagraph
            key={i}
            text={para}
            citations={result?.citations ?? []}
            onCiteClick={handleCiteClick}
            isFirst={i === 0}
          />
        ))}

        {result && (
          <>
            <SourcesStrip result={result} onCiteClick={handleCiteClick} />
            <VerificationMark result={result} />
          </>
        )}
      </div>
    </div>
  )
}
