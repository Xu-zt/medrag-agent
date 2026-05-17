import React, { useEffect, useRef, useState } from 'react'
import { useStore } from '../store'
import { useAgentStream } from '../hooks/useAgentStream'
import { fetchCorpusStats, saveThread } from '../api/client'
import type { CorpusStats } from '../types'

// ── SVG icons ─────────────────────────────────────────────────────────────
function I({ size = 16, sw = 1.6, children }: { size?: number; sw?: number; children: React.ReactNode }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth={sw} strokeLinecap="round"
      strokeLinejoin="round" aria-hidden="true">
      {children}
    </svg>
  )
}
const IconArrowUp  = (p: { size?: number; sw?: number }) => <I {...p}><path d="M12 19V5M6 11l6-6 6 6"/></I>
const IconStop     = (p: { size?: number; sw?: number }) => <I {...p}><rect x="6" y="6" width="12" height="12" rx="1.5"/></I>
const IconDatabase = (p: { size?: number; sw?: number }) => <I {...p}><ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v6c0 1.66 3.58 3 8 3s8-1.34 8-3V5M4 11v6c0 1.66 3.58 3 8 3s8-1.34 8-3v-6"/></I>
const IconLightning = (p: { size?: number; sw?: number }) => <I {...p}><path d="M13 2 4 14h7l-1 8 9-12h-7l1-8Z"/></I>
const IconNetwork  = (p: { size?: number; sw?: number }) => <I {...p}><circle cx="6" cy="6" r="2.5"/><circle cx="18" cy="6" r="2.5"/><circle cx="6" cy="18" r="2.5"/><circle cx="18" cy="18" r="2.5"/><path d="M8 7 16 17M16 7 8 17"/></I>

// ── Pipeline toggle ────────────────────────────────────────────────────────
function PipelineToggle({ value, onChange, disabled }: {
  value: 'p2' | 'p3'; onChange: (v: 'p2' | 'p3') => void; disabled: boolean
}) {
  const options: { id: 'p2' | 'p3'; label: string; sub: string; icon: React.FC<{ size?: number; sw?: number }>; hint: string }[] = [
    { id: 'p2', label: 'Hybrid',   sub: 'p2', icon: IconLightning, hint: 'Fast · dense+sparse retrieval' },
    { id: 'p3', label: 'Reranked', sub: 'p3', icon: IconNetwork,   hint: 'Precise · BGE-reranker post-processing' },
  ]
  return (
    <div style={{
      display: 'inline-flex', alignItems: 'center',
      padding: 2, gap: 2,
      background: 'var(--panel-2)',
      border: '1px solid var(--rule-soft)',
      borderRadius: 7,
    }}>
      {options.map((o) => {
        const active = value === o.id
        const Glyph = o.icon
        return (
          <button
            key={o.id}
            onClick={() => onChange(o.id)}
            disabled={disabled}
            title={o.hint}
            style={{
              display: 'inline-flex', alignItems: 'center', gap: 5,
              padding: '5px 9px',
              borderRadius: 5, border: 'none',
              background: active ? 'var(--panel)' : 'transparent',
              boxShadow: active ? '0 1px 2px oklch(50% 0.02 80 / 0.10)' : 'none',
              color: active ? 'var(--ink)' : 'var(--muted)',
              fontSize: 11, fontWeight: 600, letterSpacing: '0.005em',
              transition: 'all 120ms',
              cursor: disabled ? 'not-allowed' : 'pointer',
              opacity: disabled ? 0.5 : 1,
            }}
          >
            <Glyph size={11} sw={2} />
            {o.label}
            <span className="vm-mono" style={{ fontSize: 9.5, color: 'var(--faint)', fontWeight: 500 }}>
              {o.sub}
            </span>
          </button>
        )
      })}
    </div>
  )
}

// ── Thread pill ────────────────────────────────────────────────────────────
function ThreadPill({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  return (
    <div style={{
      display: 'inline-flex', alignItems: 'center', gap: 6,
      padding: '5px 9px',
      border: '1px solid var(--rule-soft)',
      borderRadius: 6,
      fontSize: 11, color: 'var(--muted)',
    }}>
      <span className="vm-eyebrow" style={{ fontSize: 9, letterSpacing: '0.12em' }}>Thread</span>
      <input
        value={value}
        onChange={(e) => onChange(e.target.value)}
        style={{
          width: 92, border: 'none', outline: 'none', background: 'transparent',
          fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--ink-soft)', fontWeight: 600,
        }}
      />
    </div>
  )
}

// ── QueryInput ─────────────────────────────────────────────────────────────
export function QueryInput() {
  const { query, setQuery, threadId, setThreadId, pipeline, setPipeline, isStreaming } = useStore()
  const { send, cancel } = useAgentStream()
  const [stats, setStats] = useState<CorpusStats | null>(null)
  const [focused, setFocused] = useState(false)
  const taRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    fetchCorpusStats().then(setStats).catch(() => null)
  }, [])

  // Auto-grow textarea
  useEffect(() => {
    const ta = taRef.current
    if (!ta) return
    ta.style.height = 'auto'
    ta.style.height = Math.min(160, Math.max(24, ta.scrollHeight)) + 'px'
  }, [query])

  function handleKey(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      if (!isStreaming && query.trim()) {
        saveThread(threadId)
        send()
      }
    }
  }

  const canSend = query.trim().length > 0 && !isStreaming

  return (
    <div style={{
      padding: '14px 32px 20px',
      background: 'var(--canvas)',
      flexShrink: 0,
    }}>
      <div style={{
        maxWidth: 1200, margin: '0 auto',
        background: 'var(--panel)',
        border: `1px solid ${focused ? 'var(--accent)' : 'var(--rule)'}`,
        borderRadius: 12,
        transition: 'border-color 160ms, box-shadow 160ms',
        boxShadow: focused
          ? '0 0 0 4px var(--accent-soft), var(--shadow-float)'
          : 'var(--shadow-float)',
      }}>
        {/* Textarea */}
        <div style={{ padding: '14px 18px 6px' }}>
          <textarea
            ref={taRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKey}
            onFocus={() => setFocused(true)}
            onBlur={() => setFocused(false)}
            disabled={isStreaming}
            rows={1}
            placeholder="Ask a clinical question — enter to send, shift+enter for a new line"
            style={{
              width: '100%', resize: 'none',
              border: 'none', outline: 'none', background: 'transparent',
              fontFamily: 'var(--serif)',
              fontSize: 18, lineHeight: 1.4,
              color: 'var(--ink)',
              letterSpacing: '-0.005em',
              minHeight: 24, maxHeight: 160,
            }}
          />
        </div>

        {/* Controls strip */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: 10,
          padding: '8px 12px 10px 14px',
          borderTop: '1px solid var(--rule-soft)',
        }}>
          <PipelineToggle value={pipeline} onChange={setPipeline} disabled={isStreaming} />
          <ThreadPill value={threadId} onChange={setThreadId} />

          {stats && (
            <span className="vm-mono" style={{
              marginLeft: 'auto',
              fontSize: 10.5, color: 'var(--faint)',
              display: 'inline-flex', alignItems: 'center', gap: 6,
            }}>
              <IconDatabase size={11} sw={2} />
              {stats.total_chunks.toLocaleString()} chunks · {stats.embedding_model}
            </span>
          )}

          <button
            onClick={isStreaming ? cancel : () => { if (canSend) { saveThread(threadId); send() } }}
            disabled={!isStreaming && !canSend}
            style={{
              display: 'inline-flex', alignItems: 'center', gap: 6,
              padding: '7px 12px 7px 14px',
              borderRadius: 8, border: '1px solid transparent',
              background: isStreaming ? 'var(--error)' : canSend ? 'var(--ink)' : 'var(--rule)',
              color: isStreaming ? 'white' : canSend ? 'var(--canvas)' : 'var(--faint)',
              fontSize: 12, fontWeight: 600, letterSpacing: '0.01em',
              cursor: isStreaming || canSend ? 'pointer' : 'not-allowed',
              transition: 'all 120ms',
              marginLeft: stats ? 0 : 'auto',
            }}
          >
            {isStreaming
              ? <><IconStop size={11} sw={2.2} /> Stop</>
              : <>Send <IconArrowUp size={11} sw={2.4} /></>}
          </button>
        </div>
      </div>
    </div>
  )
}
