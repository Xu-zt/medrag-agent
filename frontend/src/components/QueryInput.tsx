import React, { useEffect, useRef, useState } from 'react'
import { useStore } from '../store'
import { useAgentStream } from '../hooks/useAgentStream'
import { fetchCorpusStats, loadRecentThreads, saveThread } from '../api/client'
import type { CorpusStats } from '../types'

// ── SVG icons ──────────────────────────────────────────────────────────────
function I({ size = 16, sw = 1.6, children }: { size?: number; sw?: number; children: React.ReactNode }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth={sw} strokeLinecap="round"
      strokeLinejoin="round" aria-hidden="true">
      {children}
    </svg>
  )
}
const IconSend   = (p: { size?: number; sw?: number }) => <I {...p}><path d="M22 2 11 13M22 2 15 22l-4-9-9-4 20-7Z"/></I>
const IconStop   = (p: { size?: number; sw?: number }) => <I {...p}><rect x="6" y="6" width="12" height="12" rx="2"/></I>
const IconDB     = (p: { size?: number; sw?: number }) => <I {...p}><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.66 4.03 3 9 3s9-1.34 9-3V5M3 12c0 1.66 4.03 3 9 3s9-1.34 9-3"/></I>

// ── Pipeline toggle ────────────────────────────────────────────────────────
function PipelineToggle({ value, onChange, disabled }: {
  value: 'p2' | 'p3'; onChange: (v: 'p2' | 'p3') => void; disabled: boolean
}) {
  const opts: { id: 'p2' | 'p3'; label: string; hint: string }[] = [
    { id: 'p2', label: 'Hybrid', hint: 'Dense + sparse retrieval' },
    { id: 'p3', label: 'Reranked', hint: 'BGE-reranker post-processing' },
  ]
  return (
    <div style={{
      display: 'inline-flex', borderRadius: 6, overflow: 'hidden',
      border: '1px solid var(--rule)', background: 'var(--panel-2)',
    }}>
      {opts.map((o) => (
        <button
          key={o.id}
          title={o.hint}
          disabled={disabled}
          onClick={() => onChange(o.id)}
          style={{
            padding: '5px 12px', border: 'none', fontSize: 11, fontWeight: 600,
            letterSpacing: '0.01em',
            background: value === o.id ? 'var(--panel)' : 'transparent',
            color: value === o.id ? 'var(--ink)' : 'var(--muted)',
            boxShadow: value === o.id ? 'var(--shadow-sm)' : 'none',
            transition: 'all 120ms', cursor: disabled ? 'not-allowed' : 'pointer',
            opacity: disabled ? 0.5 : 1,
          }}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}

// ── QueryInput ────────────────────────────────────────────────────────────
export function QueryInput() {
  const {
    query, setQuery,
    threadId, setThreadId,
    pipeline, setPipeline,
    isStreaming,
  } = useStore()
  const { send, cancel } = useAgentStream()
  const [stats, setStats] = useState<CorpusStats | null>(null)
  const [threads, setThreads] = useState<string[]>([])
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    fetchCorpusStats().then(setStats).catch(() => null)
  }, [])

  useEffect(() => {
    setThreads(loadRecentThreads())
  }, [])

  // Auto-grow textarea
  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`
  }, [query])

  function handleSend() {
    if (!isStreaming && query.trim()) {
      saveThread(threadId)
      setThreads(loadRecentThreads())
      send()
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault()
      handleSend()
    }
  }

  const canSend = !isStreaming && Boolean(query.trim())

  return (
    <div style={{
      borderTop: '1px solid var(--rule)',
      background: 'var(--panel)',
      padding: '14px 20px 16px',
      flexShrink: 0,
    }}>
      {/* Corpus stats bar */}
      {stats && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 6,
          marginBottom: 10, fontSize: 10,
          color: 'var(--faint)',
        }}>
          <IconDB size={10} sw={1.8} />
          <span className="vm-mono">
            {stats.total_chunks.toLocaleString()} chunks ·{' '}
            {stats.pubmed_chunks.toLocaleString()} PubMed ·{' '}
            {stats.pmc_chunks.toLocaleString()} PMC ·{' '}
            {stats.embedding_model}
          </span>
        </div>
      )}

      {/* Main composer row */}
      <div style={{ display: 'flex', gap: 10, alignItems: 'flex-end' }}>
        <div style={{
          flex: 1,
          border: '1px solid var(--rule)',
          borderRadius: 10,
          background: 'var(--canvas)',
          transition: 'box-shadow 160ms',
          display: 'flex', flexDirection: 'column',
        }}
          onFocusCapture={(e) => e.currentTarget.style.boxShadow = '0 0 0 3px var(--accent-soft), var(--shadow-sm)'}
          onBlurCapture={(e) => e.currentTarget.style.boxShadow = ''}
        >
          <textarea
            ref={textareaRef}
            rows={2}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isStreaming}
            placeholder="Ask a medical question… (Ctrl+Enter to send)"
            style={{
              width: '100%', resize: 'none', border: 'none', outline: 'none',
              background: 'transparent',
              fontFamily: 'var(--serif)', fontSize: 16,
              color: 'var(--ink)', letterSpacing: '-0.005em',
              padding: '12px 14px 8px',
              lineHeight: 1.5,
              minHeight: 52, maxHeight: 200,
            }}
          />
          <div style={{
            display: 'flex', alignItems: 'center', gap: 10,
            padding: '6px 10px 8px',
            borderTop: '1px solid var(--rule-soft)',
          }}>
            <PipelineToggle value={pipeline} onChange={setPipeline} disabled={isStreaming} />

            {/* Thread selector */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 5, marginLeft: 4 }}>
              <span className="vm-eyebrow" style={{ fontSize: 8 }}>Thread</span>
              <input
                list="thread-history"
                value={threadId}
                onChange={(e) => setThreadId(e.target.value)}
                disabled={isStreaming}
                style={{
                  fontFamily: 'var(--mono)', fontSize: 10,
                  color: 'var(--ink-soft)', background: 'transparent',
                  border: '1px solid var(--rule-soft)', borderRadius: 4,
                  padding: '3px 7px', outline: 'none', width: 160,
                }}
              />
              {threads.length > 0 && (
                <datalist id="thread-history">
                  {threads.map((t) => <option key={t} value={t} />)}
                </datalist>
              )}
            </div>

            <div style={{ flex: 1 }} />
            <span className="vm-mono" style={{ fontSize: 9, color: 'var(--faint)' }}>⌃↵ send</span>
          </div>
        </div>

        {/* Send / Stop */}
        <button
          onClick={isStreaming ? cancel : handleSend}
          disabled={!isStreaming && !canSend}
          style={{
            width: 44, height: 44, borderRadius: 10, border: 'none',
            display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
            background: isStreaming ? 'var(--error)' : canSend ? 'var(--ink)' : 'var(--rule)',
            color: isStreaming || canSend ? 'var(--canvas)' : 'var(--faint)',
            transition: 'background 160ms',
            cursor: !isStreaming && !canSend ? 'not-allowed' : 'pointer',
          }}
        >
          {isStreaming ? <IconStop size={16} sw={2} /> : <IconSend size={15} sw={2} />}
        </button>
      </div>
    </div>
  )
}
