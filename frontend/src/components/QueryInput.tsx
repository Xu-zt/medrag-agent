import { type KeyboardEvent, useEffect, useState } from 'react'
import clsx from 'clsx'
import { Send, Square, Database } from 'lucide-react'
import { useStore } from '../store'
import { useAgentStream } from '../hooks/useAgentStream'
import { fetchCorpusStats } from '../api/client'
import type { CorpusStats } from '../types'

export function QueryInput() {
  const {
    query, setQuery,
    threadId, setThreadId,
    pipeline, setPipeline,
    isStreaming,
  } = useStore()
  const { send, cancel } = useAgentStream()
  const [stats, setStats] = useState<CorpusStats | null>(null)

  useEffect(() => {
    fetchCorpusStats().then(setStats).catch(() => null)
  }, [])

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault()
      if (!isStreaming && query.trim()) send()
    }
  }

  return (
    <div className="border-t border-slate-200 bg-white p-4">
      {/* Stats bar */}
      {stats && (
        <div className="flex items-center gap-2 text-xs text-slate-400 mb-3">
          <Database size={11} />
          <span>
            {stats.total_chunks.toLocaleString()} chunks ·{' '}
            {stats.pubmed_chunks.toLocaleString()} PubMed ·{' '}
            {stats.pmc_chunks.toLocaleString()} PMC ·{' '}
            {stats.embedding_model}
          </span>
        </div>
      )}

      <div className="flex gap-2">
        {/* Textarea */}
        <textarea
          className="flex-1 resize-none rounded-xl border border-slate-200 px-4 py-3 text-sm
                     focus:outline-none focus:ring-2 focus:ring-blue-300 transition-shadow
                     disabled:opacity-50 disabled:cursor-not-allowed"
          rows={2}
          placeholder="Ask a medical question… (Ctrl+Enter to send)"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isStreaming}
        />

        {/* Controls */}
        <div className="flex flex-col gap-2">
          {/* Pipeline selector */}
          <select
            className="rounded-lg border border-slate-200 px-2 py-1.5 text-xs text-slate-600
                       focus:outline-none focus:ring-1 focus:ring-blue-300"
            value={pipeline}
            onChange={(e) => setPipeline(e.target.value as 'p2' | 'p3')}
            disabled={isStreaming}
          >
            <option value="p2">P2 – Hybrid (fast)</option>
            <option value="p3">P3 – Reranked</option>
          </select>

          {/* Thread ID */}
          <input
            className="rounded-lg border border-slate-200 px-2 py-1.5 text-xs text-slate-600
                       focus:outline-none focus:ring-1 focus:ring-blue-300 w-full"
            placeholder="Session ID"
            value={threadId}
            onChange={(e) => setThreadId(e.target.value)}
            disabled={isStreaming}
          />

          {/* Send / Cancel */}
          <button
            onClick={isStreaming ? cancel : send}
            disabled={!isStreaming && !query.trim()}
            className={clsx(
              'flex items-center justify-center gap-1.5 rounded-xl px-4 py-2 text-sm font-semibold transition-colors',
              isStreaming
                ? 'bg-rose-500 hover:bg-rose-600 text-white'
                : 'bg-blue-600 hover:bg-blue-700 text-white disabled:opacity-40 disabled:cursor-not-allowed',
            )}
          >
            {isStreaming ? (
              <><Square size={14} /> Stop</>
            ) : (
              <><Send size={14} /> Send</>
            )}
          </button>
        </div>
      </div>
    </div>
  )
}
