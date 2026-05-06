import { useState } from 'react'
import clsx from 'clsx'
import { Search, ExternalLink } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { fetchSearch } from '../api/client'
import type { ChunkOut, SearchResponse } from '../types'

function ResultCard({ chunk, onClick }: { chunk: ChunkOut; onClick: () => void }) {
  const pct = chunk.score ? Math.round(chunk.score * 100) : 0
  const barColor =
    pct >= 70 ? 'bg-emerald-400' : pct >= 40 ? 'bg-amber-400' : 'bg-rose-400'

  return (
    <div
      className="border border-slate-200 rounded-xl p-4 hover:border-blue-300 hover:shadow-sm
                 cursor-pointer transition-all bg-white"
      onClick={onClick}
    >
      <div className="flex items-start justify-between gap-4 mb-2">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs font-semibold text-slate-600">{chunk.citation}</span>
            <span
              className={clsx(
                'text-xs px-1.5 py-0.5 rounded-full',
                chunk.source === 'pubmed'
                  ? 'bg-blue-100 text-blue-700'
                  : 'bg-purple-100 text-purple-700',
              )}
            >
              {chunk.source === 'pubmed' ? 'PubMed' : 'PMC'}
            </span>
            {chunk.section && (
              <span className="text-xs text-slate-400">{chunk.section}</span>
            )}
          </div>
          <p className="text-sm font-medium text-slate-700 line-clamp-1">{chunk.title}</p>
        </div>
        <div className="flex-shrink-0 text-right">
          <div className="text-xs font-mono text-slate-500 mb-1">{(chunk.score ?? 0).toFixed(3)}</div>
          <div className="w-16 h-1.5 bg-slate-100 rounded-full overflow-hidden">
            <div className={clsx('h-full rounded-full', barColor)} style={{ width: `${pct}%` }} />
          </div>
        </div>
      </div>
      <p className="text-xs text-slate-500 line-clamp-2">{chunk.text}</p>
      {chunk.external_url && (
        <a
          href={chunk.external_url}
          target="_blank"
          rel="noopener noreferrer"
          className="mt-2 inline-flex items-center gap-1 text-xs text-blue-500 hover:text-blue-700"
          onClick={(e) => e.stopPropagation()}
        >
          <ExternalLink size={11} />
          Open in {chunk.source === 'pubmed' ? 'PubMed' : 'PMC'}
        </a>
      )}
    </div>
  )
}

export function ExplorerPage() {
  const [q, setQ] = useState('')
  const [k, setK] = useState(10)
  const [pipeline, setPipeline] = useState<'p2' | 'p3'>('p2')
  const [results, setResults] = useState<SearchResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const navigate = useNavigate()

  async function handleSearch() {
    if (!q.trim()) return
    setLoading(true)
    try {
      const data = await fetchSearch(q, k, pipeline)
      setResults(data)
    } finally {
      setLoading(false)
    }
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === 'Enter') handleSearch()
  }

  return (
    <div className="flex h-full overflow-hidden">
      {/* Sidebar filters */}
      <div className="w-56 flex-shrink-0 border-r border-slate-200 p-4 space-y-4">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-400">Filters</h2>

        <div>
          <label className="text-xs font-medium text-slate-600 block mb-1">Results</label>
          <input
            type="number"
            min={1}
            max={20}
            value={k}
            onChange={(e) => setK(Number(e.target.value))}
            className="w-full rounded-lg border border-slate-200 px-3 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-blue-300"
          />
        </div>

        <div>
          <label className="text-xs font-medium text-slate-600 block mb-1">Pipeline</label>
          <select
            value={pipeline}
            onChange={(e) => setPipeline(e.target.value as 'p2' | 'p3')}
            className="w-full rounded-lg border border-slate-200 px-3 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-blue-300"
          >
            <option value="p2">P2 – Hybrid (fast)</option>
            <option value="p3">P3 – Reranked</option>
          </select>
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Search bar */}
        <div className="p-4 border-b border-slate-200 flex gap-2">
          <div className="relative flex-1">
            <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
            <input
              className="w-full rounded-xl border border-slate-200 pl-9 pr-4 py-2.5 text-sm
                         focus:outline-none focus:ring-2 focus:ring-blue-300"
              placeholder="Search literature…"
              value={q}
              onChange={(e) => setQ(e.target.value)}
              onKeyDown={handleKeyDown}
            />
          </div>
          <button
            onClick={handleSearch}
            disabled={loading || !q.trim()}
            className="rounded-xl bg-blue-600 hover:bg-blue-700 text-white px-5 py-2.5 text-sm
                       font-semibold transition-colors disabled:opacity-40"
          >
            {loading ? 'Searching…' : 'Search'}
          </button>
        </div>

        {/* Results */}
        <div className="flex-1 overflow-y-auto p-4">
          {results && (
            <>
              <div className="flex items-center justify-between mb-3">
                <p className="text-xs text-slate-400">
                  {results.chunks.length} results · {results.latency_ms.toFixed(0)}ms
                </p>
              </div>
              <div className="space-y-3">
                {results.chunks.map((c) => (
                  <ResultCard
                    key={c.chunk_id}
                    chunk={c}
                    onClick={() => navigate(`/document/${encodeURIComponent(c.citation)}`)}
                  />
                ))}
              </div>
            </>
          )}

          {!results && !loading && (
            <div className="flex flex-col items-center justify-center h-full text-slate-400">
              <Search size={32} className="mb-3 opacity-30" />
              <p className="text-sm">Search the corpus directly</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
