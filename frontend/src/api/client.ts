import axios from 'axios'
import type { ChunkContextResponse, CorpusStats, DocumentResponse, SearchResponse } from '../types'

// In dev: set VITE_API_URL=http://localhost:8000 in frontend/.env.local
// In production: leave unset — frontend and API share the same origin
const BASE = (import.meta.env.VITE_API_URL as string) ?? ''

export const api = axios.create({ baseURL: BASE })

export async function fetchSearch(
  q: string,
  k = 5,
  pipeline: 'p2' | 'p3' = 'p2',
  highlight = true,
): Promise<SearchResponse> {
  const r = await api.get('/api/search', { params: { q, k, pipeline, highlight } })
  return r.data
}

export async function fetchDocument(citation: string): Promise<DocumentResponse> {
  const r = await api.get(`/api/document/${encodeURIComponent(citation)}`)
  return r.data
}

export async function fetchChunk(
  chunkId: string,
  contextWindow = 1,
): Promise<ChunkContextResponse> {
  const r = await api.get(`/api/chunk/${encodeURIComponent(chunkId)}`, {
    params: { context_window: contextWindow },
  })
  return r.data
}

export async function fetchCorpusStats(): Promise<CorpusStats> {
  const r = await api.get('/api/corpus/stats')
  return r.data
}

export async function fetchHealth(): Promise<{ status: string; qdrant: string; llm: string }> {
  const r = await api.get('/api/health')
  return r.data
}

export function loadRecentThreads(): string[] {
  try {
    return JSON.parse(localStorage.getItem('vm_threads') ?? '[]') as string[]
  } catch { return [] }
}

export function saveThread(threadId: string): void {
  try {
    const existing = loadRecentThreads().filter((t) => t !== threadId)
    localStorage.setItem('vm_threads', JSON.stringify([threadId, ...existing].slice(0, 20)))
  } catch { /* ignore */ }
}

// ── WebSocket URL helper ──────────────────────────────────────────────────

export function wsAskUrl(): string {
  const apiBase = (import.meta.env.VITE_API_URL as string) ?? ''
  if (apiBase) {
    // Convert http(s):// to ws(s)://
    return apiBase.replace(/^http/, 'ws') + '/api/ask'
  }
  // Production: derive from current page origin
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${proto}//${window.location.host}/api/ask`
}
