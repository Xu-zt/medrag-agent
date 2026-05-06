import axios from 'axios'
import type { ChunkContextResponse, CorpusStats, DocumentResponse, SearchResponse } from '../types'

const BASE = ''  // proxied through Vite to http://localhost:8000

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

// ── WebSocket URL helper ──────────────────────────────────────────────────

export function wsAskUrl(): string {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  // During dev the Vite proxy forwards ws:// to the backend
  return `${proto}//${window.location.host}/api/ask`
}
