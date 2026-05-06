// ── Data models mirroring the Python Pydantic models ──────────────────────

export interface ChunkOut {
  chunk_id: string
  citation: string
  source: 'pubmed' | 'pmc'
  doc_id: string
  title: string
  section: string | null
  pmid: string | null
  chunk_idx: number
  total_chunks: number
  text: string
  score: number | null
  highlight_ranges: [number, number][]
  external_url: string
}

export type AgentEventType =
  | 'node_start'
  | 'node_end'
  | 'chunk_retrieved'
  | 'answer_token'
  | 'done'
  | 'error'

export interface AgentEvent {
  event: AgentEventType
  node: string | null
  data: Record<string, unknown>
}

export interface AnswerOut {
  answer: string
  citations: string[]
  confidence: number
  faithful: boolean
  faithfulness_issues: string
  iterations: number
  regen_count: number
  rewritten_queries: string[]
  chunks: ChunkOut[]
  thread_id: string
  latency_ms: number
}

// ── Timeline node status ──────────────────────────────────────────────────

export type NodeStatus = 'waiting' | 'running' | 'done' | 'rewrite' | 'error'

export interface TimelineNode {
  name: string
  label: string
  status: NodeStatus
  summary: string
  detail?: Record<string, unknown>
  timestamp?: number
}

// ── Corpus stats ──────────────────────────────────────────────────────────

export interface CorpusStats {
  total_chunks: number
  pubmed_chunks: number
  pmc_chunks: number
  collection: string
  embedding_model: string
}

// ── Search response ───────────────────────────────────────────────────────

export interface SearchResponse {
  query: string
  pipeline: string
  latency_ms: number
  chunks: ChunkOut[]
}

// ── Document response ─────────────────────────────────────────────────────

export interface DocumentChunkSlim {
  chunk_id: string
  chunk_idx: number
  section: string | null
  text: string
}

export interface DocumentResponse {
  citation: string
  source: string
  doc_id: string
  title: string
  pmid: string | null
  external_url: string
  total_chunks: number
  chunks: DocumentChunkSlim[]
}

// ── Chunk context ─────────────────────────────────────────────────────────

export interface ChunkSlim {
  chunk_id: string
  text: string
  score: number | null
}

export interface ChunkContextResponse {
  chunk: ChunkSlim
  prev_chunk: ChunkSlim | null
  next_chunk: ChunkSlim | null
  document: { title: string; citation: string; external_url: string }
}
