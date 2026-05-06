import { create } from 'zustand'
import type { AnswerOut, ChunkOut, TimelineNode } from '../types'

// ── Chunk colours (cycle through 8 distinct hues) ────────────────────────
const CHUNK_COLORS = [
  'blue', 'emerald', 'violet', 'amber', 'rose', 'cyan', 'fuchsia', 'lime',
] as const

export type ChunkColor = typeof CHUNK_COLORS[number]

export function chunkColor(idx: number): ChunkColor {
  return CHUNK_COLORS[idx % CHUNK_COLORS.length]
}

// ── Store ─────────────────────────────────────────────────────────────────

export interface AppState {
  // Session
  threadId: string
  pipeline: 'p2' | 'p3'
  setThreadId: (id: string) => void
  setPipeline: (p: 'p2' | 'p3') => void

  // Query
  query: string
  setQuery: (q: string) => void

  // Streaming state
  isStreaming: boolean
  setStreaming: (v: boolean) => void

  // Timeline
  timeline: TimelineNode[]
  setTimeline: (nodes: TimelineNode[]) => void
  updateNode: (name: string, patch: Partial<TimelineNode>) => void
  pushNode: (node: TimelineNode) => void

  // Live chunks arriving during retrieval
  liveChunks: ChunkOut[]
  pushLiveChunk: (chunk: ChunkOut) => void
  clearLiveChunks: () => void

  // Answer streaming
  streamingAnswer: string
  appendAnswerToken: (token: string) => void
  clearStreamingAnswer: () => void

  // Final result
  result: AnswerOut | null
  setResult: (r: AnswerOut | null) => void

  // Selected chunk (for EvidencePanel highlight)
  selectedChunkId: string | null
  setSelectedChunkId: (id: string | null) => void
}

export const useStore = create<AppState>((set) => ({
  threadId: `session-${Date.now()}`,
  pipeline: 'p2',
  setThreadId: (id) => set({ threadId: id }),
  setPipeline: (p) => set({ pipeline: p }),

  query: '',
  setQuery: (q) => set({ query: q }),

  isStreaming: false,
  setStreaming: (v) => set({ isStreaming: v }),

  timeline: [],
  setTimeline: (nodes) => set({ timeline: nodes }),
  updateNode: (name, patch) =>
    set((s) => ({
      timeline: s.timeline.map((n) => (n.name === name ? { ...n, ...patch } : n)),
    })),
  pushNode: (node) => set((s) => ({ timeline: [...s.timeline, node] })),

  liveChunks: [],
  pushLiveChunk: (chunk) => set((s) => ({ liveChunks: [...s.liveChunks, chunk] })),
  clearLiveChunks: () => set({ liveChunks: [] }),

  streamingAnswer: '',
  appendAnswerToken: (token) =>
    set((s) => ({ streamingAnswer: s.streamingAnswer + token })),
  clearStreamingAnswer: () => set({ streamingAnswer: '' }),

  result: null,
  setResult: (r) => set({ result: r }),

  selectedChunkId: null,
  setSelectedChunkId: (id) => set({ selectedChunkId: id }),
}))
