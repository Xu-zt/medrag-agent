import { useCallback, useRef } from 'react'
import { wsAskUrl } from '../api/client'
import { useStore } from '../store'
import type { ChunkOut } from '../types'
import type { AgentEvent, NodeEndData } from '../types/ws'

// Node display labels
const NODE_LABELS: Record<string, string> = {
  route: 'Route',
  retrieve: 'Retrieve',
  rerank: 'Rerank',
  grade: 'Grade',
  rewrite: 'Rewrite',
  generate: 'Generate',
  check: 'Faithfulness Check',
  increment_regen: 'Regen Counter',
  summarize_gate: 'Summarize Gate',
  summarize: 'Summarize',
}

export function useAgentStream() {
  const ws = useRef<WebSocket | null>(null)
  const {
    threadId,
    pipeline,
    query,
    setStreaming,
    setTimeline,
    updateNode,
    pushNode,
    pushLiveChunk,
    clearLiveChunks,
    setResult,
    setSelectedChunkId,
    setErrorMessage,
  } = useStore()

  const send = useCallback(() => {
    if (ws.current) {
      ws.current.close()
    }

    setTimeline([])
    clearLiveChunks()
    setResult(null)
    setSelectedChunkId(null)
    setErrorMessage(null)
    setStreaming(true)

    const socket = new WebSocket(wsAskUrl())
    ws.current = socket

    socket.onopen = () => {
      socket.send(JSON.stringify({ query, thread_id: threadId, pipeline }))
    }

    socket.onmessage = (msg: MessageEvent) => {
      let ev: AgentEvent
      try {
        ev = JSON.parse(msg.data as string) as AgentEvent
      } catch {
        return
      }
      handleEvent(ev)
    }

    socket.onerror = (e) => {
      console.error('WebSocket error', e)
      setErrorMessage('WebSocket connection error — is the backend running?')
      setStreaming(false)
    }

    socket.onclose = (e) => {
      if (e.code !== 1000 && e.code !== 1001) {
        console.warn('WebSocket closed unexpectedly', e.code, e.reason)
      }
      setStreaming(false)
    }

    function handleEvent(ev: AgentEvent) {
      if (ev.event === 'node_start') {
        pushNode({
          name: ev.node,
          label: NODE_LABELS[ev.node] ?? ev.node,
          status: 'running',
          summary: '',
          timestamp: Date.now(),
        })
      }

      if (ev.event === 'node_end') {
        const d: NodeEndData = ev.data
        let summary = ''
        if (ev.node === 'retrieve' || ev.node === 'rerank') {
          summary = `${d.count ?? 0} candidates`
        } else if (ev.node === 'grade') {
          const score = d.relevance_score ?? 0
          summary = `score: ${score.toFixed(2)} · ${(d.relevant ?? score >= 0.6) ? 'relevant' : 'insufficient'}`
        } else if (ev.node === 'rewrite') {
          const q = d.new_query ?? ''
          summary = q.slice(0, 60) + (q.length > 60 ? '…' : '')
        } else if (ev.node === 'generate') {
          summary = 'answer generated'
        } else if (ev.node === 'check') {
          summary = (d.faithful ?? false) ? 'faithful ✓' : 'issues found ⚠'
        }

        updateNode(ev.node, {
          status: ev.node === 'rewrite' ? 'rewrite' : 'done',
          summary,
          detail: d,
        })
      }

      if (ev.event === 'chunk_retrieved') {
        const d = ev.data
        const chunk: ChunkOut = {
          chunk_id: d.chunk_id,
          citation: d.citation,
          source: d.source as 'pubmed' | 'pmc',
          doc_id: d.citation.replace(/^(PMID|PMC):/, ''),
          title: d.title,
          section: null,
          pmid: d.citation.startsWith('PMID:') ? d.citation.slice(5) : null,
          chunk_idx: 0,
          total_chunks: 1,
          text: d.text_snippet,
          score: d.score ?? null,
          highlight_ranges: [],
          external_url: d.external_url,
        }
        pushLiveChunk(chunk)
      }

      if (ev.event === 'done') {
        setResult(ev.data)
        setStreaming(false)
      }

      if (ev.event === 'error') {
        console.error('Agent error event:', ev.data.message)
        setErrorMessage(`Server error: ${ev.data.message}`)
        setStreaming(false)
      }
    }
  }, [
    query, threadId, pipeline,
    setStreaming, setTimeline, updateNode, pushNode,
    pushLiveChunk, clearLiveChunks,
    setResult, setSelectedChunkId, setErrorMessage,
  ])

  const cancel = useCallback(() => {
    ws.current?.close()
    setStreaming(false)
  }, [setStreaming])

  return { send, cancel }
}
