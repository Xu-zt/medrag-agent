import { useCallback, useRef } from 'react'
import { wsAskUrl } from '../api/client'
import { useStore } from '../store'
import type { AgentEvent, AnswerOut, ChunkOut } from '../types'

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
    appendAnswerToken,
    clearStreamingAnswer,
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
    clearStreamingAnswer()
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
      setErrorMessage('WebSocket connection error — is the backend running on port 8000?')
      setStreaming(false)
    }

    socket.onclose = (e) => {
      if (e.code !== 1000 && e.code !== 1001) {
        console.warn('WebSocket closed unexpectedly', e.code, e.reason)
      }
      setStreaming(false)
    }

    function handleEvent(ev: AgentEvent) {
      const { event, node, data } = ev

      if (event === 'node_start' && node) {
        pushNode({
          name: node,
          label: NODE_LABELS[node] ?? node,
          status: 'running',
          summary: '',
          timestamp: Date.now(),
        })
      }

      if (event === 'node_end' && node) {
        let summary = ''
        if (node === 'retrieve') {
          summary = `${(data as { count?: number }).count ?? 0} candidates`
        } else if (node === 'grade') {
          const score = (data as { relevance_score?: number }).relevance_score ?? 0
          const rel = score >= 0.6
          summary = `score: ${score.toFixed(2)} · ${rel ? 'relevant' : 'insufficient'}`
        } else if (node === 'rewrite') {
          const q = (data as { new_query?: string }).new_query ?? ''
          summary = q.slice(0, 60) + (q.length > 60 ? '…' : '')
        } else if (node === 'generate') {
          summary = 'answer generated'
        } else if (node === 'check') {
          const faithful = (data as { faithful?: boolean }).faithful ?? false
          summary = faithful ? 'faithful ✓' : 'issues found ⚠'
        }

        const isRewrite = node === 'rewrite'
        updateNode(node, {
          status: isRewrite ? 'rewrite' : 'done',
          summary,
          detail: data as Record<string, unknown>,
        })
      }

      if (event === 'chunk_retrieved') {
        const d = data as {
          chunk_id?: string
          citation?: string
          title?: string
          score?: number
          text_snippet?: string
          source?: string
          external_url?: string
        }
        const chunk: ChunkOut = {
          chunk_id: d.chunk_id ?? '',
          citation: d.citation ?? '',
          source: (d.source ?? 'pubmed') as 'pubmed' | 'pmc',
          doc_id: d.citation?.replace(/^(PMID|PMC):/, '') ?? '',
          title: d.title ?? '',
          section: null,
          pmid: d.citation?.startsWith('PMID:') ? d.citation.slice(5) : null,
          chunk_idx: 0,
          total_chunks: 1,
          text: d.text_snippet ?? '',
          score: d.score ?? null,
          highlight_ranges: [],
          external_url: d.external_url ?? '',
        }
        pushLiveChunk(chunk)
      }

      if (event === 'answer_token') {
        const token = (data as { token?: string }).token ?? ''
        appendAnswerToken(token)
      }

      if (event === 'done') {
        const answer = data as unknown as AnswerOut
        setResult(answer)
        setStreaming(false)
      }

      if (event === 'error') {
        const msg = (data as { message?: string }).message ?? 'Unknown server error'
        console.error('Agent error event:', msg)
        setErrorMessage(`Server error: ${msg}`)
        setStreaming(false)
      }
    }
  }, [
    query, threadId, pipeline,
    setStreaming, setTimeline, updateNode, pushNode,
    pushLiveChunk, clearLiveChunks,
    appendAnswerToken, clearStreamingAnswer,
    setResult, setSelectedChunkId, setErrorMessage,
  ])

  const cancel = useCallback(() => {
    ws.current?.close()
    setStreaming(false)
  }, [setStreaming])

  return { send, cancel }
}
