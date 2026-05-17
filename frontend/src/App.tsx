import React, { useEffect, useState } from 'react'
import { BrowserRouter, Routes, Route, useNavigate, useLocation } from 'react-router-dom'
import { AnswerPage } from './pages/AnswerPage'
import { ExplorerPage } from './pages/ExplorerPage'
import { DocumentPage } from './pages/DocumentPage'
import { fetchHealth } from './api/client'

// ── SVG helpers ────────────────────────────────────────────────────────────
function I({ size = 16, sw = 1.6, children, style }: {
  size?: number; sw?: number; children: React.ReactNode; style?: React.CSSProperties
}) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth={sw} strokeLinecap="round"
      strokeLinejoin="round" aria-hidden="true" style={style}>
      {children}
    </svg>
  )
}
const IconBook    = (p: { size?: number; sw?: number }) =>
  <I {...p}><path d="M4 4.5A2.5 2.5 0 0 1 6.5 2H20v17H6.5A2.5 2.5 0 0 0 4 21.5v-17Z"/><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/></I>
const IconSearch  = (p: { size?: number; sw?: number }) =>
  <I {...p}><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></I>
const IconSun     = (p: { size?: number; sw?: number }) =>
  <I {...p}><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"/></I>
const IconMoon    = (p: { size?: number; sw?: number }) =>
  <I {...p}><path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/></I>
const IconClinical = (p: { size?: number; sw?: number }) =>
  <I {...p}><path d="M12 5v14M5 12h14"/><rect x="3" y="3" width="18" height="18" rx="2"/></I>

// ── BrandMark ──────────────────────────────────────────────────────────────
function BrandMark() {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 9, userSelect: 'none' }}>
      <div style={{
        width: 28, height: 28, borderRadius: 6,
        background: 'var(--ink)', display: 'flex', alignItems: 'center', justifyContent: 'center',
        position: 'relative', flexShrink: 0,
      }}>
        <span style={{
          fontFamily: 'var(--serif)', fontStyle: 'italic', fontSize: 18,
          color: 'var(--canvas)', lineHeight: 1, letterSpacing: '-0.02em',
        }}>V</span>
        <div style={{
          position: 'absolute', bottom: 3, left: 4, right: 4, height: 2,
          background: 'var(--accent)', borderRadius: 1,
        }} />
      </div>
      <span style={{
        fontFamily: 'var(--serif)', fontSize: 17, letterSpacing: '-0.02em',
        color: 'var(--ink)', lineHeight: 1,
      }}>
        VeritasMed
      </span>
    </div>
  )
}

// ── NavTab ─────────────────────────────────────────────────────────────────
function NavTab({ to, label, icon, active }: {
  to: string; label: string; icon: React.ReactNode; active: boolean
}) {
  const navigate = useNavigate()
  const [hovered, setHovered] = useState(false)

  return (
    <button
      onClick={() => navigate(to)}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        display: 'flex', alignItems: 'center', gap: 6,
        padding: '6px 12px', borderRadius: 6, border: 'none',
        background: active ? 'var(--panel)' : hovered ? 'var(--panel-2)' : 'transparent',
        color: active ? 'var(--ink)' : 'var(--muted)',
        fontSize: 13, fontWeight: active ? 600 : 500,
        letterSpacing: '-0.005em',
        boxShadow: active ? 'var(--shadow-sm)' : 'none',
        transition: 'all 120ms',
        position: 'relative',
      }}
    >
      {icon}
      {label}
      {active && (
        <div style={{
          position: 'absolute', bottom: -1, left: 8, right: 8, height: 2,
          background: 'var(--accent)', borderRadius: 1,
        }} />
      )}
    </button>
  )
}

// ── ThemeToggle ────────────────────────────────────────────────────────────
type Theme = 'paper' | 'midnight' | 'clinical'

function ThemeToggle({ theme, setTheme }: { theme: Theme; setTheme: (t: Theme) => void }) {
  const themes: { id: Theme; icon: React.ReactNode; label: string }[] = [
    { id: 'paper',    icon: <IconSun size={13} sw={2} />,      label: 'Paper' },
    { id: 'midnight', icon: <IconMoon size={13} sw={2} />,     label: 'Midnight' },
    { id: 'clinical', icon: <IconClinical size={13} sw={2} />, label: 'Clinical' },
  ]
  return (
    <div style={{
      display: 'flex', gap: 2,
      background: 'var(--panel-2)', borderRadius: 7, padding: 3,
      border: '1px solid var(--rule-soft)',
    }}>
      {themes.map((t) => (
        <button
          key={t.id}
          title={t.label}
          onClick={() => setTheme(t.id)}
          style={{
            padding: '4px 8px', borderRadius: 4, border: 'none',
            background: theme === t.id ? 'var(--panel)' : 'transparent',
            color: theme === t.id ? 'var(--ink)' : 'var(--faint)',
            boxShadow: theme === t.id ? 'var(--shadow-sm)' : 'none',
            transition: 'all 120ms',
            display: 'flex', alignItems: 'center',
          }}
        >
          {t.icon}
        </button>
      ))}
    </div>
  )
}

// ── StatusPill ─────────────────────────────────────────────────────────────
type HealthStatus = 'online' | 'degraded' | 'offline' | 'checking'

function StatusPill({ status }: { status: HealthStatus }) {
  const colors: Record<HealthStatus, string> = {
    online:   'var(--verified)',
    degraded: 'var(--warn)',
    offline:  'var(--error)',
    checking: 'var(--faint)',
  }
  const labels: Record<HealthStatus, string> = {
    online:   'Online',
    degraded: 'Degraded',
    offline:  'Offline',
    checking: 'Checking…',
  }
  const color = colors[status]

  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 6,
      padding: '4px 10px', borderRadius: 20,
      border: '1px solid var(--rule-soft)',
      background: 'var(--panel-2)',
    }}>
      <div style={{ position: 'relative', width: 7, height: 7 }}>
        <div style={{
          width: 7, height: 7, borderRadius: '50%',
          background: color,
        }} />
        {status === 'online' && (
          <div style={{
            position: 'absolute', inset: 0,
            borderRadius: '50%', background: color,
            animation: 'vmPing 1.8s ease-out infinite',
          }} />
        )}
      </div>
      <span className="vm-mono" style={{ fontSize: 10, color: 'var(--ink-soft)', fontWeight: 600 }}>
        {labels[status]}
      </span>
    </div>
  )
}

// ── Header ─────────────────────────────────────────────────────────────────
function Header({ theme, setTheme }: { theme: Theme; setTheme: (t: Theme) => void }) {
  const location = useLocation()
  const isAsk     = location.pathname === '/'
  const isExplore = location.pathname === '/explore'

  const [serverStatus, setServerStatus] = useState<HealthStatus>('checking')

  useEffect(() => {
    let cancelled = false
    async function poll() {
      try {
        const h = await fetchHealth()
        if (!cancelled) {
          setServerStatus(h.status === 'ok' ? 'online' : 'degraded')
        }
      } catch {
        if (!cancelled) setServerStatus('offline')
      }
    }
    poll()
    const id = setInterval(poll, 30_000)
    return () => { cancelled = true; clearInterval(id) }
  }, [])

  return (
    <header style={{
      height: 52, flexShrink: 0,
      borderBottom: '1px solid var(--rule)',
      background: 'var(--panel)',
      display: 'flex', alignItems: 'center',
      padding: '0 20px', gap: 20,
    }}>
      <BrandMark />

      <nav style={{ display: 'flex', gap: 4, marginLeft: 12 }}>
        <NavTab to="/"        label="Ask"     icon={<IconBook   size={13} sw={2} />} active={isAsk} />
        <NavTab to="/explore" label="Explore" icon={<IconSearch size={13} sw={2} />} active={isExplore} />
      </nav>

      <div style={{ flex: 1 }} />
      <StatusPill status={serverStatus} />
      <ThemeToggle theme={theme} setTheme={setTheme} />
    </header>
  )
}

// ── App ────────────────────────────────────────────────────────────────────
export default function App() {
  const [theme, setTheme] = useState<Theme>('paper')

  useEffect(() => {
    if (theme === 'paper') {
      document.documentElement.removeAttribute('data-theme')
    } else {
      document.documentElement.setAttribute('data-theme', theme)
    }
  }, [theme])

  return (
    <BrowserRouter>
      <div style={{ height: '100vh', display: 'flex', flexDirection: 'column', overflow: 'hidden', background: 'var(--canvas)' }}>
        <Header theme={theme} setTheme={setTheme} />
        <main style={{ flex: 1, overflow: 'hidden' }}>
          <Routes>
            <Route path="/"                       element={<AnswerPage />} />
            <Route path="/explore"                element={<ExplorerPage />} />
            <Route path="/document/:citation"     element={<DocumentPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}
