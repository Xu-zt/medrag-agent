import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom'
import clsx from 'clsx'
import { BookOpen, Search, Activity } from 'lucide-react'
import { AnswerPage } from './pages/AnswerPage'
import { ExplorerPage } from './pages/ExplorerPage'
import { DocumentPage } from './pages/DocumentPage'

function Header() {
  const navCls = ({ isActive }: { isActive: boolean }) =>
    clsx(
      'flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors',
      isActive
        ? 'bg-white text-slate-800 shadow-sm'
        : 'text-slate-400 hover:text-slate-600',
    )

  return (
    <header className="h-14 border-b border-slate-200 bg-slate-50 flex items-center px-6 gap-6 flex-shrink-0">
      {/* Logo */}
      <div className="flex items-center gap-2">
        <Activity size={20} className="text-blue-600" />
        <span className="font-bold text-slate-800 text-base">VeritasMed</span>
        <span className="text-xs text-slate-400 font-normal hidden sm:block">
          Self-Verifying Medical QA
        </span>
      </div>

      {/* Nav */}
      <nav className="flex items-center gap-1 ml-4 bg-slate-100 rounded-xl p-1">
        <NavLink to="/" end className={navCls}>
          <BookOpen size={14} />
          Ask
        </NavLink>
        <NavLink to="/explore" className={navCls}>
          <Search size={14} />
          Explore
        </NavLink>
      </nav>
    </header>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <div className="h-screen flex flex-col bg-slate-50 overflow-hidden">
        <Header />
        <main className="flex-1 overflow-hidden">
          <Routes>
            <Route path="/" element={<AnswerPage />} />
            <Route path="/explore" element={<ExplorerPage />} />
            <Route path="/document/:citation" element={<DocumentPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}
