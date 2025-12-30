import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Sidebar from './components/layout/Sidebar'
import ThreeBackground from './components/three/ThreeBackground'
import Dashboard from './pages/Dashboard'
import Jobs from './pages/Jobs'
import Candidates from './pages/Candidates'
import './index.css'

function App() {
  return (
    <BrowserRouter>
      <div className="app-layout">
        <ThreeBackground />
        <Sidebar />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/jobs" element={<Jobs />} />
            <Route path="/candidates" element={<Candidates />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}

export default App
