import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
    Search,
    User,
    ExternalLink,
    Clock
} from 'lucide-react'
import { getCandidates } from '../services/api'
import './Candidates.css'

export default function Candidates() {
    const [candidates, setCandidates] = useState([])
    const [loading, setLoading] = useState(true)
    const [searchQuery, setSearchQuery] = useState('')

    useEffect(() => {
        loadCandidates()
    }, [])

    async function loadCandidates() {
        try {
            const data = await getCandidates()
            setCandidates(data.candidates || [])
        } catch (error) {
            console.error('Failed to load candidates:', error)
        } finally {
            setLoading(false)
        }
    }

    const filteredCandidates = candidates.filter(c =>
        c.name?.toLowerCase().includes(searchQuery.toLowerCase()) ||
        c.current_title?.toLowerCase().includes(searchQuery.toLowerCase()) ||
        c.current_company?.toLowerCase().includes(searchQuery.toLowerCase())
    )

    return (
        <div className="candidates-page">
            <motion.div
                className="page-header"
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
            >
                <div>
                    <h1>Candidates</h1>
                    <p>All scraped LinkedIn profiles and their scoring history.</p>
                </div>
            </motion.div>

            <motion.div
                className="search-bar"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
            >
                <Search size={20} className="search-icon" />
                <input
                    type="text"
                    className="input search-input"
                    placeholder="Search by name, title, or company..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                />
            </motion.div>

            <div className="candidates-list">
                {loading ? (
                    Array(4).fill(0).map((_, i) => (
                        <div key={i} className="candidate-row skeleton-row">
                            <div className="skeleton" style={{ width: 48, height: 48, borderRadius: '50%' }} />
                            <div style={{ flex: 1 }}>
                                <div className="skeleton" style={{ height: 20, width: '30%', marginBottom: 8 }} />
                                <div className="skeleton" style={{ height: 16, width: '50%' }} />
                            </div>
                        </div>
                    ))
                ) : filteredCandidates.length === 0 ? (
                    <motion.div
                        className="empty-state"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                    >
                        <User size={48} />
                        <h3>No candidates found</h3>
                        <p>Scrape LinkedIn profiles to add candidates.</p>
                    </motion.div>
                ) : (
                    filteredCandidates.map((candidate, index) => (
                        <motion.div
                            key={candidate.id}
                            className="candidate-row"
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: index * 0.05 }}
                        >
                            <div className="candidate-avatar">
                                <User size={24} />
                            </div>
                            <div className="candidate-info">
                                <h4>{candidate.name || 'Unknown'}</h4>
                                <p>{candidate.current_title} {candidate.current_company && `at ${candidate.current_company}`}</p>
                            </div>
                            <div className="candidate-meta">
                                <span className="candidate-time">
                                    <Clock size={14} />
                                    {new Date(candidate.scraped_at).toLocaleDateString()}
                                </span>
                                <a
                                    href={candidate.linkedin_url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="candidate-link"
                                >
                                    <ExternalLink size={16} />
                                </a>
                            </div>
                        </motion.div>
                    ))
                )}
            </div>
        </div>
    )
}
