import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
    Briefcase,
    Users,
    TrendingUp,
    Sparkles,
    ArrowUpRight,
    Clock,
    X,
    Link as LinkIcon,
    FileText
} from 'lucide-react'
import { getJobs, getCandidates, createJob, scrapeCandidate, quickScore } from '../services/api'
import './Dashboard.css'

export default function Dashboard() {
    const navigate = useNavigate()
    const [stats, setStats] = useState({
        jobs: 0,
        candidates: 0,
        avgScore: 0,
        matches: 0
    })
    const [recentActivity, setRecentActivity] = useState([])
    const [showJobModal, setShowJobModal] = useState(false)
    const [showCandidateModal, setShowCandidateModal] = useState(false)
    const [showQuickScoreModal, setShowQuickScoreModal] = useState(false)
    const [loading, setLoading] = useState(false)

    const [jobForm, setJobForm] = useState({ title: '', company: '', description: '', requirements: '', location: '' })
    const [candidateForm, setCandidateForm] = useState({ linkedin_url: '' })
    const [quickScoreForm, setQuickScoreForm] = useState({ linkedin_url: '', job_description: '' })
    const [quickScoreResult, setQuickScoreResult] = useState(null)

    useEffect(() => {
        loadData()
    }, [])

    async function loadData() {
        try {
            const [jobsData, candidatesData] = await Promise.all([
                getJobs(),
                getCandidates()
            ])

            setStats({
                jobs: jobsData.count || 0,
                candidates: candidatesData.count || 0,
                avgScore: 72,
                matches: Math.floor((candidatesData.count || 0) * 0.3)
            })

            // Create activity from recent data
            const activities = []
            if (candidatesData.candidates?.length > 0) {
                const recent = candidatesData.candidates.slice(0, 3)
                recent.forEach(c => {
                    activities.push({
                        action: 'Profile scraped',
                        details: c.name || 'LinkedIn profile',
                        time: formatTime(c.scraped_at)
                    })
                })
            }
            if (jobsData.jobs?.length > 0) {
                activities.push({
                    action: 'Job created',
                    details: jobsData.jobs[0].title,
                    time: formatTime(jobsData.jobs[0].created_at)
                })
            }
            setRecentActivity(activities.slice(0, 4))
        } catch (error) {
            console.error('Failed to load data:', error)
        }
    }

    function formatTime(dateStr) {
        if (!dateStr) return 'Recently'
        const date = new Date(dateStr)
        const now = new Date()
        const diff = now - date
        const minutes = Math.floor(diff / 60000)
        const hours = Math.floor(diff / 3600000)
        const days = Math.floor(diff / 86400000)

        if (minutes < 60) return `${minutes}m ago`
        if (hours < 24) return `${hours}h ago`
        return `${days}d ago`
    }

    async function handleCreateJob(e) {
        e.preventDefault()
        setLoading(true)
        try {
            await createJob(jobForm)
            setShowJobModal(false)
            setJobForm({ title: '', company: '', description: '', requirements: '', location: '' })
            loadData()
        } catch (error) {
            console.error('Failed to create job:', error)
            alert('Failed to create job: ' + error.message)
        } finally {
            setLoading(false)
        }
    }

    async function handleAddCandidate(e) {
        e.preventDefault()
        setLoading(true)
        try {
            await scrapeCandidate(candidateForm.linkedin_url)
            setShowCandidateModal(false)
            setCandidateForm({ linkedin_url: '' })
            loadData()
            alert('Candidate profile scraped successfully!')
        } catch (error) {
            console.error('Failed to scrape candidate:', error)
            alert('Failed to scrape profile: ' + error.message)
        } finally {
            setLoading(false)
        }
    }

    async function handleQuickScore(e) {
        e.preventDefault()
        setLoading(true)
        setQuickScoreResult(null)
        try {
            const result = await quickScore(quickScoreForm.linkedin_url, quickScoreForm.job_description)
            setQuickScoreResult(result)
        } catch (error) {
            console.error('Failed to quick score:', error)
            alert('Failed to score: ' + error.message)
        } finally {
            setLoading(false)
        }
    }

    const statCards = [
        { label: 'Total Jobs', value: stats.jobs, change: 'Active postings', icon: Briefcase, color: 'primary' },
        { label: 'Candidates', value: stats.candidates, change: 'Scraped profiles', icon: Users, color: 'teal' },
        { label: 'Avg. Score', value: stats.avgScore, change: 'Match percentage', icon: TrendingUp, color: 'success' },
        { label: 'Matches', value: stats.matches, change: 'Strong matches', icon: Sparkles, color: 'warning' },
    ]

    return (
        <div className="dashboard">
            <motion.div
                className="page-header"
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
            >
                <div>
                    <h1>Dashboard</h1>
                    <p>Welcome back! Here's your recruiting overview.</p>
                </div>
                <button className="btn btn-primary" onClick={() => setShowQuickScoreModal(true)}>
                    <Sparkles size={18} />
                    New Analysis
                </button>
            </motion.div>

            <div className="stats-grid">
                {statCards.map((stat, index) => (
                    <motion.div
                        key={stat.label}
                        className={`stat-card stat-${stat.color}`}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: index * 0.1 }}
                    >
                        <div className="stat-icon">
                            <stat.icon size={24} />
                        </div>
                        <div className="stat-content">
                            <span className="stat-value">{stat.value}</span>
                            <span className="stat-label">{stat.label}</span>
                            <span className="stat-change">{stat.change}</span>
                        </div>
                    </motion.div>
                ))}
            </div>

            <div className="dashboard-grid">
                <motion.div
                    className="card activity-card"
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.4 }}
                >
                    <div className="card-header">
                        <h3>Recent Activity</h3>
                        <button className="view-all" onClick={() => navigate('/candidates')}>
                            View all <ArrowUpRight size={16} />
                        </button>
                    </div>
                    <div className="activity-list">
                        {recentActivity.length === 0 ? (
                            <p className="no-activity">No recent activity. Start by creating a job or adding candidates!</p>
                        ) : (
                            recentActivity.map((item, index) => (
                                <div key={index} className="activity-item">
                                    <div className="activity-dot" />
                                    <div className="activity-content">
                                        <span className="activity-action">{item.action}</span>
                                        <span className="activity-details">{item.details}</span>
                                        {item.score && (
                                            <span className="activity-score">Score: {item.score}/100</span>
                                        )}
                                    </div>
                                    <span className="activity-time">
                                        <Clock size={14} />
                                        {item.time}
                                    </span>
                                </div>
                            ))
                        )}
                    </div>
                </motion.div>

                <motion.div
                    className="card quick-actions-card"
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.5 }}
                >
                    <h3>Quick Actions</h3>
                    <div className="quick-actions">
                        <button className="quick-action" onClick={() => setShowJobModal(true)}>
                            <Briefcase size={20} />
                            <span>Create Job</span>
                        </button>
                        <button className="quick-action" onClick={() => setShowCandidateModal(true)}>
                            <Users size={20} />
                            <span>Add Candidate</span>
                        </button>
                        <button className="quick-action" onClick={() => setShowQuickScoreModal(true)}>
                            <Sparkles size={20} />
                            <span>Quick Score</span>
                        </button>
                    </div>
                </motion.div>
            </div>

            {/* Create Job Modal */}
            {showJobModal && (
                <div className="modal-overlay" onClick={() => setShowJobModal(false)}>
                    <motion.div
                        className="modal"
                        onClick={e => e.stopPropagation()}
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                    >
                        <div className="modal-header">
                            <h2>Create New Job</h2>
                            <button className="btn-ghost modal-close" onClick={() => setShowJobModal(false)}>
                                <X size={20} />
                            </button>
                        </div>
                        <form onSubmit={handleCreateJob}>
                            <div className="form-group">
                                <label>Job Title *</label>
                                <input
                                    type="text"
                                    className="input"
                                    placeholder="e.g. Senior Software Engineer"
                                    value={jobForm.title}
                                    onChange={e => setJobForm({ ...jobForm, title: e.target.value })}
                                    required
                                />
                            </div>
                            <div className="form-group">
                                <label>Company</label>
                                <input
                                    type="text"
                                    className="input"
                                    placeholder="e.g. TechCorp"
                                    value={jobForm.company}
                                    onChange={e => setJobForm({ ...jobForm, company: e.target.value })}
                                />
                            </div>
                            <div className="form-group">
                                <label>Location</label>
                                <input
                                    type="text"
                                    className="input"
                                    placeholder="e.g. Remote, New York, etc."
                                    value={jobForm.location}
                                    onChange={e => setJobForm({ ...jobForm, location: e.target.value })}
                                />
                            </div>
                            <div className="form-group">
                                <label>Description *</label>
                                <textarea
                                    className="input textarea"
                                    placeholder="Describe the role and responsibilities..."
                                    rows={4}
                                    value={jobForm.description}
                                    onChange={e => setJobForm({ ...jobForm, description: e.target.value })}
                                    required
                                />
                            </div>
                            <div className="modal-footer">
                                <button type="button" className="btn btn-secondary" onClick={() => setShowJobModal(false)}>
                                    Cancel
                                </button>
                                <button type="submit" className="btn btn-primary" disabled={loading}>
                                    {loading ? 'Creating...' : 'Create Job'}
                                </button>
                            </div>
                        </form>
                    </motion.div>
                </div>
            )}

            {/* Add Candidate Modal */}
            {showCandidateModal && (
                <div className="modal-overlay" onClick={() => setShowCandidateModal(false)}>
                    <motion.div
                        className="modal"
                        onClick={e => e.stopPropagation()}
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                    >
                        <div className="modal-header">
                            <h2>Add Candidate</h2>
                            <button className="btn-ghost modal-close" onClick={() => setShowCandidateModal(false)}>
                                <X size={20} />
                            </button>
                        </div>
                        <form onSubmit={handleAddCandidate}>
                            <div className="form-group">
                                <label><LinkIcon size={16} style={{ marginRight: 8, verticalAlign: 'middle' }} />LinkedIn Profile URL *</label>
                                <input
                                    type="url"
                                    className="input"
                                    placeholder="https://www.linkedin.com/in/username/"
                                    value={candidateForm.linkedin_url}
                                    onChange={e => setCandidateForm({ ...candidateForm, linkedin_url: e.target.value })}
                                    required
                                />
                                <p className="form-hint">Enter the full LinkedIn profile URL to scrape candidate information.</p>
                            </div>
                            <div className="modal-footer">
                                <button type="button" className="btn btn-secondary" onClick={() => setShowCandidateModal(false)}>
                                    Cancel
                                </button>
                                <button type="submit" className="btn btn-primary" disabled={loading}>
                                    {loading ? 'Scraping...' : 'Scrape Profile'}
                                </button>
                            </div>
                        </form>
                    </motion.div>
                </div>
            )}

            {/* Quick Score Modal */}
            {showQuickScoreModal && (
                <div className="modal-overlay" onClick={() => !loading && setShowQuickScoreModal(false)}>
                    <motion.div
                        className="modal modal-large"
                        onClick={e => e.stopPropagation()}
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                    >
                        <div className="modal-header">
                            <h2><Sparkles size={24} style={{ marginRight: 12, verticalAlign: 'middle' }} />Quick Score</h2>
                            <button className="btn-ghost modal-close" onClick={() => !loading && setShowQuickScoreModal(false)}>
                                <X size={20} />
                            </button>
                        </div>

                        {quickScoreResult ? (
                            <div className="score-result">
                                <div className="score-header">
                                    <div className="score-circle large">
                                        {quickScoreResult.result?.score || 0}
                                    </div>
                                    <div className="score-meta">
                                        <span className={`badge badge-${quickScoreResult.result?.recommendation === 'strong_match' ? 'success' : quickScoreResult.result?.recommendation === 'good_match' ? 'info' : 'warning'}`}>
                                            {quickScoreResult.result?.recommendation?.replace('_', ' ').toUpperCase() || 'N/A'}
                                        </span>
                                        <p style={{ marginTop: 8 }}>Analyzed by {quickScoreResult.result?.llm_provider || 'AI'}</p>
                                    </div>
                                </div>
                                <div className="score-analysis">
                                    <h4>Analysis</h4>
                                    <p>{quickScoreResult.result?.analysis}</p>
                                </div>
                                <div className="score-skills">
                                    <div className="skill-section">
                                        <h5>✅ Skills Matched</h5>
                                        <div className="skill-tags">
                                            {quickScoreResult.result?.skill_matches?.length > 0
                                                ? quickScoreResult.result.skill_matches.map((s, i) => <span key={i} className="skill-tag success">{s}</span>)
                                                : <span className="no-skills">None identified</span>}
                                        </div>
                                    </div>
                                    <div className="skill-section">
                                        <h5>❌ Skills Missing</h5>
                                        <div className="skill-tags">
                                            {quickScoreResult.result?.skill_gaps?.length > 0
                                                ? quickScoreResult.result.skill_gaps.map((s, i) => <span key={i} className="skill-tag error">{s}</span>)
                                                : <span className="no-skills">None identified</span>}
                                        </div>
                                    </div>
                                </div>
                                <div className="modal-footer">
                                    <button className="btn btn-secondary" onClick={() => { setQuickScoreResult(null); setQuickScoreForm({ linkedin_url: '', job_description: '' }) }}>
                                        Score Another
                                    </button>
                                    <button className="btn btn-primary" onClick={() => setShowQuickScoreModal(false)}>
                                        Done
                                    </button>
                                </div>
                            </div>
                        ) : (
                            <form onSubmit={handleQuickScore}>
                                <div className="form-group">
                                    <label><LinkIcon size={16} style={{ marginRight: 8, verticalAlign: 'middle' }} />LinkedIn Profile URL *</label>
                                    <input
                                        type="url"
                                        className="input"
                                        placeholder="https://www.linkedin.com/in/username/"
                                        value={quickScoreForm.linkedin_url}
                                        onChange={e => setQuickScoreForm({ ...quickScoreForm, linkedin_url: e.target.value })}
                                        required
                                    />
                                </div>
                                <div className="form-group">
                                    <label><FileText size={16} style={{ marginRight: 8, verticalAlign: 'middle' }} />Job Description *</label>
                                    <textarea
                                        className="input textarea"
                                        placeholder="Paste the job requirements here..."
                                        rows={6}
                                        value={quickScoreForm.job_description}
                                        onChange={e => setQuickScoreForm({ ...quickScoreForm, job_description: e.target.value })}
                                        required
                                    />
                                </div>
                                <div className="modal-footer">
                                    <button type="button" className="btn btn-secondary" onClick={() => setShowQuickScoreModal(false)}>
                                        Cancel
                                    </button>
                                    <button type="submit" className="btn btn-primary" disabled={loading}>
                                        {loading ? 'Analyzing...' : 'Score Candidate'}
                                    </button>
                                </div>
                            </form>
                        )}
                    </motion.div>
                </div>
            )}
        </div>
    )
}
