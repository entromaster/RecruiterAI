import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
    Plus,
    Search,
    Briefcase,
    Users,
    ChevronRight,
    X
} from 'lucide-react'
import { getJobs, createJob } from '../services/api'
import './Jobs.css'

export default function Jobs() {
    const [jobs, setJobs] = useState([])
    const [loading, setLoading] = useState(true)
    const [showModal, setShowModal] = useState(false)
    const [searchQuery, setSearchQuery] = useState('')
    const [formData, setFormData] = useState({
        title: '',
        company: '',
        description: '',
        requirements: '',
        location: ''
    })

    useEffect(() => {
        loadJobs()
    }, [])

    async function loadJobs() {
        try {
            const data = await getJobs()
            setJobs(data.jobs || [])
        } catch (error) {
            console.error('Failed to load jobs:', error)
        } finally {
            setLoading(false)
        }
    }

    async function handleCreateJob(e) {
        e.preventDefault()
        try {
            await createJob(formData)
            setShowModal(false)
            setFormData({ title: '', company: '', description: '', requirements: '', location: '' })
            loadJobs()
        } catch (error) {
            console.error('Failed to create job:', error)
        }
    }

    const filteredJobs = jobs.filter(job =>
        job.title?.toLowerCase().includes(searchQuery.toLowerCase()) ||
        job.company?.toLowerCase().includes(searchQuery.toLowerCase())
    )

    return (
        <div className="jobs-page">
            <motion.div
                className="page-header"
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
            >
                <div>
                    <h1>Jobs</h1>
                    <p>Manage your job postings and find the best candidates.</p>
                </div>
                <button className="btn btn-primary" onClick={() => setShowModal(true)}>
                    <Plus size={18} />
                    New Job
                </button>
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
                    placeholder="Search jobs by title or company..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                />
            </motion.div>

            <div className="jobs-grid">
                {loading ? (
                    Array(3).fill(0).map((_, i) => (
                        <div key={i} className="card skeleton-card">
                            <div className="skeleton" style={{ height: 24, width: '60%', marginBottom: 8 }} />
                            <div className="skeleton" style={{ height: 16, width: '40%', marginBottom: 16 }} />
                            <div className="skeleton" style={{ height: 60, marginBottom: 16 }} />
                            <div className="skeleton" style={{ height: 32, width: '30%' }} />
                        </div>
                    ))
                ) : filteredJobs.length === 0 ? (
                    <motion.div
                        className="empty-state"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                    >
                        <Briefcase size={48} />
                        <h3>No jobs found</h3>
                        <p>Create your first job to start finding candidates.</p>
                        <button className="btn btn-primary" onClick={() => setShowModal(true)}>
                            <Plus size={18} />
                            Create Job
                        </button>
                    </motion.div>
                ) : (
                    filteredJobs.map((job, index) => (
                        <motion.div
                            key={job.id}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: index * 0.1 }}
                        >
                            <Link to={`/jobs/${job.id}`} className="job-card card">
                                <div className="job-header">
                                    <div className="job-icon">
                                        <Briefcase size={24} />
                                    </div>
                                    <div className="job-info">
                                        <h3>{job.title}</h3>
                                        <span className="job-company">{job.company || 'Company'}</span>
                                    </div>
                                </div>
                                <p className="job-description">
                                    {job.description?.slice(0, 150)}...
                                </p>
                                <div className="job-footer">
                                    <div className="job-candidates">
                                        <Users size={16} />
                                        <span>{job.candidate_count || 0} candidates</span>
                                    </div>
                                    <ChevronRight size={18} className="job-arrow" />
                                </div>
                            </Link>
                        </motion.div>
                    ))
                )}
            </div>

            {/* Create Job Modal */}
            {showModal && (
                <div className="modal-overlay" onClick={() => setShowModal(false)}>
                    <motion.div
                        className="modal"
                        onClick={e => e.stopPropagation()}
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                    >
                        <div className="modal-header">
                            <h2>Create New Job</h2>
                            <button className="btn-ghost modal-close" onClick={() => setShowModal(false)}>
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
                                    value={formData.title}
                                    onChange={e => setFormData({ ...formData, title: e.target.value })}
                                    required
                                />
                            </div>
                            <div className="form-group">
                                <label>Company</label>
                                <input
                                    type="text"
                                    className="input"
                                    placeholder="e.g. TechCorp"
                                    value={formData.company}
                                    onChange={e => setFormData({ ...formData, company: e.target.value })}
                                />
                            </div>
                            <div className="form-group">
                                <label>Location</label>
                                <input
                                    type="text"
                                    className="input"
                                    placeholder="e.g. Remote, New York, etc."
                                    value={formData.location}
                                    onChange={e => setFormData({ ...formData, location: e.target.value })}
                                />
                            </div>
                            <div className="form-group">
                                <label>Description *</label>
                                <textarea
                                    className="input textarea"
                                    placeholder="Describe the role and responsibilities..."
                                    rows={4}
                                    value={formData.description}
                                    onChange={e => setFormData({ ...formData, description: e.target.value })}
                                    required
                                />
                            </div>
                            <div className="form-group">
                                <label>Requirements</label>
                                <textarea
                                    className="input textarea"
                                    placeholder="List the skills and qualifications needed..."
                                    rows={3}
                                    value={formData.requirements}
                                    onChange={e => setFormData({ ...formData, requirements: e.target.value })}
                                />
                            </div>
                            <div className="modal-footer">
                                <button type="button" className="btn btn-secondary" onClick={() => setShowModal(false)}>
                                    Cancel
                                </button>
                                <button type="submit" className="btn btn-primary">
                                    Create Job
                                </button>
                            </div>
                        </form>
                    </motion.div>
                </div>
            )}
        </div>
    )
}
