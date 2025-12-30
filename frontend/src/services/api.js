const API_BASE = 'http://localhost:5000/api'

async function fetchAPI(endpoint, options = {}) {
    const response = await fetch(`${API_BASE}${endpoint}`, {
        headers: {
            'Content-Type': 'application/json',
            ...options.headers,
        },
        ...options,
    })

    const data = await response.json()

    if (!response.ok) {
        throw new Error(data.error || data.message || 'API request failed')
    }

    return data
}

// Jobs
export async function getJobs() {
    return fetchAPI('/jobs')
}

export async function getJob(id) {
    return fetchAPI(`/jobs/${id}`)
}

export async function createJob(data) {
    return fetchAPI('/jobs', {
        method: 'POST',
        body: JSON.stringify(data),
    })
}

export async function deleteJob(id) {
    return fetchAPI(`/jobs/${id}`, {
        method: 'DELETE',
    })
}

// Candidates
export async function getCandidates() {
    return fetchAPI('/candidates')
}

export async function getCandidate(id) {
    return fetchAPI(`/candidates/${id}`)
}

export async function scrapeCandidate(linkedinUrl, refresh = false) {
    return fetchAPI('/candidates/scrape', {
        method: 'POST',
        body: JSON.stringify({ linkedin_url: linkedinUrl, refresh }),
    })
}

// Scoring
export async function scoreCandidate(jobId, candidateId, provider = null) {
    return fetchAPI('/score', {
        method: 'POST',
        body: JSON.stringify({
            job_id: jobId,
            candidate_id: candidateId,
            llm_provider: provider
        }),
    })
}

export async function quickScore(linkedinUrl, jobDescription, provider = null) {
    return fetchAPI('/quick-score', {
        method: 'POST',
        body: JSON.stringify({
            linkedin_url: linkedinUrl,
            job_description: jobDescription,
            llm_provider: provider
        }),
    })
}

// Providers
export async function getProviders() {
    return fetchAPI('/providers')
}
