"""
API Routes
==========
REST API endpoints for the RecruiterAI backend.
"""

from flask import Blueprint, request, jsonify
from models.models import JobModel, CandidateModel, ScoreModel
from services.scoring_service import ScoringService
from services.linkedin_service import get_linkedin_service

api_bp = Blueprint("api", __name__)


# ============================================================================
# JOB ENDPOINTS
# ============================================================================

@api_bp.route("/jobs", methods=["GET"])
def get_jobs():
    """Get all jobs"""
    jobs = JobModel.get_all()
    return jsonify({
        "jobs": jobs,
        "count": len(jobs)
    })


@api_bp.route("/jobs", methods=["POST"])
def create_job():
    """Create a new job"""
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    if not data.get("title") or not data.get("description"):
        return jsonify({"error": "Title and description are required"}), 400
    
    job = JobModel.create(
        title=data["title"],
        company=data.get("company"),
        description=data["description"],
        requirements=data.get("requirements"),
        location=data.get("location")
    )
    
    return jsonify({
        "message": "Job created successfully",
        "job": job
    }), 201


@api_bp.route("/jobs/<job_id>", methods=["GET"])
def get_job(job_id):
    """Get a specific job with its candidates"""
    job = JobModel.get_by_id(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    
    # Get all scores for this job with candidate info
    scores = ScoreModel.get_by_job(job_id)
    
    # Enrich scores with candidate data
    for score in scores:
        candidate = CandidateModel.get_by_id(score["candidate_id"])
        if candidate:
            score["candidate"] = {
                "id": candidate["id"],
                "name": candidate.get("name"),
                "current_title": candidate.get("current_title"),
                "current_company": candidate.get("current_company"),
                "linkedin_url": candidate.get("linkedin_url")
            }
    
    return jsonify({
        "job": job,
        "candidates": scores
    })


@api_bp.route("/jobs/<job_id>", methods=["DELETE"])
def delete_job(job_id):
    """Delete a job"""
    if JobModel.delete(job_id):
        return jsonify({"message": "Job deleted successfully"})
    return jsonify({"error": "Job not found"}), 404


# ============================================================================
# CANDIDATE ENDPOINTS
# ============================================================================

@api_bp.route("/candidates", methods=["GET"])
def get_candidates():
    """Get all candidates"""
    candidates = CandidateModel.get_all()
    # Remove profile_data from list view for performance
    for c in candidates:
        c.pop("profile_data", None)
    return jsonify({
        "candidates": candidates,
        "count": len(candidates)
    })


@api_bp.route("/candidates/<candidate_id>", methods=["GET"])
def get_candidate(candidate_id):
    """Get a specific candidate with full profile"""
    candidate = CandidateModel.get_by_id(candidate_id)
    if not candidate:
        return jsonify({"error": "Candidate not found"}), 404
    
    scores = ScoreModel.get_by_candidate(candidate_id)
    
    # Enrich scores with job data
    for score in scores:
        job = JobModel.get_by_id(score["job_id"])
        if job:
            score["job"] = {
                "id": job["id"],
                "title": job.get("title"),
                "company": job.get("company")
            }
    
    return jsonify({
        "candidate": candidate,
        "scores": scores
    })


@api_bp.route("/candidates/scrape", methods=["POST"])
def scrape_candidate():
    """Scrape a LinkedIn profile and save as candidate"""
    data = request.get_json()
    
    if not data or not data.get("linkedin_url"):
        return jsonify({"error": "linkedin_url is required"}), 400
    
    linkedin_url = data["linkedin_url"]
    
    # Check if candidate already exists
    existing = CandidateModel.get_by_linkedin_url(linkedin_url)
    if existing and not data.get("refresh"):
        return jsonify({
            "message": "Candidate already exists",
            "candidate": existing
        })
    
    try:
        # Scrape the profile
        linkedin_service = get_linkedin_service()
        profile_data = linkedin_service.scrape_profile(linkedin_url)
        
        # Create or update candidate
        candidate = CandidateModel.upsert_by_url(
            linkedin_url=linkedin_url,
            name=profile_data.get("name"),
            about=profile_data.get("about"),
            current_title=profile_data.get("job_title"),
            current_company=profile_data.get("company"),
            profile_data=profile_data
        )
        
        return jsonify({
            "message": "Profile scraped successfully",
            "candidate": candidate
        }), 201
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================================
# SCORING ENDPOINTS
# ============================================================================

@api_bp.route("/score", methods=["POST"])
def score_candidate_endpoint():
    """Score a candidate against a job"""
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    job_id = data.get("job_id")
    candidate_id = data.get("candidate_id")
    llm_provider = data.get("llm_provider")  # Optional: 'openai' or 'gemini'
    
    if not job_id or not candidate_id:
        return jsonify({"error": "job_id and candidate_id are required"}), 400
    
    # Get job and candidate
    job = JobModel.get_by_id(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    
    candidate = CandidateModel.get_by_id(candidate_id)
    if not candidate:
        return jsonify({"error": "Candidate not found"}), 404
    
    # Check if score already exists
    existing_score = ScoreModel.get_by_job_and_candidate(job_id, candidate_id)
    if existing_score and not data.get("refresh"):
        return jsonify({
            "message": "Score already exists",
            "score": existing_score
        })
    
    try:
        # Get profile data
        profile_data = candidate.get("profile_data", {})
        if not profile_data:
            return jsonify({"error": "Candidate has no profile data"}), 400
        
        # Score the candidate
        scoring_service = ScoringService(llm_provider)
        result = scoring_service.score_candidate(profile_data, job["description"])
        
        # Create or update score
        score = ScoreModel.upsert(
            job_id=job_id,
            candidate_id=candidate_id,
            score=result["score"],
            analysis=result["analysis"],
            skill_matches=result["skill_matches"],
            recommendation=result["recommendation"],
            llm_provider=result["llm_provider"]
        )
        
        # Add job and candidate info to response
        score["job"] = {"id": job["id"], "title": job.get("title")}
        score["candidate"] = {"id": candidate["id"], "name": candidate.get("name")}
        
        return jsonify({
            "message": "Candidate scored successfully",
            "score": score
        }), 201
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/quick-score", methods=["POST"])
def quick_score():
    """Quick score a LinkedIn profile against a job description (uses cached profile only)"""
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    linkedin_url = data.get("linkedin_url")
    job_description = data.get("job_description")
    llm_provider = data.get("llm_provider")
    
    if not linkedin_url or not job_description:
        return jsonify({"error": "linkedin_url and job_description are required"}), 400
    
    try:
        # Check if we already have this candidate
        candidate = CandidateModel.get_by_linkedin_url(linkedin_url)
        
        if candidate and candidate.get("profile_data"):
            profile_data = candidate["profile_data"]
        else:
            # Profile not found - return helpful error
            return jsonify({
                "error": "Profile not found in database. Please use 'Add Candidate' first to scrape this LinkedIn profile, or use an existing profile like: https://www.linkedin.com/in/kalrahariom/"
            }), 404
        
        # Score the candidate
        scoring_service = ScoringService(llm_provider)
        result = scoring_service.score_candidate(profile_data, job_description)
        
        return jsonify({
            "profile": {
                "name": profile_data.get("name"),
                "job_title": profile_data.get("job_title"),
                "company": profile_data.get("company")
            },
            "result": result
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================================
# UTILITY ENDPOINTS
# ============================================================================

@api_bp.route("/providers", methods=["GET"])
def get_providers():
    """Get available LLM providers"""
    from config import Config
    return jsonify({
        "providers": Config.get_available_llm_providers(),
        "default": Config.DEFAULT_LLM_PROVIDER
    })
