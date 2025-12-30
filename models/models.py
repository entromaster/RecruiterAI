"""MongoDB models for RecruiterAI

These are document schemas and helper functions for MongoDB collections.
"""

from datetime import datetime
from bson import ObjectId
from extensions import get_db


def serialize_doc(doc):
    """Convert MongoDB document to JSON-serializable dict"""
    if doc is None:
        return None
    if "_id" in doc:
        doc["id"] = str(doc["_id"])
        del doc["_id"]
    return doc


# ============================================================================
# JOB MODEL
# ============================================================================

class JobModel:
    """Job document operations"""
    
    collection_name = "jobs"
    
    @classmethod
    def get_collection(cls):
        return get_db()[cls.collection_name]
    
    @classmethod
    def create(cls, title: str, description: str, company: str = None, 
               requirements: str = None, location: str = None) -> dict:
        """Create a new job"""
        doc = {
            "title": title,
            "company": company,
            "description": description,
            "requirements": requirements,
            "location": location,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        result = cls.get_collection().insert_one(doc)
        doc["_id"] = result.inserted_id
        return serialize_doc(doc)
    
    @classmethod
    def get_all(cls) -> list:
        """Get all jobs"""
        docs = cls.get_collection().find().sort("created_at", -1)
        return [serialize_doc(doc) for doc in docs]
    
    @classmethod
    def get_by_id(cls, job_id: str) -> dict:
        """Get job by ID"""
        try:
            doc = cls.get_collection().find_one({"_id": ObjectId(job_id)})
            return serialize_doc(doc)
        except:
            return None
    
    @classmethod
    def update(cls, job_id: str, **updates) -> dict:
        """Update a job"""
        updates["updated_at"] = datetime.utcnow()
        try:
            cls.get_collection().update_one(
                {"_id": ObjectId(job_id)},
                {"$set": updates}
            )
            return cls.get_by_id(job_id)
        except:
            return None
    
    @classmethod
    def delete(cls, job_id: str) -> bool:
        """Delete a job and its related scores"""
        try:
            # Delete related scores first
            ScoreModel.delete_by_job(job_id)
            result = cls.get_collection().delete_one({"_id": ObjectId(job_id)})
            return result.deleted_count > 0
        except:
            return False


# ============================================================================
# CANDIDATE MODEL
# ============================================================================

class CandidateModel:
    """Candidate document operations"""
    
    collection_name = "candidates"
    
    @classmethod
    def get_collection(cls):
        return get_db()[cls.collection_name]
    
    @classmethod
    def create(cls, linkedin_url: str, name: str = None, about: str = None,
               current_title: str = None, current_company: str = None,
               profile_data: dict = None) -> dict:
        """Create a new candidate"""
        doc = {
            "linkedin_url": linkedin_url,
            "name": name,
            "about": about,
            "current_title": current_title,
            "current_company": current_company,
            "profile_data": profile_data or {},
            "scraped_at": datetime.utcnow()
        }
        result = cls.get_collection().insert_one(doc)
        doc["_id"] = result.inserted_id
        return serialize_doc(doc)
    
    @classmethod
    def get_all(cls) -> list:
        """Get all candidates"""
        docs = cls.get_collection().find().sort("scraped_at", -1)
        return [serialize_doc(doc) for doc in docs]
    
    @classmethod
    def get_by_id(cls, candidate_id: str) -> dict:
        """Get candidate by ID"""
        try:
            doc = cls.get_collection().find_one({"_id": ObjectId(candidate_id)})
            return serialize_doc(doc)
        except:
            return None
    
    @classmethod
    def get_by_linkedin_url(cls, linkedin_url: str) -> dict:
        """Get candidate by LinkedIn URL"""
        doc = cls.get_collection().find_one({"linkedin_url": linkedin_url})
        return serialize_doc(doc)
    
    @classmethod
    def update(cls, candidate_id: str, **updates) -> dict:
        """Update a candidate"""
        try:
            cls.get_collection().update_one(
                {"_id": ObjectId(candidate_id)},
                {"$set": updates}
            )
            return cls.get_by_id(candidate_id)
        except:
            return None
    
    @classmethod
    def upsert_by_url(cls, linkedin_url: str, **data) -> dict:
        """Create or update candidate by LinkedIn URL"""
        existing = cls.get_by_linkedin_url(linkedin_url)
        if existing:
            data["scraped_at"] = datetime.utcnow()
            return cls.update(existing["id"], **data)
        else:
            return cls.create(linkedin_url=linkedin_url, **data)


# ============================================================================
# SCORE MODEL
# ============================================================================

class ScoreModel:
    """Score document operations"""
    
    collection_name = "scores"
    
    @classmethod
    def get_collection(cls):
        return get_db()[cls.collection_name]
    
    @classmethod
    def create(cls, job_id: str, candidate_id: str, score: float,
               analysis: str = None, skill_matches: list = None,
               recommendation: str = None, llm_provider: str = None) -> dict:
        """Create a new score"""
        doc = {
            "job_id": job_id,
            "candidate_id": candidate_id,
            "score": score,
            "analysis": analysis,
            "skill_matches": skill_matches or [],
            "recommendation": recommendation,
            "llm_provider": llm_provider,
            "created_at": datetime.utcnow()
        }
        result = cls.get_collection().insert_one(doc)
        doc["_id"] = result.inserted_id
        return serialize_doc(doc)
    
    @classmethod
    def get_by_id(cls, score_id: str) -> dict:
        """Get score by ID"""
        try:
            doc = cls.get_collection().find_one({"_id": ObjectId(score_id)})
            return serialize_doc(doc)
        except:
            return None
    
    @classmethod
    def get_by_job_and_candidate(cls, job_id: str, candidate_id: str) -> dict:
        """Get score by job and candidate"""
        doc = cls.get_collection().find_one({
            "job_id": job_id,
            "candidate_id": candidate_id
        })
        return serialize_doc(doc)
    
    @classmethod
    def get_by_job(cls, job_id: str) -> list:
        """Get all scores for a job, sorted by score descending"""
        docs = cls.get_collection().find({"job_id": job_id}).sort("score", -1)
        return [serialize_doc(doc) for doc in docs]
    
    @classmethod
    def get_by_candidate(cls, candidate_id: str) -> list:
        """Get all scores for a candidate"""
        docs = cls.get_collection().find({"candidate_id": candidate_id})
        return [serialize_doc(doc) for doc in docs]
    
    @classmethod
    def update(cls, score_id: str, **updates) -> dict:
        """Update a score"""
        try:
            cls.get_collection().update_one(
                {"_id": ObjectId(score_id)},
                {"$set": updates}
            )
            return cls.get_by_id(score_id)
        except:
            return None
    
    @classmethod
    def upsert(cls, job_id: str, candidate_id: str, **data) -> dict:
        """Create or update score for job-candidate pair"""
        existing = cls.get_by_job_and_candidate(job_id, candidate_id)
        if existing:
            return cls.update(existing["id"], **data)
        else:
            return cls.create(job_id=job_id, candidate_id=candidate_id, **data)
    
    @classmethod
    def delete_by_job(cls, job_id: str) -> int:
        """Delete all scores for a job"""
        result = cls.get_collection().delete_many({"job_id": job_id})
        return result.deleted_count
    
    @classmethod
    def delete_by_candidate(cls, candidate_id: str) -> int:
        """Delete all scores for a candidate"""
        result = cls.get_collection().delete_many({"candidate_id": candidate_id})
        return result.deleted_count
