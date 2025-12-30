"""
Scoring Service
===============
Combines LinkedIn data with LLM analysis to generate candidate scores.
"""

from typing import Optional
from services.llm_service import get_llm_provider


class ScoringService:
    """Service for scoring candidates against job descriptions"""
    
    def __init__(self, llm_provider: Optional[str] = None):
        """
        Initialize the scoring service.
        
        Args:
            llm_provider: 'openai' or 'gemini'. If None, uses the default.
        """
        self.llm = get_llm_provider(llm_provider)
        self.provider_name = llm_provider or "default"
    
    def score_candidate(self, profile: dict, job_description: str) -> dict:
        """
        Score a candidate against a job description.
        
        Args:
            profile: Candidate profile data (from LinkedIn scraper)
            job_description: The job description text
            
        Returns:
            Dictionary with score, analysis, skill_matches, recommendation
        """
        result = self.llm.analyze_profile(profile, job_description)
        
        # Ensure all required fields are present
        return {
            "score": result.get("score", 50),
            "analysis": result.get("analysis", ""),
            "skill_matches": result.get("skill_matches", []),
            "skill_gaps": result.get("skill_gaps", []),
            "recommendation": result.get("recommendation", "weak_match"),
            "llm_provider": self.provider_name
        }
    
    def generate_profile_summary(self, profile: dict) -> str:
        """
        Generate a summary of a candidate's profile.
        
        Args:
            profile: Candidate profile data
            
        Returns:
            Summary string
        """
        return self.llm.generate_summary(profile)
    
    @staticmethod
    def get_recommendation_label(recommendation: str) -> str:
        """Convert recommendation code to human-readable label"""
        labels = {
            "strong_match": "Strong Match ✅",
            "good_match": "Good Match 👍",
            "weak_match": "Weak Match ⚠️",
            "no_match": "Not a Match ❌"
        }
        return labels.get(recommendation, recommendation)
    
    @staticmethod
    def get_score_color(score: float) -> str:
        """Get color code for a score (for UI)"""
        if score >= 80:
            return "#22c55e"  # Green
        elif score >= 60:
            return "#84cc16"  # Lime
        elif score >= 40:
            return "#eab308"  # Yellow
        elif score >= 20:
            return "#f97316"  # Orange
        else:
            return "#ef4444"  # Red
