"""
LLM Service
===========
Abstraction layer for LLM providers (OpenAI and Google Gemini).
"""

from abc import ABC, abstractmethod
from typing import Optional
import json

from config import Config


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def analyze_profile(self, profile: dict, job_description: str) -> dict:
        """Analyze a profile against a job description"""
        pass
    
    @abstractmethod
    def generate_summary(self, profile: dict) -> str:
        """Generate a summary of a profile"""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider"""
    
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.model = "gpt-4o-mini"  # Cost-effective option
    
    def _call_api(self, system_prompt: str, user_prompt: str) -> str:
        """Make a call to the OpenAI API"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    
    def analyze_profile(self, profile: dict, job_description: str) -> dict:
        """Analyze a profile against a job description"""
        system_prompt = """You are an expert recruiter AI assistant. Analyze the candidate's profile against the job description.
        
Return your analysis as a JSON object with these fields:
- score: A number from 0-100 indicating how well the candidate matches the job
- analysis: A detailed paragraph explaining the match/mismatch
- skill_matches: An array of skills from the job that the candidate has
- skill_gaps: An array of required skills the candidate is missing
- recommendation: One of 'strong_match', 'good_match', 'weak_match', 'no_match'

Be objective and thorough in your analysis."""

        user_prompt = f"""## Job Description:
{job_description}

## Candidate Profile:
Name: {profile.get('name', 'Unknown')}
Current Role: {profile.get('job_title', 'N/A')} at {profile.get('company', 'N/A')}
About: {profile.get('about', 'N/A')}

Experience:
{self._format_experiences(profile.get('experiences', []))}

Education:
{self._format_education(profile.get('educations', []))}

Skills: {', '.join(profile.get('skills', [])) if profile.get('skills') else 'Not listed'}
"""

        response = self._call_api(system_prompt, user_prompt)
        
        # Parse JSON from response
        try:
            # Handle potential markdown code blocks
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            return json.loads(response.strip())
        except json.JSONDecodeError:
            return {
                "score": 50,
                "analysis": response,
                "skill_matches": [],
                "skill_gaps": [],
                "recommendation": "weak_match"
            }
    
    def generate_summary(self, profile: dict) -> str:
        """Generate a summary of a profile"""
        system_prompt = "You are a professional recruiter. Create a concise 2-3 sentence summary of this candidate's profile highlighting their key strengths and experience."
        
        user_prompt = f"""Name: {profile.get('name', 'Unknown')}
Current Role: {profile.get('job_title', 'N/A')} at {profile.get('company', 'N/A')}
About: {profile.get('about', 'N/A')}
Experience: {len(profile.get('experiences', []))} positions
Education: {len(profile.get('educations', []))} degrees"""

        return self._call_api(system_prompt, user_prompt)
    
    def _format_experiences(self, experiences: list) -> str:
        if not experiences:
            return "No experience listed"
        
        lines = []
        for exp in experiences[:5]:  # Limit to 5 most recent
            title = exp.get('title', 'Unknown Role')
            company = exp.get('company', 'Unknown Company')
            duration = exp.get('duration', '')
            lines.append(f"- {title} at {company} ({duration})")
        return "\n".join(lines)
    
    def _format_education(self, educations: list) -> str:
        if not educations:
            return "No education listed"
        
        lines = []
        for edu in educations:
            institution = edu.get('institution', 'Unknown')
            degree = edu.get('degree', '')
            field = edu.get('field', '')
            lines.append(f"- {degree} {field} from {institution}")
        return "\n".join(lines)


class GeminiProvider(LLMProvider):
    """Google Gemini provider"""
    
    def __init__(self):
        import google.generativeai as genai
        genai.configure(api_key=Config.GOOGLE_API_KEY)
        # Use gemini-2.0-flash for latest API
        self.model = genai.GenerativeModel('gemini-2.5-flash')
    
    def _call_api(self, prompt: str) -> str:
        """Make a call to the Gemini API"""
        response = self.model.generate_content(prompt)
        return response.text
    
    def analyze_profile(self, profile: dict, job_description: str) -> dict:
        """Analyze a profile against a job description"""
        prompt = f"""You are an expert recruiter AI assistant. Analyze the candidate's profile against the job description.

## Job Description:
{job_description}

## Candidate Profile:
Name: {profile.get('name', 'Unknown')}
Current Role: {profile.get('job_title', 'N/A')} at {profile.get('company', 'N/A')}
About: {profile.get('about', 'N/A')}

Experience:
{self._format_experiences(profile.get('experiences', []))}

Education:
{self._format_education(profile.get('educations', []))}

Skills: {', '.join(profile.get('skills', [])) if profile.get('skills') else 'Not listed'}

Return your analysis as a JSON object with these fields:
- score: A number from 0-100 indicating how well the candidate matches the job
- analysis: A detailed paragraph explaining the match/mismatch
- skill_matches: An array of skills from the job that the candidate has
- skill_gaps: An array of required skills the candidate is missing
- recommendation: One of 'strong_match', 'good_match', 'weak_match', 'no_match'

Return ONLY the JSON object, no other text."""

        response = self._call_api(prompt)
        
        # Parse JSON from response
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            return json.loads(response.strip())
        except json.JSONDecodeError:
            return {
                "score": 50,
                "analysis": response,
                "skill_matches": [],
                "skill_gaps": [],
                "recommendation": "weak_match"
            }
    
    def generate_summary(self, profile: dict) -> str:
        """Generate a summary of a profile"""
        prompt = f"""You are a professional recruiter. Create a concise 2-3 sentence summary of this candidate's profile highlighting their key strengths and experience.

Name: {profile.get('name', 'Unknown')}
Current Role: {profile.get('job_title', 'N/A')} at {profile.get('company', 'N/A')}
About: {profile.get('about', 'N/A')}
Experience: {len(profile.get('experiences', []))} positions
Education: {len(profile.get('educations', []))} degrees"""

        return self._call_api(prompt)
    
    def _format_experiences(self, experiences: list) -> str:
        if not experiences:
            return "No experience listed"
        
        lines = []
        for exp in experiences[:5]:
            title = exp.get('title', 'Unknown Role')
            company = exp.get('company', 'Unknown Company')
            duration = exp.get('duration', '')
            lines.append(f"- {title} at {company} ({duration})")
        return "\n".join(lines)
    
    def _format_education(self, educations: list) -> str:
        if not educations:
            return "No education listed"
        
        lines = []
        for edu in educations:
            institution = edu.get('institution', 'Unknown')
            degree = edu.get('degree', '')
            field = edu.get('field', '')
            lines.append(f"- {degree} {field} from {institution}")
        return "\n".join(lines)


def get_llm_provider(provider: Optional[str] = None) -> LLMProvider:
    """
    Get an LLM provider instance.
    
    Args:
        provider: 'openai' or 'gemini'. If None, uses the default from config.
        
    Returns:
        LLMProvider instance
        
    Raises:
        ValueError: If the requested provider is not available
    """
    provider = provider or Config.DEFAULT_LLM_PROVIDER
    available = Config.get_available_llm_providers()
    
    if not available:
        raise ValueError("No LLM providers configured. Set OPENAI_API_KEY or GOOGLE_API_KEY in .env")
    
    if provider not in available:
        # Fall back to first available provider
        provider = available[0]
    
    if provider == "openai":
        return OpenAIProvider()
    elif provider == "gemini":
        return GeminiProvider()
    else:
        raise ValueError(f"Unknown provider: {provider}")
