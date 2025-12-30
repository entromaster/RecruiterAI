"""
Full Flow Test Script
=====================
Tests LinkedIn scraping + Gemini scoring end-to-end.

Run: python test_full_flow.py
"""

import requests
import json
import time
from services.linkedin_service import get_linkedin_service
from services.scoring_service import ScoringService
from models.models import JobModel, CandidateModel, ScoreModel
from config import Config

BASE_URL = "http://localhost:5000"


def test_scrape_and_score():
    print("=" * 60)
    print("   RECRUITER AI - FULL FLOW TEST")
    print("=" * 60)
    
    # Step 1: Get or create a test job
    print("\n📋 Step 1: Setting up test job...")
    jobs = JobModel.get_all()
    
    if jobs:
        job = jobs[0]
        print(f"   Using existing job: {job['title']}")
    else:
        job = JobModel.create(
            title="Full Stack Developer",
            company="TechCorp",
            description="""We are looking for a Full Stack Developer with:
            - 3+ years of experience with Python and JavaScript
            - Experience with React, Node.js, or similar frameworks
            - Database experience (MongoDB, PostgreSQL)
            - Strong problem-solving skills
            - Good communication skills"""
        )
        print(f"   Created new job: {job['title']}")
    
    print(f"   Job ID: {job['id']}")
    
    # Step 2: Get LinkedIn URL
    print("\n🔗 Step 2: LinkedIn Profile Scraping")
    default_url = "https://www.linkedin.com/in/satlouis/"
    linkedin_url = input(f"   Enter LinkedIn profile URL (or press Enter for default):\n   > ").strip()
    
    if not linkedin_url:
        linkedin_url = default_url
    
    print(f"   Target: {linkedin_url}")
    
    # Check if we already have this candidate
    existing = CandidateModel.get_by_linkedin_url(linkedin_url)
    if existing and existing.get("profile_data"):
        print(f"   ✅ Found existing profile: {existing.get('name')}")
        use_existing = input("   Use existing data? (y/n): ").strip().lower()
        if use_existing == 'y':
            candidate = existing
            profile_data = existing.get("profile_data", {})
        else:
            existing = None
    
    if not existing or not existing.get("profile_data"):
        print("\n   🌐 Opening browser for LinkedIn login...")
        print("   ⚠️  Complete any captcha or 2FA if prompted")
        
        try:
            linkedin_service = get_linkedin_service()
            linkedin_service.login()
            
            input("\n   Press Enter after login is complete...")
            
            print("   📥 Scraping profile...")
            profile_data = linkedin_service.scrape_profile(linkedin_url)
            
            # Save to database
            candidate = CandidateModel.upsert_by_url(
                linkedin_url=linkedin_url,
                name=profile_data.get("name"),
                about=profile_data.get("about"),
                current_title=profile_data.get("job_title"),
                current_company=profile_data.get("company"),
                profile_data=profile_data
            )
            
            print(f"   ✅ Scraped: {profile_data.get('name')}")
            print(f"   Title: {profile_data.get('job_title')}")
            print(f"   Company: {profile_data.get('company')}")
            
        except Exception as e:
            print(f"   ❌ Scraping failed: {e}")
            return
        finally:
            # Keep browser open for inspection if needed
            close = input("\n   Close browser? (y/n): ").strip().lower()
            if close == 'y':
                linkedin_service.close()
    
    # Step 3: Score with LLM
    provider = Config.DEFAULT_LLM_PROVIDER
    print(f"\n🤖 Step 3: Scoring with {provider.upper()}...")
    
    try:
        scoring_service = ScoringService(provider)
        result = scoring_service.score_candidate(profile_data, job["description"])
        
        # Save score to database
        score = ScoreModel.upsert(
            job_id=job["id"],
            candidate_id=candidate["id"],
            score=result["score"],
            analysis=result["analysis"],
            skill_matches=result["skill_matches"],
            recommendation=result["recommendation"],
            llm_provider=provider
        )
        
        print("\n" + "=" * 60)
        print("   SCORING RESULTS")
        print("=" * 60)
        print(f"\n   Candidate: {candidate.get('name')}")
        print(f"   Job: {job['title']}")
        print(f"\n   📊 Score: {result['score']}/100")
        print(f"   🏷️  Recommendation: {result['recommendation']}")
        print(f"\n   ✅ Skills Matched: {', '.join(result.get('skill_matches', []))}")
        print(f"   ❌ Skills Missing: {', '.join(result.get('skill_gaps', []))}")
        print(f"\n   📝 Analysis:\n   {result['analysis'][:500]}...")
        
        print("\n   ✅ Score saved to MongoDB!")
        
    except Exception as e:
        print(f"   ❌ Scoring failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    print("   TEST COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    test_scrape_and_score()
