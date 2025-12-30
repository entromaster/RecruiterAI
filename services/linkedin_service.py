"""
LinkedIn Scraper Service
========================
Wraps the linkedin_scraper library for profile extraction.
"""

from linkedin_scraper import Person, actions
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time

from config import Config


class LinkedInService:
    """Service for scraping LinkedIn profiles"""
    
    def __init__(self):
        self.driver = None
        self.is_logged_in = False
    
    def _setup_driver(self, headless=False):
        """Set up Chrome driver with webdriver-manager"""
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        return self.driver
    
    def login(self, email=None, password=None):
        """Login to LinkedIn"""
        if not self.driver:
            self._setup_driver(headless=False)
        
        email = email or Config.LINKEDIN_EMAIL
        password = password or Config.LINKEDIN_PASSWORD
        
        if not email or not password:
            raise ValueError("LinkedIn credentials not provided")
        
        actions.login(self.driver, email, password)
        self.is_logged_in = True
        
        # Give time for login to complete
        time.sleep(3)
        
        return True
    
    def scrape_profile(self, linkedin_url: str) -> dict:
        """
        Scrape a LinkedIn profile and return structured data.
        
        Args:
            linkedin_url: Full LinkedIn profile URL
            
        Returns:
            Dictionary with profile data
        """
        import traceback
        
        if not self.driver:
            self._setup_driver()
        
        if not self.is_logged_in:
            self.login()
        
        try:
            # Navigate to profile first
            self.driver.get(linkedin_url)
            time.sleep(3)  # Wait for page to load
            
            # Try using Person with scrape=False first, then manually scrape
            try:
                person = Person(linkedin_url, driver=self.driver, scrape=False, close_on_complete=False)
                person.scrape(close_on_complete=False)
            except Exception as e1:
                # If that fails, try direct scraping approach
                print(f"Person scrape failed: {e1}")
                traceback.print_exc()
                
                # Fallback: manually extract basic info from page
                return self._manual_scrape(linkedin_url)
            
            return self._person_to_dict(person)
        except Exception as e:
            traceback.print_exc()
            raise Exception(f"Failed to scrape profile: {str(e)}")
    
    def _manual_scrape(self, linkedin_url: str) -> dict:
        """Fallback manual scraping when library fails"""
        from selenium.webdriver.common.by import By
        
        data = {
            "name": None,
            "about": None,
            "job_title": None,
            "company": None,
            "linkedin_url": linkedin_url,
            "experiences": [],
            "educations": [],
            "skills": [],
        }
        
        try:
            # Try to get name
            name_elem = self.driver.find_elements(By.CSS_SELECTOR, "h1.text-heading-xlarge")
            if name_elem:
                data["name"] = name_elem[0].text
            
            # Try to get headline (job title)
            headline = self.driver.find_elements(By.CSS_SELECTOR, "div.text-body-medium")
            if headline:
                data["job_title"] = headline[0].text
            
            # Try to get about
            about = self.driver.find_elements(By.CSS_SELECTOR, "div.pv-shared-text-with-see-more")
            if about:
                data["about"] = about[0].text
                
        except Exception as e:
            print(f"Manual scrape partial failure: {e}")
        
        return data
    
    def _person_to_dict(self, person) -> dict:
        """Convert Person object to a dictionary"""
        data = {
            "name": getattr(person, 'name', None),
            "about": getattr(person, 'about', None),
            "job_title": getattr(person, 'job_title', None),
            "company": getattr(person, 'company', None),
            "linkedin_url": getattr(person, 'linkedin_url', None),
            "experiences": [],
            "educations": [],
            "skills": [],
        }
        
        # Safely extract skills
        try:
            skills = getattr(person, 'skills', [])
            if skills:
                data["skills"] = list(skills) if hasattr(skills, '__iter__') and not isinstance(skills, str) else []
        except Exception:
            pass
        
        # Extract experiences with error handling
        try:
            experiences = getattr(person, 'experiences', [])
            if experiences:
                for exp in experiences:
                    try:
                        # Handle different experience formats
                        if isinstance(exp, dict):
                            exp_data = exp
                        else:
                            exp_data = {
                                "title": getattr(exp, 'title', None) or getattr(exp, 'position_title', None),
                                "company": getattr(exp, 'company', None) or getattr(exp, 'institution_name', None),
                                "duration": getattr(exp, 'duration', None),
                                "location": getattr(exp, 'location', None),
                                "description": getattr(exp, 'description', None),
                            }
                        data["experiences"].append(exp_data)
                    except Exception:
                        continue
        except Exception:
            pass
        
        # Extract education with error handling
        try:
            educations = getattr(person, 'educations', [])
            if educations:
                for edu in educations:
                    try:
                        # Handle different education formats
                        if isinstance(edu, dict):
                            edu_data = edu
                        else:
                            edu_data = {
                                "institution": getattr(edu, 'institution_name', None) or getattr(edu, 'name', None),
                                "degree": getattr(edu, 'degree', None),
                                "field": getattr(edu, 'field_of_study', None),
                                "dates": getattr(edu, 'dates', None),
                            }
                        data["educations"].append(edu_data)
                    except Exception:
                        continue
        except Exception:
            pass
        
        return data
    
    def close(self):
        """Close the browser"""
        if self.driver:
            self.driver.quit()
            self.driver = None
            self.is_logged_in = False


# Singleton instance for reuse
_linkedin_service = None


def get_linkedin_service() -> LinkedInService:
    """Get or create the LinkedIn service singleton"""
    global _linkedin_service
    if _linkedin_service is None:
        _linkedin_service = LinkedInService()
    return _linkedin_service
