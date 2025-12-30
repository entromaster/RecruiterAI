"""
LinkedIn Scraper Test Script
============================
This script tests the linkedin_scraper library to understand its capabilities
and limitations with/without login.

Run: python test_scraper.py
"""

from linkedin_scraper import Person, actions
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import json
import os
from dotenv import load_dotenv

load_dotenv()

def setup_driver():
    """Set up Chrome driver with webdriver-manager (auto-downloads chromedriver)"""
    chrome_options = Options()
    # chrome_options.add_argument("--headless")  # Uncomment to run without visible browser
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver


def person_to_dict(person):
    """Convert Person object to a dictionary for display"""
    data = {
        "name": person.name,
        "about": person.about,
        "job_title": person.job_title,
        "company": person.company,
        "linkedin_url": person.linkedin_url,
        "experiences": [],
        "educations": [],
    }
    
    # Extract experiences
    if person.experiences:
        for exp in person.experiences:
            exp_data = {
                "title": getattr(exp, 'title', None) or getattr(exp, 'position_title', None),
                "company": getattr(exp, 'company', None) or getattr(exp, 'institution_name', None),
                "duration": getattr(exp, 'duration', None),
                "location": getattr(exp, 'location', None),
                "description": getattr(exp, 'description', None),
            }
            data["experiences"].append(exp_data)
    
    # Extract education
    if person.educations:
        for edu in person.educations:
            edu_data = {
                "institution": getattr(edu, 'institution_name', None) or getattr(edu, 'name', None),
                "degree": getattr(edu, 'degree', None),
                "field": getattr(edu, 'field_of_study', None),
                "dates": getattr(edu, 'dates', None),
            }
            data["educations"].append(edu_data)
    
    return data


def test_without_login(linkedin_url):
    """Test scraping without login - likely to fail for most profiles"""
    print("\n" + "="*60)
    print("TEST 1: Scraping WITHOUT Login")
    print("="*60)
    print(f"Target URL: {linkedin_url}")
    print("\nAttempting to scrape (this will likely fail or return limited data)...")
    
    driver = setup_driver()
    
    try:
        person = Person(linkedin_url, driver=driver, scrape=True, close_on_complete=False)
        data = person_to_dict(person)
        
        print("\n--- Results ---")
        print(json.dumps(data, indent=2, default=str))
        
        if data["name"] and data["experiences"]:
            print("\n✅ SUCCESS: Scraped data without login!")
        else:
            print("\n⚠️  LIMITED: Got minimal or no data without login")
            
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
    finally:
        input("\nPress Enter to close the browser...")
        driver.quit()


def test_with_login(linkedin_url):
    """Test scraping with login - should work better"""
    print("\n" + "="*60)
    print("TEST 2: Scraping WITH Login")
    print("="*60)
    
    # Get credentials from environment or prompt
    email = os.getenv("LINKEDIN_EMAIL")
    password = os.getenv("LINKEDIN_PASSWORD")
    
    if not email:
        email = input("Enter your LinkedIn email: ")
    if not password:
        password = input("Enter your LinkedIn password: ")
    
    print(f"\nTarget URL: {linkedin_url}")
    print("Logging in and scraping...")
    
    driver = setup_driver()
    
    try:
        # Login first
        actions.login(driver, email, password)
        print("✅ Login initiated...")
        
        # Wait for user to handle any 2FA or captcha
        input("\nIf there's a captcha or 2FA, complete it manually, then press Enter to continue...")
        
        # Now scrape
        person = Person(linkedin_url, driver=driver, scrape=True, close_on_complete=False)
        data = person_to_dict(person)
        
        print("\n--- Results ---")
        print(json.dumps(data, indent=2, default=str))
        
        if data["name"] and data["experiences"]:
            print("\n✅ SUCCESS: Scraped full profile data!")
        else:
            print("\n⚠️  LIMITED: Got minimal data even with login")
            
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
    finally:
        input("\nPress Enter to close the browser...")
        driver.quit()


def main():
    print("="*60)
    print("      LINKEDIN SCRAPER TEST")
    print("="*60)
    
    # Default test URL (public profile of a famous person)
    default_url = "https://www.linkedin.com/in/satlouis/"
    
    linkedin_url = input(f"\nEnter LinkedIn profile URL\n(or press Enter to use default: {default_url})\n> ").strip()
    
    if not linkedin_url:
        linkedin_url = default_url
    
    print("\nWhich test do you want to run?")
    print("1. Test WITHOUT login (quick test)")
    print("2. Test WITH login (requires your LinkedIn credentials)")
    print("3. Run both tests")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == "1":
        test_without_login(linkedin_url)
    elif choice == "2":
        test_with_login(linkedin_url)
    elif choice == "3":
        test_without_login(linkedin_url)
        test_with_login(linkedin_url)
    else:
        print("Invalid choice. Running test without login by default.")
        test_without_login(linkedin_url)


if __name__ == "__main__":
    main()
