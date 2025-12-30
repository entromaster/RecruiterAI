# RecruiterAI 🤖

AI-powered recruitment assistant that analyzes LinkedIn profiles and scores candidates against job descriptions using LLMs.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![React](https://img.shields.io/badge/React-18-61dafb)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## ✨ Features

- **LinkedIn Profile Scraping** - Extract candidate information from LinkedIn profiles
- **AI-Powered Scoring** - Score candidates against job requirements using Gemini or OpenAI
- **Modern Dashboard** - Beautiful React frontend with Three.js animations
- **MongoDB Storage** - Persist jobs, candidates, and scores in the cloud
- **RESTful API** - Flask backend with comprehensive endpoints

## 🖼️ Screenshots

*Dashboard with stats, recent activity, and quick actions*

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- MongoDB Atlas account (free tier works)
- Google Gemini API key (or OpenAI API key)

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/RecruiterAI.git
cd RecruiterAI
```

### 2. Backend Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env`:
```env
# Flask
SECRET_KEY=your-secret-key-here
DEBUG=True

# LinkedIn Credentials (for scraping)
LINKEDIN_EMAIL=your-linkedin-email
LINKEDIN_PASSWORD=your-linkedin-password

# LLM Providers (at least one required)
GOOGLE_API_KEY=your-gemini-api-key
OPENAI_API_KEY=your-openai-api-key
DEFAULT_LLM_PROVIDER=gemini

# MongoDB Atlas
MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/recruiter_ai?retryWrites=true&w=majority
```

### 4. Frontend Setup

```bash
cd frontend
npm install
```

### 5. Run the Application

**Terminal 1 - Backend:**
```bash
cd RecruiterAI
.venv\Scripts\activate  # or source .venv/bin/activate
python app.py
```

**Terminal 2 - Frontend:**
```bash
cd RecruiterAI/frontend
npm run dev
```

Open http://localhost:5173 in your browser.

## 📁 Project Structure

```
RecruiterAI/
├── app.py                  # Flask application entry point
├── config.py               # Configuration management
├── extensions.py           # MongoDB connection
├── requirements.txt        # Python dependencies
├── .env.example            # Environment template
│
├── models/
│   └── models.py           # MongoDB document models
│
├── routes/
│   ├── api.py              # REST API endpoints
│   └── health.py           # Health check endpoint
│
├── services/
│   ├── linkedin_service.py # LinkedIn scraping
│   ├── llm_service.py      # LLM provider abstraction
│   └── scoring_service.py  # Candidate scoring logic
│
└── frontend/               # React + Vite frontend
    ├── src/
    │   ├── components/     # Reusable components
    │   ├── pages/          # Page components
    │   └── services/       # API client
    └── package.json
```

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/jobs` | List all jobs |
| POST | `/api/jobs` | Create a job |
| GET | `/api/jobs/:id` | Get job with candidates |
| DELETE | `/api/jobs/:id` | Delete a job |
| GET | `/api/candidates` | List all candidates |
| POST | `/api/candidates/scrape` | Scrape LinkedIn profile |
| POST | `/api/score` | Score candidate against job |
| POST | `/api/quick-score` | Quick score (cached profiles) |
| GET | `/api/providers` | List available LLM providers |

## 🧪 Testing

Run the interactive test script:

```bash
python test_full_flow.py
```

This will:
1. Create/use a test job
2. Scrape a LinkedIn profile (opens browser for login)
3. Score the candidate using AI
4. Save results to MongoDB

## ⚙️ Configuration

### LLM Providers

- **Gemini** (default): Set `GOOGLE_API_KEY` and `DEFAULT_LLM_PROVIDER=gemini`
- **OpenAI**: Set `OPENAI_API_KEY` and `DEFAULT_LLM_PROVIDER=openai`

### MongoDB

Use MongoDB Atlas (free tier) or local MongoDB. Update `MONGO_URI` accordingly.

## 🛠️ Tech Stack

**Backend:**
- Flask + Flask-CORS
- PyMongo (MongoDB)
- Selenium + linkedin-scraper
- Google Generative AI / OpenAI

**Frontend:**
- React 18 + Vite
- Three.js + @react-three/fiber
- Framer Motion
- Lucide React Icons

## 📝 License

MIT License - feel free to use this project for learning and development.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

Built with ❤️ using AI-powered development
