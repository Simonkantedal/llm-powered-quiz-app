# 🧠 AI-Powered Quiz App

A modern web-based quiz application that uses **Google's Gemini AI** for intelligent, automated grading of free-text answers. Create custom quizzes, share them with others, and get instant, AI-powered feedback on responses.

## ✨ Features

- **🤖 AI-Powered Grading**: Uses Google Gemini to intelligently evaluate free-text answers
- **📝 Custom Quiz Creation**: Create quizzes with unlimited questions and ideal answers
- **🔗 Easy Sharing**: Share quizzes via simple quiz IDs
- **📊 Instant Results**: Get immediate scoring and feedback
- **🛡️ Secure & Robust**: Built with security best practices and comprehensive error handling
- **🎨 Clean UI**: Modern, responsive interface built with Tailwind CSS
- **⚡ Fast & Efficient**: Optimized database queries with proper indexing

## 🚀 Quick Start

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Google API key for Gemini

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd quiz-app
   ```

2. **Install dependencies**
   ```bash
   # Using uv (recommended)
   uv sync

   # Or using pip
   pip install -r requirements.txt
   ```

3. **Set up environment**
   ```bash
   cp .env.example .env
   # Edit .env and add your Google API key
   ```

4. **Run the application**
   ```bash
   uv run python main.py
   # Or: python main.py
   ```

5. **Open your browser**
   ```
   http://127.0.0.1:5001
   ```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | Your Google Gemini API key | Required |
| `GEMINI_MODEL` | Gemini model to use | `gemini-1.5-flash` |
| `DATABASE_PATH` | SQLite database file path | `quiz_app.db` |
| `DEBUG_MODE` | Enable debug mode | `false` |
| `HOST` | Server host | `127.0.0.1` |
| `PORT` | Server port | `5001` |

### Getting a Google API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create a new API key
3. Add it to your `.env` file as `GOOGLE_API_KEY`

## 📖 Usage

### Creating a Quiz

1. Click **"Create New Quiz"**
2. Add questions and their ideal answers
3. Click **"Create Quiz"** to get a unique Quiz ID
4. Share the Quiz ID with participants

### Taking a Quiz

1. Enter a Quiz ID on the main page
2. Click **"Start Quiz"**
3. Answer each question in the text areas
4. Submit to get your AI-graded score

### API Usage

The app provides REST endpoints for programmatic access:

```bash
# Create a quiz
curl -X POST http://localhost:5001/api/create_quiz \
  -H "Content-Type: application/json" \
  -d '{"questions":[{"question":"What is 2+2?","ideal_answer":"4"}]}'

# Get quiz questions
curl http://localhost:5001/api/get_quiz/abc123

# Submit answers
curl -X POST http://localhost:5001/api/submit_quiz/abc123 \
  -H "Content-Type: application/json" \
  -d '{"answers":["4"]}'
```

## 🏗️ Architecture

### Tech Stack

- **Backend**: Python Flask with SQLite database
- **Frontend**: Vanilla JavaScript with Tailwind CSS
- **AI**: Google Gemini for intelligent grading
- **Development**: Ruff for formatting/linting, pre-commit hooks

### Project Structure

```
quiz-app/
├── main.py              # Main Flask application
├── index.html           # Frontend UI
├── pyproject.toml       # Dependencies and configuration
├── .env.example         # Environment template
├── .pre-commit-config.yaml # Code quality hooks
└── README.md           # This file
```

### Database Schema

```sql
-- Quizzes table
CREATE TABLE quizzes (
    quiz_id TEXT PRIMARY KEY,
    questions TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Submissions table  
CREATE TABLE submissions (
    submission_id INTEGER PRIMARY KEY AUTOINCREMENT,
    quiz_id TEXT NOT NULL,
    answers TEXT NOT NULL,
    score INTEGER NOT NULL,
    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (quiz_id) REFERENCES quizzes (quiz_id)
);
```

## 🛠️ Development

### Setup Development Environment

```bash
# Install dev dependencies
uv sync --group dev

# Install pre-commit hooks (optional)
uv run pre-commit install
```

### Code Quality

```bash
# Format code
uv run ruff format .

# Lint and auto-fix
uv run ruff check --fix .

# Check everything (CI-ready)
uv run ruff check . && uv run ruff format --check .
```

### Testing

```bash
# Run the application
uv run python main.py

# Test API endpoints
curl -X POST http://localhost:5001/api/create_quiz \
  -H "Content-Type: application/json" \
  -d '{"questions":[{"question":"Test question","ideal_answer":"Test answer"}]}'
```

## 🚀 Production Deployment

### Using Gunicorn

```bash
# Install Gunicorn
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5001 main:app
```

### Environment Setup

1. Set `DEBUG_MODE=false` in production
2. Use a production database (PostgreSQL recommended)
3. Set up SSL certificates
4. Configure reverse proxy (Nginx)
5. Set up monitoring and logging

### Recommended Production Stack

- **Database**: PostgreSQL or MySQL
- **Web Server**: Nginx + Gunicorn
- **Platform**: Railway, Render, or cloud providers
- **Monitoring**: Sentry for error tracking

## 🔒 Security Features

- ✅ **SQL Injection Protection**: Parameterized queries
- ✅ **Input Validation**: Server-side validation for all inputs
- ✅ **CORS Configuration**: Proper cross-origin handling
- ✅ **Error Handling**: Comprehensive error logging without info leakage
- ✅ **Rate Limiting Ready**: Structured for easy rate limiting addition

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run code quality checks (`uv run ruff check . && uv run ruff format .`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: Check this README and code comments
- **Issues**: Report bugs via GitHub Issues
- **API**: Google Gemini API documentation at [ai.google.dev](https://ai.google.dev)

## 🔮 Roadmap

- [ ] User authentication and accounts
- [ ] Quiz analytics and reporting
- [ ] Multiple choice question support
- [ ] Team collaboration features
- [ ] Public quiz library
- [ ] Mobile app
- [ ] Advanced AI grading options

---

Built with ❤️ using Python, Flask, and Google Gemini AI