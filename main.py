import os
import sqlite3
import json
import uuid
import logging
from typing import Dict, List, Any, Optional, Tuple
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('quiz_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

DATABASE = os.getenv('DATABASE_PATH', 'quiz_app.db')
DEBUG_MODE = os.getenv('DEBUG_MODE', 'False').lower() == 'true'
PORT = int(os.getenv('PORT', '5001'))
HOST = os.getenv('HOST', '127.0.0.1')

# Placeholder for LLM call. Replace with your actual implementation.
def call_llm_for_grading(question: str, ideal_answer: str, user_answer: str) -> int:
    """
    Placeholder function for LLM-based grading.
    
    Currently returns 1 (correct) for all answers as a placeholder.
    Replace this with actual LLM integration (OpenAI, Gemini, Anthropic, etc.).
    
    Args:
        question: The quiz question text
        ideal_answer: The expected correct answer
        user_answer: The user's submitted answer
    
    Returns:
        int: 1 for correct answer, 0 for incorrect
    """
    try:
        logger.info(f"Grading question: {question[:50]}...")
        
        # TODO: Replace with actual LLM implementation
        # Example implementation structure:
        # prompt = f"Grade this answer. Question: '{question}' Expected: '{ideal_answer}' User answer: '{user_answer}' Return 1 for correct, 0 for incorrect."
        # response = llm_client.generate(prompt)
        # return int(response.strip()) if response.strip() in ['0', '1'] else 0
        
        # Placeholder: always return correct (1) for testing
        # You can change this logic temporarily for testing
        return 1
        
    except Exception as e:
        logger.error(f"Error in LLM grading: {e}")
        return 0


def get_db() -> sqlite3.Connection:
    """
    Get database connection with row factory for easier access.
    
    Returns:
        sqlite3.Connection: Database connection object
    """
    db = sqlite3.connect(DATABASE)
    db.row_factory = sqlite3.Row
    return db

def init_db() -> None:
    """
    Initialize the database with required tables and indexes.
    """
    with app.app_context():
        db = get_db()
        cursor = db.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quizzes (
                quiz_id TEXT PRIMARY KEY,
                questions TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS submissions (
                submission_id INTEGER PRIMARY KEY AUTOINCREMENT,
                quiz_id TEXT NOT NULL,
                answers TEXT NOT NULL,
                score INTEGER NOT NULL,
                submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (quiz_id) REFERENCES quizzes (quiz_id)
            )
        ''')
        
        # Create indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_quiz_id ON submissions(quiz_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON quizzes(created_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_submitted_at ON submissions(submitted_at)')
        
        db.commit()
        logger.info("Database initialized successfully")

def validate_quiz_data(data: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate quiz creation data.
    
    Args:
        data: Request data dictionary
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not data or 'questions' not in data:
        return False, "Invalid data: missing 'questions' field"
    
    questions = data['questions']
    if not isinstance(questions, list) or len(questions) == 0:
        return False, "Questions must be a non-empty list"
    
    if len(questions) > 50:  # Reasonable limit
        return False, "Too many questions (max 50)"
    
    for i, question in enumerate(questions):
        if not isinstance(question, dict):
            return False, f"Question {i+1} must be an object"
        
        if 'question' not in question or 'ideal_answer' not in question:
            return False, f"Question {i+1} missing required fields"
        
        if not isinstance(question['question'], str) or not question['question'].strip():
            return False, f"Question {i+1} text is invalid"
        
        if not isinstance(question['ideal_answer'], str) or not question['ideal_answer'].strip():
            return False, f"Question {i+1} ideal answer is invalid"
        
        if len(question['question']) > 1000:
            return False, f"Question {i+1} text too long (max 1000 chars)"
        
        if len(question['ideal_answer']) > 2000:
            return False, f"Question {i+1} ideal answer too long (max 2000 chars)"
    
    return True, ""

def validate_quiz_submission(data: Dict[str, Any], expected_count: int) -> Tuple[bool, str]:
    """
    Validate quiz submission data.
    
    Args:
        data: Request data dictionary
        expected_count: Expected number of answers
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not data or 'answers' not in data:
        return False, "Invalid submission data: missing 'answers' field"
    
    answers = data['answers']
    if not isinstance(answers, list):
        return False, "Answers must be a list"
    
    if len(answers) != expected_count:
        return False, f"Expected {expected_count} answers, got {len(answers)}"
    
    for i, answer in enumerate(answers):
        if not isinstance(answer, str):
            return False, f"Answer {i+1} must be a string"
        
        if len(answer) > 5000:
            return False, f"Answer {i+1} too long (max 5000 chars)"
    
    return True, ""

@app.route('/')
def serve_index():
    """
    Serve the main index.html file.
    """
    return send_from_directory('.', 'index.html')

@app.route('/api/create_quiz', methods=['POST'])
def create_quiz():
    """
    Create a new quiz with validated questions.
    
    Returns:
        JSON response with quiz_id or error message
    """
    try:
        data = request.get_json()
        
        # Validate input
        is_valid, error_msg = validate_quiz_data(data)
        if not is_valid:
            logger.warning(f"Invalid quiz creation attempt: {error_msg}")
            return jsonify({"error": error_msg}), 400

        quiz_id = str(uuid.uuid4())[:8]
        questions = json.dumps(data['questions'])

        db = get_db()
        cursor = db.cursor()
        cursor.execute(
            'INSERT INTO quizzes (quiz_id, questions) VALUES (?, ?)', 
            (quiz_id, questions)
        )
        db.commit()
        
        logger.info(f"Quiz created successfully: {quiz_id}")
        return jsonify({"quiz_id": quiz_id})
        
    except Exception as e:
        logger.error(f"Error creating quiz: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/get_quiz/<quiz_id>', methods=['GET'])
def get_quiz(quiz_id: str):
    """
    Retrieve quiz questions by quiz ID (without ideal answers).
    
    Args:
        quiz_id: The unique quiz identifier
    
    Returns:
        JSON response with questions or error message
    """
    try:
        # Validate quiz_id format
        if not quiz_id or len(quiz_id) > 50:  # Reasonable limit
            return jsonify({"error": "Invalid quiz ID"}), 400
        
        db = get_db()
        cursor = db.cursor()
        cursor.execute(
            'SELECT questions FROM quizzes WHERE quiz_id = ?', 
            (quiz_id,)
        )
        row = cursor.fetchone()

        if row:
            questions_data = json.loads(row['questions'])
            # Return only the questions, not the ideal answers
            questions_for_user = [{"question": q["question"]} for q in questions_data]
            logger.info(f"Quiz retrieved: {quiz_id}")
            return jsonify({"questions": questions_for_user})
        else:
            logger.warning(f"Quiz not found: {quiz_id}")
            return jsonify({"error": "Quiz not found"}), 404
            
    except json.JSONDecodeError:
        logger.error(f"Corrupted quiz data for ID: {quiz_id}")
        return jsonify({"error": "Corrupted quiz data"}), 500
    except Exception as e:
        logger.error(f"Error retrieving quiz {quiz_id}: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/submit_quiz/<quiz_id>', methods=['POST'])
def submit_quiz(quiz_id: str):
    """
    Submit quiz answers for grading.
    
    Args:
        quiz_id: The unique quiz identifier
    
    Returns:
        JSON response with score or error message
    """
    try:
        # Validate quiz_id format
        if not quiz_id or len(quiz_id) > 50:
            return jsonify({"error": "Invalid quiz ID"}), 400
        
        data = request.get_json()
        
        db = get_db()
        cursor = db.cursor()
        cursor.execute(
            'SELECT questions FROM quizzes WHERE quiz_id = ?', 
            (quiz_id,)
        )
        row = cursor.fetchone()

        if not row:
            logger.warning(f"Quiz submission attempt for non-existent quiz: {quiz_id}")
            return jsonify({"error": "Quiz not found"}), 404

        stored_questions = json.loads(row['questions'])
        total_questions = len(stored_questions)
        
        # Validate submission data
        is_valid, error_msg = validate_quiz_submission(data, total_questions)
        if not is_valid:
            logger.warning(f"Invalid quiz submission for {quiz_id}: {error_msg}")
            return jsonify({"error": error_msg}), 400
        
        user_answers = data['answers']
        score = 0
        
        for i, stored_q in enumerate(stored_questions):
            question_text = stored_q['question']
            ideal_answer_text = stored_q['ideal_answer']
            user_answer_text = user_answers[i]

            # Grade the answer using the LLM
            grade = call_llm_for_grading(question_text, ideal_answer_text, user_answer_text)
            if grade == 1:
                score += 1

        # Store the submission
        answers_json = json.dumps(user_answers)
        cursor.execute(
            'INSERT INTO submissions (quiz_id, answers, score) VALUES (?, ?, ?)',
            (quiz_id, answers_json, score)
        )
        db.commit()
        
        logger.info(f"Quiz submitted: {quiz_id}, Score: {score}/{total_questions}")
        return jsonify({"score": score, "total": total_questions})
        
    except json.JSONDecodeError:
        logger.error(f"Corrupted quiz data for submission: {quiz_id}")
        return jsonify({"error": "Corrupted quiz data"}), 500
    except Exception as e:
        logger.error(f"Error submitting quiz {quiz_id}: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    init_db()
    logger.info(f"Starting quiz app on {HOST}:{PORT} (debug={DEBUG_MODE})")
    # For production, use a WSGI server like Gunicorn or Waitress
    # Example: gunicorn -w 4 -b 0.0.0.0:5001 main:app
    app.run(debug=DEBUG_MODE, host=HOST, port=PORT)