from flask import Flask, request, jsonify, send_from_directory
import whisper
import os
from datetime import datetime
import uuid
import logging

app = Flask(__name__, static_folder='static')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize model
try:
    model = whisper.load_model("base")
    logger.info("Whisper model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    model = None

# Ensure directories exist
os.makedirs('uploads', exist_ok=True)
os.makedirs('transcripts', exist_ok=True)

@app.route('/')
def home():
    return send_from_directory('static', 'index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if not model:
        return jsonify({"status": "error", "message": "Model not loaded"}), 500
        
    if 'audio' not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No file selected"}), 400

    try:
        # Create safe filename
        file_id = str(uuid.uuid4())
        file_ext = os.path.splitext(file.filename)[1]
        audio_filename = f"{file_id}{file_ext}"
        audio_path = os.path.join('uploads', audio_filename)
        
        # Save file
        file.save(audio_path)
        logger.info(f"File saved to: {audio_path}")
        
        # Verify file exists
        if not os.path.exists(audio_path):
            logger.error("File save verification failed")
            return jsonify({"status": "error", "message": "File save failed"}), 500

        # Transcribe with detailed logging
        logger.info("Starting transcription...")
        result = model.transcribe(audio_path, language='en')
        logger.info("Transcription completed")
        
        clean_text = " ".join(result["text"].split())
        logger.info(f"Transcript length: {len(clean_text)} characters")
        
        # Save transcript
        transcript_filename = f"{file_id}.txt"
        transcript_path = os.path.join('transcripts', transcript_filename)
        
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(clean_text)
        logger.info(f"Transcript saved to: {transcript_path}")
        
        return jsonify({
            "status": "success",
            "id": file_id,
            "filename": file.filename,
            "text": clean_text,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
    except Exception as e:
        logger.error(f"Error in transcription: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/get_transcript/<file_id>')
def get_transcript(file_id):
    try:
        transcript_path = os.path.join('transcripts', f"{file_id}.txt")
        logger.info(f"Attempting to load: {transcript_path}")
        
        if not os.path.exists(transcript_path):
            logger.error("Transcript file not found")
            return jsonify({"status": "error", "message": "Transcript not found"}), 404
            
        with open(transcript_path, 'r', encoding='utf-8') as f:
            content = f.read()
            logger.info(f"Successfully loaded transcript: {len(content)} characters")
            return jsonify({
                "status": "success", 
                "text": content,
                "filename": f"{file_id}.txt"
            })
            
    except Exception as e:
        logger.error(f"Error loading transcript: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)