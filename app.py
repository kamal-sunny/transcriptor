from flask import Flask, request, jsonify, send_from_directory
import whisper
import os
from datetime import datetime
import uuid
from pathlib import Path

app = Flask(__name__, static_folder='static')

# Configure paths
BASE_DIR = Path(__file__).parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
TRANSCRIPT_FOLDER = BASE_DIR / "transcripts"

# Create folders
UPLOAD_FOLDER.mkdir(exist_ok=True)
TRANSCRIPT_FOLDER.mkdir(exist_ok=True)

# Load model
model = whisper.load_model("base")

@app.route('/')
def home():
    return send_from_directory('static', 'index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({"error": "No file"}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    try:
        # Generate unique filename
        file_id = str(uuid.uuid4())
        file_ext = os.path.splitext(file.filename)[1]
        audio_path = UPLOAD_FOLDER / f"{file_id}{file_ext}"
        
        # Save file
        file.save(str(audio_path))
        
        # Transcribe
        result = model.transcribe(str(audio_path), language='en')
        clean_text = " ".join(result["text"].split())
        
        # Save transcript
        transcript_path = TRANSCRIPT_FOLDER / f"{file_id}.txt"
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(clean_text)
        
        return jsonify({
            "id": file_id,
            "text": clean_text,
            "filename": file.filename,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_transcript/<file_id>')
def get_transcript(file_id):
    try:
        transcript_path = TRANSCRIPT_FOLDER / f"{file_id}.txt"
        with open(transcript_path, 'r', encoding='utf-8') as f:
            return jsonify({"text": f.read()})
    except:
        return jsonify({"error": "Not found"}), 404

if __name__ == '__main__':
    app.run(debug=True, port=5000)
