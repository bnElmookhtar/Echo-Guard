import os
import tensorflow as tf
import librosa
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_from_directory

# Initialize Flask app
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the pre-trained model
model_path = os.path.join(os.path.dirname(__file__), 'model', 'VGG16_v2.h5')
if not os.path.exists(model_path):
    raise FileNotFoundError("Model file not found. Please check the path.")
model = tf.keras.models.load_model(model_path)


# Allowed file check
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Preprocess audio
def preprocess_audio(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=22050)
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=512, n_mels=224)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize values
        mel_spec_db = librosa.util.normalize(mel_spec_db)

        # Resize to (224, 224)
        mel_spec_db = librosa.util.fix_length(mel_spec_db, size=224, axis=0)
        mel_spec_db = librosa.util.fix_length(mel_spec_db, size=224, axis=1)

        # Add channel dimensions
        mel_spec_db = np.expand_dims(mel_spec_db, axis=(0, -1))
        mel_spec_db = np.repeat(mel_spec_db, 3, axis=-1)  # Convert single-channel to 3-channel

        return mel_spec_db
    except Exception as e:
        print(f"Error processing the audio file: {e}")
        return None

# Render the home page with file upload form
@app.route('/')
def index():
    return render_template('index.html')

# Handle file upload and render the uploaded confirmation page
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        # Save the file
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Render confirmation page
        return render_template('uploaded.html', filename=filename)
    else:
        return "Invalid file type. Only .wav and .mp3 files are allowed."

# Serve the uploaded file for playback
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Make a prediction and render the results page with a confidence percentage
@app.route('/predict/<filename>')
def predict(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    processed_audio = preprocess_audio(file_path)

    if processed_audio is not None:
        # Get model prediction
        prediction = model.predict(processed_audio)
        real_confidence = float(prediction[0][0])  # Confidence score for "Real" class
        fake_confidence = float(prediction[0][1])  # Confidence score for "Fake" class

        # Determine the label and corresponding confidence
        if fake_confidence > real_confidence:
            label = "Fake"
            percentage = fake_confidence * 100
        else:
            label = "Real"
            percentage = real_confidence * 100

        # Debugging output
        print(f"Predicted Label: {label}, Confidence: {percentage:.2f}%")
    else:
        label = "Error"
        percentage = 0.0

    return render_template('result.html', filename=filename, label=label, percentage=percentage)


if __name__ == '__main__':
    app.run(debug=True)
