import os
import cv2
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from PIL import Image

# Flask app setup
app = Flask(__name__)
app.secret_key = "supersecretkey"  # Replace with a secure key
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Create folders if not exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def remove_chromakey(image_path, chroma_key=(0, 255, 0), tolerance=60):
    """
    Remove a chromakey color from an image.
    :param image_path: Path to the image file.
    :param chroma_key: Tuple with the BGR color to remove.
    :param tolerance: Tolerance for color matching.
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
    
    # Convert chroma key to NumPy array
    chroma_key = np.array(chroma_key, dtype=np.uint8)

    # Create mask for the chroma key color
    lower_bound = np.clip(chroma_key - tolerance, 0, 255)
    upper_bound = np.clip(chroma_key + tolerance, 0, 255)
    mask = cv2.inRange(image, lower_bound, upper_bound)

    # Apply mask to make the chroma key transparent
    image[mask != 0] = [0, 0, 0]  # Black background
    return image

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        chroma_key = request.form.get('chromakey', '0,255,0').split(',')
        chroma_key = tuple(map(int, chroma_key))
        tolerance = int(request.form.get('tolerance', 60))
        
        # Check if files were submitted
        if 'files[]' not in request.files:
            flash('No files uploaded!')
            return redirect(request.url)
        
        files = request.files.getlist('files[]')
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Process the image
                processed_image = remove_chromakey(filepath, chroma_key, tolerance)
                processed_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
                cv2.imwrite(processed_path, processed_image)
        flash('Images processed successfully!')
        return redirect(url_for('index'))

    return render_template('index.html')

@app.route('/processed/<filename>')
def processed_file(filename):
    return redirect(url_for('static', filename=f'processed/{filename}'))

if __name__ == '__main__':
    app.run(debug=True)
