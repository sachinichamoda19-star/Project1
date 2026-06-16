import os
import json
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# 1. Model එක සහ Class Names ලෝඩ් කිරීම
MODEL_PATH = 'fruit_model.keras'
model = tf.keras.models.load_model(MODEL_PATH)

with open('class_names.json', 'r') as f:
    class_names = json.load(f)

@app.route('/', sorted_methods=['GET'])
def index():
    # මුල් UI එක පෙන්වීම
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    if file:
        # 2. Upload කරපු ඉමේජ් එක තාවකාලිකව save කිරීම
        upload_dir = 'static/uploads'
        os.makedirs(upload_dir, exist_ok=True)
        filepath = os.path.join(upload_dir, file.filename)
        file.save(filepath)

        # 3. Image එක model එකට ගැලපෙන සේ සකස් කිරීම
        img = image.load_img(filepath, target_size=(180, 180))
        img_array = image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        # 4. Prediction එකක් ගැනීම
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]) * 100)

        # Result එක Frontend එකට යැවීම
        return jsonify({
            'prediction': predicted_class,
            'confidence': f"{confidence:.2f}%",
            'image_path': filepath
        })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
