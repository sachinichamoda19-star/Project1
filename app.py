import os
from flask import Flask, request, jsonify, render_template_string
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# ==========================================
# 1. MODEL & CONFIGURATION
# ==========================================
# Oyage .h5 model eke nama methana danna (Model file eka me file eka thiyena folder ekatama danna)
MODEL_PATH = 'fruit_model.h5'
model = load_model(MODEL_PATH)

# Oyage model eke thiyana fruit classes tika me order ekatama danna
CLASS_NAMES = ['Apple', 'Banana', 'Mango', 'Orange', 'Strawberry'] 

UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def predict_fruit(img_path):
    # Oyage model eka train karapu size ekata target_size eka wenas karanna (e.g., 224x224, 150x150)
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale karala train kala nam pamanak meka danna

    predictions = model.predict(img_array)
    highest_match_index = np.argmax(predictions[0])
    
    fruit_name = CLASS_NAMES[highest_match_index]
    confidence = float(predictions[0][highest_match_index] * 100)
    
    return fruit_name, round(confidence, 2)


# ==========================================
# 2. FRONTEND HTML CODE (String ekak widihata)
# ==========================================
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fruit Classifier</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f0f4f8;
            margin: 0; padding: 0;
            display: flex; justify-content: center; align-items: center;
            min-height: 100vh;
        }
        .container {
            background: white; padding: 30px; border-radius: 15px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.08); width: 400px; text-align: center;
        }
        h2 { color: #2c3e50; margin-bottom: 25px; }
        .upload-box {
            border: 2px dashed #3498db; padding: 20px; border-radius: 10px;
            background-color: #fbfcfd; cursor: pointer; margin-bottom: 20px;
        }
        input[type="file"] { display: none; }
        .btn {
            background-color: #2ecc71; color: white; border: none;
            padding: 12px 25px; font-size: 16px; border-radius: 8px;
            cursor: pointer; width: 100%; font-weight: bold;
        }
        .btn:hover { background-color: #27ae60; }
        #preview {
            max-width: 100%; height: 200px; object-fit: cover;
            border-radius: 8px; margin-top: 15px; display: none;
        }
        #result-box {
            margin-top: 25px; padding: 15px; border-radius: 8px;
            background-color: #e8f8f5; display: none; border-left: 5px solid #2ecc71;
        }
        .loading { display: none; color: #7f8c8d; margin-top: 15px; }
    </style>
</head>
<body>

<div class="container">
    <h2>Fruit Classifier AI</h2>
    
    <form id="upload-form" enctype="multipart/form-data">
        <div class="upload-box" onclick="document.getElementById('image-input').click()">
            <p style="margin:0; color:#7f8c8d;">Click here to upload Fruit Image</p>
            <input type="file" id="image-input" name="file" accept="image/*" onchange="previewImage(event)">
            <img id="preview" src="" alt="Preview">
        </div>
        <button type="submit" class="btn">Classify Fruit</button>
    </form>

    <div class="loading" id="loader">Processing Image... Please wait...</div>

    <div id="result-box">
        <h3 style="margin:0; color:#16a085;">Result: <span id="fruit-res">None</span></h3>
        <p style="margin:5px 0 0 0; color:#7f8c8d;">Accuracy: <span id="accuracy-res">0</span>%</p>
    </div>
</div>

<script>
    function previewImage(event) {
        var reader = new FileReader();
        reader.onload = function() {
            var output = document.getElementById('preview');
            output.src = reader.result;
            output.style.display = 'block';
        }
        reader.readAsDataURL(event.target.files[0]);
    }

    document.getElementById('upload-form').onsubmit = async function(e) {
        e.preventDefault();
        let fileInput = document.getElementById('image-input').files[0];
        if(!fileInput) { alert("Please select an image first!"); return; }

        let formData = new FormData();
        formData.append('file', fileInput);

        document.getElementById('loader').style.display = 'block';
        document.getElementById('result-box').style.display = 'none';

        let response = await fetch('/predict', { method: 'POST', body: formData });
        let data = await response.json();
        document.getElementById('loader').style.display = 'none';

        if(data.error) {
            alert(data.error);
        } else {
            document.getElementById('fruit-res').innerText = data.fruit;
            document.getElementById('accuracy-res').innerText = data.accuracy;
            document.getElementById('result-box').style.display = 'block';
        }
    }
</script>
</body>
</html>
"""

# ==========================================
# 3. FLASK ROUTING
# ==========================================
@app.route('/', methods=['GET'])
def index():
    # Templates folder ekak nathuwa string ekak kelinma render karanawa
    return render_template_string(html_content)

@app.route('/predict', methods=['POST'])
def upload_and_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Prediction logic
        fruit, accuracy = predict_fruit(file_path)

        return jsonify({
            'fruit': fruit,
            'accuracy': accuracy
        })

if __name__ == '__main__':
    app.run(debug=True)