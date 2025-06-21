import os
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Tạo thư mục uploads nếu chưa tồn tại
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model
MODEL_PATH = 'model/model_epoch_15 (1).h5'
model = load_model(MODEL_PATH)

# Load class mapping
with open('model/class_indices_moi.json', 'r') as f:
    class_indices = json.load(f)

# Tạo dictionary phân loại và thông tin bệnh
disease_info = {
    "Apple___Apple_scab": {
        "name": "Apple Scab",
        "description": "Bệnh đốm vảy trên táo do nấm Venturia inaequalis gây ra, xuất hiện các đốm nâu/xám trên lá và quả, làm giảm năng suất và chất lượng quả.",
        "treatment": "Cắt bỏ lá bệnh, phun thuốc gốc đồng hoặc thuốc chứa mancozeb, vệ sinh vườn cây."
    },
    "Apple___Cedar_apple_rust": {
        "name": "Apple Cedar Rust",
        "description": "Bệnh rỉ sắt táo cedar do nấm Gymnosporangium juniperi-virginianae gây ra, tạo đốm vàng cam trên lá táo.",
        "treatment": "Loại bỏ cây bách xanh hoặc tuyết tùng gần vườn táo, phun thuốc diệt nấm, thu gom và tiêu hủy lá bệnh."
    },
    "Apple___healthy": {
        "name": "Healthy Apple Leaf",
        "description": "Lá táo khỏe mạnh, không có dấu hiệu bệnh.",
        "treatment": "Tiếp tục chăm sóc thông thường và theo dõi sự phát triển của cây."
    },
}
# Đặt thông tin mặc định khi không tìm thấy thông tin bệnh
default_info = {
    "name": "Unknown Disease",
    "description": "Không có thông tin chi tiết về bệnh này.",
    "treatment": "Tham khảo ý kiến chuyên gia nông nghiệp."
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_disease(image_path):
    processed_image = preprocess_image(image_path)
    prediction = model.predict(processed_image)
    predicted_class_index = np.argmax(prediction[0])
    predicted_class_name = class_indices[str(predicted_class_index)]
    confidence = float(prediction[0][predicted_class_index])

    # Lấy thông tin bệnh
    info = disease_info.get(predicted_class_name, default_info)

    return {
        "class_name": predicted_class_name,
        "disease_name": info["name"],
        "description": info["description"],
        "treatment": info["treatment"],
        "confidence": confidence
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        # If user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Predict disease
            result = predict_disease(file_path)

            return render_template('result.html',
                                  filename=filename,
                                  disease_name=result["disease_name"],
                                  class_name=result["class_name"],
                                  description=result["description"],
                                  treatment=result["treatment"],
                                  confidence=result["confidence"] * 100)

    return render_template('index.html')

@app.route('/analyze_webcam', methods=['POST'])
def analyze_webcam():
    # Đây là phần xử lý ảnh từ webcam, sẽ được implement sau
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Predict disease
        result = predict_disease(file_path)

        return jsonify(result)

# @app.route('/analyze_esp32', methods=['POST'])
# def analyze_esp32():
#     # Phần này dành cho ESP32 camera, sẽ được implement sau khi cần
#     pass

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
