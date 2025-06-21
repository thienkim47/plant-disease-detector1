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
# Dictionary đầy đủ thông tin về tất cả các loại bệnh
disease_info = {
    "Apple___Apple_scab": {
        "name": "Apple Scab", 
        "description": "Bệnh Apple Scab (đốm vảy) trên táo do nấm Venturia inaequalis gây ra, xuất hiện các đốm nâu/xám trên lá và quả, làm giảm năng suất và chất lượng quả. Đây là một trong những bệnh phổ biến nhất trên cây táo.",
        "treatment": "Cắt bỏ lá bệnh, phun thuốc gốc đồng hoặc thuốc chứa mancozeb, vệ sinh vườn cây. Tỉa cành để tăng thông thoáng. Sử dụng giống kháng bệnh."
    },
    "Apple___Cedar_apple_rust": {
        "name": "Apple Cedar Rust",
        "description": "Bệnh Apple Cedar Rust (rỉ sắt táo cedar) do nấm Gymnosporangium juniperi-virginianae gây ra, tạo đốm vàng cam trên lá táo. Bệnh này cần hai loại cây chủ để hoàn thành chu trình sống.",
        "treatment": "Loại bỏ cây bách xanh hoặc tuyết tùng gần vườn táo, phun thuốc diệt nấm, thu gom và tiêu hủy lá bệnh. Sử dụng giống táo kháng bệnh."
    },
    "Apple___healthy": {
        "name": "Healthy Apple",
        "description": "Lá táo khỏe mạnh, không có dấu hiệu bệnh. Lá có màu xanh tươi, không có đốm, vết loét hay biến màu bất thường.",
        "treatment": "Tiếp tục chăm sóc thông thường: tưới nước đầy đủ, bón phân theo lịch, theo dõi sự phát triển của cây và kiểm tra định kỳ để phát hiện sớm các dấu hiệu bệnh."
    },
    "Blueberry___healthy": {
        "name": "Healthy Blueberry",
        "description": "Lá việt quất khỏe mạnh, không có dấu hiệu bệnh. Lá có màu xanh bình thường, không có đốm hay biến đổi bất thường.",
        "treatment": "Tiếp tục chăm sóc thông thường: duy trì độ pH đất thích hợp (4.5-5.5), tưới nước đầy đủ, bón phân axit và theo dõi sự phát triển của cây."
    },
    "Cherry_(including_sour)___healthy": {
        "name": "Healthy Cherry",
        "description": "Lá anh đào (bao gồm anh đào chua) khỏe mạnh, không có dấu hiệu bệnh. Lá có màu xanh tươi, hình dạng bình thường.",
        "treatment": "Tiếp tục chăm sóc thông thường: tưới nước đều đặn, bón phân cân bằng, tỉa cành định kỳ và kiểm tra sức khỏe cây thường xuyên."
    },
    "Corn_(maize)___Common_rust_": {
        "name": "Corn Common Rust",
        "description": "Bệnh Corn Common Rust (rỉ sắt ngô) phổ biến do nấm Puccinia sorghi gây ra, tạo ra các đốm gỉ sắt màu nâu đỏ trên lá ngô. Bệnh phát triển mạnh trong điều kiện ẩm ướt.",
        "treatment": "Sử dụng giống kháng bệnh, phun thuốc diệt nấm khi cần thiết, luân canh cây trồng. Cải thiện thoát nước và tăng khoảng cách giữa các cây."
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "name": "Corn Northern Leaf Blight",
        "description": "Bệnh Corn Northern Leaf Blight (đốm lá phía bắc) trên ngô do nấm Exserohilum turcicum gây ra, tạo ra các vết đốm hình thoi màu xám hoặc nâu trên lá. Có thể làm giảm năng suất đáng kể.",
        "treatment": "Sử dụng giống kháng bệnh, phun thuốc diệt nấm, luân canh cây trồng, cày vùi tàn dư thực vật. Tránh tưới nước lên lá."
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "name": "Corn Gray Leaf Spot",
        "description": "Bệnh Corn Gray Leaf Spot (đốm xám lá ngô) do nấm Cercospora zeae-maydis gây ra, tạo ra các vết đốm hình chữ nhật màu xám hoặc nâu với viền rõ ràng. Phổ biến ở vùng khí hậu ẩm.",
        "treatment": "Luân canh cây trồng, sử dụng giống kháng bệnh, phun thuốc diệt nấm khi cần thiết, cải thiện thoát nước và tránh trồng quá dày."
    },
    "Grape___Black_rot": {
        "name": "Grape Black Rot",
        "description": "Bệnh Grape Black Rot do nấm Guignardia bidwellii gây ra, tạo ra các đốm tròn trên lá và quả rồi lan rộng. Quả bị nhiễm sẽ héo khô và chuyển màu đen.",
        "treatment": "Loại bỏ và tiêu hủy các phần bị nhiễm bệnh, phun thuốc diệt nấm, cắt tỉa để cải thiện lưu thông không khí. Vệ sinh vườn sau thu hoạch."
    },
    "Grape___Esca_(Black_Measles)": {
        "name": "Grape Esca (Black Measles)",
        "description": "Bệnh Esca do phức hợp nấm gây ra, tạo ra các đốm không đều trên lá nho, có thể dẫn đến héo cây. Là bệnh khó điều trị ở nho.",
        "treatment": "Cắt bỏ các cành bị nhiễm, sử dụng thuốc diệt nấm hệ thống, cải thiện dinh dưỡng cho cây. Tránh làm tổn thương thân cây."
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "name": "Grape Leaf Blight (Isariopsis Leaf Spot)",
        "description": "Bệnh Grape Leaf Blight (Isariopsis Leaf Spot) do nấm Isariopsis clavispora gây ra, tạo ra các đốm nâu trên lá, có thể lan rộng và làm lá khô cháy.",
        "treatment": "Phun thuốc diệt nấm chứa copper, cải thiện lưu thông không khí, tránh tưới nước lên lá. Thu gom và tiêu hủy lá bệnh."
    },
    "Grape___healthy": {
        "name": "Healthy Grape",
        "description": "Lá nho khỏe mạnh, không có dấu hiệu bệnh. Lá có màu xanh đặc trưng, hình dạng bình thường và không có đốm hay vết loét.",
        "treatment": "Tiếp tục chăm sóc thông thường: tưới nước hợp lý, bón phân cân bằng, tỉa cành để thông thoáng và kiểm tra sức khỏe cây định kỳ."
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "name": "Orange Huanglongbing (Citrus Greening)",
        "description": "Bệnh vàng lá cam (HLB) do vi khuẩn Candidatus Liberibacter gây ra, được truyền qua rệp muỗi. Là bệnh rất nguy hiểm với cây có múi.",
        "treatment": "Kiểm soát rệp muỗi vector, loại bỏ cây bệnh, sử dụng cây giống sạch bệnh. Hiện chưa có thuốc điều trị hiệu quả."
    },
    "Peach___Bacterial_spot": {
        "name": "Peach Bacterial Spot",
        "description": "Bệnh đốm vi khuẩn ở đào do vi khuẩn Xanthomonas arboricola gây ra, tạo ra các đốm nhỏ trên lá và quả đào, có thể làm quả nứt và biến dạng.",
        "treatment": "Phun thuốc kháng khuẩn gốc đồng, tỉa cành để tăng thông thoáng, tránh tưới nước lên lá. Sử dụng giống kháng bệnh."
    },
    "Peach___healthy": {
        "name": "Healthy Peach",
        "description": "Lá đào khỏe mạnh, không có dấu hiệu bệnh. Lá có màu xanh tươi, hình dạng bình thường và không có đốm hay vết loét.",
        "treatment": "Tiếp tục chăm sóc thông thường: tưới nước đầy đủ, bón phân theo nhu cầu, tỉa cành hợp lý và theo dõi sức khỏe cây thường xuyên."
    },
    "Pepper,_bell___Bacterial_spot": {
        "name": "Bell Pepper Bacterial Spot",
        "description": "Bệnh đốm vi khuẩn ở ớt chuông do vi khuẩn Xanthomonas campestris gây ra, tạo ra các đốm tròn nhỏ màu nâu trên lá và quả, có viền vàng xung quanh.",
        "treatment": "Sử dụng hạt giống không nhiễm bệnh, tưới nước ở gốc, luân canh cây trồng, phun thuốc kháng khuẩn gốc đồng. Tránh làm việc khi lá ướt."
    },
    "Pepper,_bell___healthy": {
        "name": "Healthy Bell Pepper",
        "description": "Lá ớt chuông khỏe mạnh, không có dấu hiệu bệnh. Lá có màu xanh đậm, bóng và không có đốm hay biến màu bất thường.",
        "treatment": "Tiếp tục chăm sóc thông thường: tưới nước đều đặn ở gốc, bón phân cân bằng, đảm bảo ánh sáng đầy đủ và kiểm tra sức khỏe cây định kỳ."
    },
    "Potato___Early_blight": {
        "name": "Potato Early Blight",
        "description": "Bệnh Early Blight ở khoai tây do nấm Alternaria solani gây ra, tạo ra các vết đốm nâu với các vòng đồng tâm trên lá già trước, sau đó lan sang lá non.",
        "treatment": "Luân canh cây trồng, tưới nước đầy đủ ở gốc, phun thuốc diệt nấm khi cần thiết, giữ khoảng cách hợp lý giữa các cây để tăng thông thoáng."
    },
    "Potato___Late_blight": {
        "name": "Potato Late Blight",
        "description": "Bệnh Late Blight ở khoai tây do nấm Phytophthora infestans gây ra, tạo ra các vết đốm nâu không đều trên lá và thân. Đây là bệnh rất nguy hiểm có thể phá hủy toàn bộ vườn khoai tây.",
        "treatment": "Phun thuốc diệt nấm phòng ngừa, tránh tưới nước quá nhiều, loại bỏ và tiêu hủy cây bệnh, tăng khoảng cách giữa các cây. Sử dụng giống kháng bệnh."
    },
    "Potato___healthy": {
        "name": "Healthy Potato",
        "description": "Lá khoai tây khỏe mạnh, không có dấu hiệu bệnh. Lá có màu xanh tươi, hình dạng bình thường và không có đốm hay vết loét.",
        "treatment": "Tiếp tục chăm sóc thông thường: tưới nước hợp lý, bón phân đầy đủ, vun gốc định kỳ và kiểm tra sức khỏe cây thường xuyên."
    },
    "Raspberry___healthy": {
        "name": "Healthy Raspberry",
        "description": "Lá mâm xôi khỏe mạnh, không có dấu hiệu bệnh. Lá có màu xanh đặc trưng, hình dạng răng cưa bình thường.",
        "treatment": "Tiếp tục chăm sóc thông thường: tưới nước đều đặn, bón phân hữu cơ, tỉa cành già và theo dõi sự phát triển của cây."
    },
    "Soybean___healthy": {
        "name": "Healthy Soybean",
        "description": "Lá đậu nành khỏe mạnh, không có dấu hiệu bệnh. Lá có màu xanh tươi, hình dạng ba lá đặc trưng và không có đốm hay biến màu.",
        "treatment": "Tiếp tục chăm sóc thông thường: tưới nước phù hợp, bón phân cân bằng, kiểm soát cỏ dại và theo dõi sự phát triển của cây."
    },
    "Squash___Powdery_mildew": {
        "name": "Squash Powdery Mildew",
        "description": "Bệnh phấn trắng ở bí ngô do nhiều loại nấm khác nhau gây ra, tạo ra lớp phấn trắng trên bề mặt lá, làm lá vàng và chết dần.",
        "treatment": "Tăng không gian giữa các cây, tưới nước ở gốc, phun dung dịch baking soda (1 muỗng cà phê/1 lít nước) hoặc thuốc diệt nấm hữu cơ. Cải thiện lưu thông không khí."
    },
    "Strawberry___Leaf_scorch": {
        "name": "Strawberry Leaf Scorch",
        "description": "Bệnh Leaf Scorch ở dâu tây do nấm Diplocarpon earlianum gây ra, tạo ra các đốm tím đỏ trên lá, sau đó lá chuyển màu nâu và khô cháy.",
        "treatment": "Loại bỏ lá bệnh, cải thiện thoát nước, tránh tưới nước lên lá, phun thuốc diệt nấm và sử dụng giống kháng bệnh."
    },
    "Strawberry___healthy": {
        "name": "Healthy Strawberry",
        "description": "Lá dâu tây khỏe mạnh, không có dấu hiệu bệnh. Lá có màu xanh tươi, hình dạng ba lá răng cưa đặc trưng.",
        "treatment": "Tiếp tục chăm sóc thông thường: tưới nước đều đặn ở gốc, bón phân hữu cơ, làm cỏ và theo dõi sự phát triển của cây."
    },
    "Tomato___Bacterial_spot": {
        "name": "Tomato Bacterial Spot",
        "description": "Bệnh đốm vi khuẩn ở cà chua do vi khuẩn Xanthomonas campestris pv. vesicatoria gây ra, tạo ra các đốm tròn nhỏ màu nâu trên lá và quả với viền vàng xung quanh.",
        "treatment": "Sử dụng hạt giống không nhiễm bệnh, tưới nước ở gốc, luân canh cây trồng, phun thuốc kháng khuẩn gốc đồng. Tránh làm việc khi cây ướt."
    },
    "Tomato___Early_blight": {
        "name": "Tomato Early Blight",
        "description": "Bệnh Early Blight ở cà chua do nấm Alternaria solani gây ra, tạo ra các vết đốm nâu với các vòng đồng tâm trên lá già, thường bắt đầu từ lá dưới cùng.",
        "treatment": "Loại bỏ lá bệnh, tưới nước ở gốc, giữ khoảng cách giữa các cây, phun thuốc diệt nấm khi cần thiết. Bón phân cân bằng để tăng sức đề kháng."
    },
    "Tomato___Late_blight": {
        "name": "Tomato Late Blight",
        "description": "Bệnh Late Blight ở cà chua do nấm Phytophthora infestans gây ra, tạo ra các vết đốm không đều, trông ướt trên lá và thân. Có thể phá hủy cây trong vài ngày.",
        "treatment": "Phun thuốc diệt nấm phòng ngừa, tránh tưới nước quá nhiều, loại bỏ và tiêu hủy cây bệnh, tăng khoảng cách giữa các cây. Sử dụng giống kháng bệnh."
    },
    "Tomato___Leaf_Mold": {
        "name": "Tomato Leaf Mold",
        "description": "Bệnh mốc lá cà chua do nấm Fulvia fulva gây ra, tạo ra các đốm vàng trên mặt trên của lá và nấm mốc xám-nâu ở mặt dưới. Phổ biến trong nhà kính.",
        "treatment": "Tăng lưu thông không khí, giảm độ ẩm, loại bỏ lá bệnh, phun thuốc diệt nấm khi cần thiết. Tránh tưới nước lên lá."
    },
    "Tomato___Septoria_leaf_spot": {
        "name": "Tomato Septoria Leaf Spot",
        "description": "Bệnh đốm lá Septoria trên cà chua do nấm Septoria lycopersici gây ra, tạo ra các đốm tròn nhỏ với viền sẫm màu và tâm màu xám trắng.",
        "treatment": "Loại bỏ lá bệnh, tưới nước ở gốc, phun thuốc diệt nấm, luân canh cây trồng. Tránh làm việc khi cây ướt."
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "name": "Tomato Spider Mites",
        "description": "Nhện đỏ hai chấm tấn công cà chua, tạo ra các đốm nhỏ vàng hoặc trắng trên lá do chúng hút chất dinh dưỡng từ tế bào lá. Lá có thể có mạng nhện mỏng.",
        "treatment": "Xịt nước áp lực mạnh lên lá, sử dụng xà phòng insecticidal, thuốc diệt côn trùng gốc dầu hoặc thuốc diệt côn trùng hữu cơ. Tăng độ ẩm xung quanh cây."
    },
    "Tomato___Target_Spot": {
        "name": "Tomato Target Spot",
        "description": "Bệnh đốm đồng tâm cà chua do nấm Corynespora cassiicola gây ra, tạo ra các đốm tròn với các vòng đồng tâm trên lá, thân và quả, giống như bia bắn.",
        "treatment": "Tưới nước ở gốc, giữ khoảng cách giữa các cây, loại bỏ lá bệnh, phun thuốc diệt nấm khi cần thiết. Cải thiện lưu thông không khí."
    },
    "Tomato___Tomato_mosaic_virus": {
        "name": "Tomato Mosaic Virus",
        "description": "Bệnh virus khảm cà chua tạo ra các mảng xen kẽ màu vàng và xanh trên lá, làm biến dạng lá và giảm năng suất. Virus lây truyền qua tiếp xúc cơ học.",
        "treatment": "Không có biện pháp chữa trị; loại bỏ và tiêu hủy cây bệnh, khử trùng dụng cụ, rửa tay thường xuyên khi làm việc với cây. Sử dụng giống kháng bệnh."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "name": "Tomato Yellow Leaf Curl Virus",
        "description": "Bệnh virus xoăn vàng lá cà chua làm cho lá bị xoăn, nhỏ lại và có màu vàng, cây còi cọc và ít ra hoa. Virus được truyền qua rệp bọ phấn trắng.",
        "treatment": "Kiểm soát bọ phấn trắng là vector truyền bệnh, sử dụng lưới chống côn trùng, trồng giống kháng bệnh, loại bỏ cây bệnh ngay khi phát hiện."
    },
    "Tomato___healthy": {
        "name": "Healthy Tomato",
        "description": "Lá cà chua khỏe mạnh, không có dấu hiệu bệnh. Lá có màu xanh đậm đặc trưng, hình dạng răng cưa bình thường và không có đốm hay vết loét.",
        "treatment": "Tiếp tục chăm sóc thông thường: tưới nước đều đặn ở gốc, bón phân cân bằng NPK, cắt cành phụ, dựng giàn và kiểm tra sức khỏe cây thường xuyên."
    }
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
    
    # Lấy thông tin bệnh từ dictionary
    if predicted_class_name in disease_info:
        info = disease_info[predicted_class_name]
    else:
        # Nếu không tìm thấy thông tin, tạo thông tin cơ bản
        info = {
            "name": predicted_class_name.replace('___', ' - ').replace('_', ' '),
            "description": "Đây là loại bệnh hoặc tình trạng lá cây được nhận diện bởi hệ thống.",
            "treatment": "Vui lòng tham khảo ý kiến chuyên gia nông nghiệp để có hướng dẫn điều trị cụ thể."
        }
    
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
                                  treatment=result["treatment"])
    
    return render_template('index.html')

@app.route('/analyze_webcam', methods=['POST'])
def analyze_webcam():
    # Đây là phần xử lý ảnh từ webcam
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
