from flask import Flask, request, jsonify
from personal_color_analysis import personal_color
from PIL import Image
import io
import numpy as np

app = Flask(__name__)

# 허용되는 파일 확장자
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/analyze', methods=['POST'])
def analyze_image():
    # 요청에서 파일 가져오기
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        # 파일을 메모리에서 읽기
        image_data = file.read()
        image = Image.open(io.BytesIO(image_data))
        # RGBA 이미지를 RGB로 변환 (필요시)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        # PIL 이미지를 OpenCV 형식으로 변환 (NumPy 배열)
        opencv_image = np.array(image)[:, :, ::-1]  # RGB -> BGR 변환

        # 이미지 객체를 분석
        try:
            # 이미지 객체를 분석하는 로직
            resultTone = personal_color.analysis(opencv_image)
            return jsonify({"tone": resultTone}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "File type not allowed"}), 400

if __name__ == '__main__':
    app.run(debug=True)