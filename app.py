import os
import pathlib
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from flask import Flask, url_for, redirect,  render_template, request


# 取得目前檔案所在的資料夾 
SRC_PATH =  pathlib.Path(__file__).parent.absolute()
# 結合目前的檔案路徑和static及uploads路徑 
UPLOAD_FOLDER = os.path.join(SRC_PATH,  'static', 'uploads')
model = os.path.join(SRC_PATH, 'model1.h5')
def run_model(model,UPLOAD_FOLDER):
    model = load_model(model)
    img_path = os.path.join(UPLOAD_FOLDER,  ''.join(os.listdir(UPLOAD_FOLDER)))
    class_names = ['calculus','gingivitis','original','toothdiscoloration','ulcer']#電腦資料夾名稱
    img = image.load_img(img_path, target_size=(64, 64))  # 預測模型可能需要特定大小的輸入
    img = image.img_to_array(img)
    predictions = model.predict(np.array([img]))  # Vector of probabilities
    max_index = np.argmax(predictions)
    # 這部分你可以根據你的需求進行後續的處理，例如印出前幾名的預測結果
    return class_names[max_index]


app = Flask(__name__)
@app.route('/', methods=['GET'])
def index():
    return render_template('main.html')

@app.route('/', methods=['POST'])
def upload_file():
    file = request.files['filename']
    if file.filename != '':
        file.save(os.path.join(UPLOAD_FOLDER, file.filename))
    return redirect(url_for('result'))

@app.route('/result', methods=['GET'])
def result():
    result = str(run_model(model,UPLOAD_FOLDER))
    img_path = os.path.join(UPLOAD_FOLDER,  ''.join(os.listdir(UPLOAD_FOLDER)))
    return render_template('result.html',result=result,url=img_path)

@app.route('/result', methods=['POST'])
def goback():
    img_path = os.path.join(UPLOAD_FOLDER,  ''.join(os.listdir(UPLOAD_FOLDER)))
    os.remove(img_path)
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run()
