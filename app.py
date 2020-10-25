import os
import numpy as np
import io
from flask import Flask, render_template, url_for, request, send_file
from werkzeug.utils import secure_filename
import cv2

from PIL import Image, ImageFile
import matplotlib.pyplot as plt

from io import BytesIO
import base64

import detect

app = Flask(__name__)
net = detect.load_model(model_name="u2netp")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/Prediction', methods=['GET', 'POST'])
def pred():
    if request.method == 'POST':
        data = request.files['file'].read()
        img = Image.open(io.BytesIO(data))
        output = detect.predict(net, np.array(img))
        output = output.resize((img.size), resample=Image.BILINEAR)  # remove resample

        empty_img = Image.new("RGBA", (img.size), 0)
        new_img = Image.composite(img, empty_img, output.convert("L"))

        # Rendering Predicted Image
        a = np.asarray(new_img)
        im = Image.fromarray(a)
        file_obj = BytesIO()
        im.save(file_obj, 'PNG')
        file_obj.seek(0)
        predicted = base64.b64encode(file_obj.getvalue()).decode('utf8')

        # Rendering Original image
        arr = np.array(img)
        # convert numpy array to PIL Image
        img_o = Image.fromarray(arr.astype('uint8'))
        # create file-object in memory
        file_object = io.BytesIO()
        # write PNG in file-object
        img_o.save(file_object, 'PNG')
        file_object.seek(0)
        img_base64 = base64.b64encode(file_object.getvalue()).decode('utf8')

        return render_template('show.html', plot_url0=predicted, base64img = img_base64)


@app.route('/PredictedFirst', methods=['GET', 'POST'])
def One():
    return render_template('pOne.html')


@app.route('/PredictedSecond', methods=['GET', 'POST'])
def Two():
    return render_template('pTwo.html')


@app.route('/PredictedThird', methods=['GET', 'POST'])
def Three():
    return render_template('pThree.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
