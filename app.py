from crypt import methods
from flask import Flask, request
from flask_cors import CORS
import torch
import numpy as np
from PIL import Image

app = Flask(__name__)
CORS(app)

model = torch.jit.load('densenet161.zip')

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/resnet50')
def resnet50():
    return 'Resnet50!'

@app.route('/densenet161', methods=['POST'])
def densenet161():
    #load image
    img = Image.open(request.files['file'].stream).convert('RGB').resize((32, 32))
    img = np.array(img)
    img = torch.FloatTensor(img.transpose(2, 0, 1) / 255)
    img = img.unsqueeze(0)
    img = torch.cat([torch.zeros(1, img.shape[1], img.shape[2], img.shape[3]), img])

    # get prediction
    preds = model(img)
    probs = torch.softmax(preds, axis=0)
    ix = torch.argmax(probs[1], axis=0).item()

    labels_map = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}

    return {
        'label': labels_map[ix],
        'score': probs[1, ix].item()
    }
    # return 'densenet161!\n'

if __name__ == '__main__':
    # app.run()
    app.run(debug=True)