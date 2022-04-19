from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/resnet50')
def resnet50():
    return 'Resnet50!'

if __name__ == '__main__':
    app.run()
    # app.run(debug=True)