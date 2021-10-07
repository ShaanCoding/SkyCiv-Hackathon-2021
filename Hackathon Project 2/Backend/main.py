from flask import Flask
from flask import jsonify
from flask.wrappers import Request
from flask_restful import Resource, Api, reqparse
import werkzeug
import cv2
import numpy as np
import warnings
from opencv import main


warnings.filterwarnings('ignore')
app = Flask(__name__)
api = Api(app)

class SaveImage(Resource):
    def get(self):
        response = main()
        return {'response': str(response).split("\n")[0]}
        
api.add_resource(SaveImage, '/image')

if __name__ == '__main__':
    app.run(debug=True)