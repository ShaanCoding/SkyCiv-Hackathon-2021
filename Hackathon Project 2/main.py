from flask import Flask
from flask_restful import Resource, Api, reqparse
import werkzeug
import cv2
import numpy as np

app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('file',
                    type=werkzeug.datastructures.FileStorage,
                    location='files',
                    required=True,
                    help='provide a file')

class SaveImage(Resource):
    def post(self):
        args = parser.parse_args()
        # read like a stream
        stream = args['file'].read()
        # convert to numpy array
        npimg = np.fromstring(stream, np.uint8)
        # convert numpy array to image
        img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
        cv2.imwrite("D:/GitHub/SkyCiv-Hackathon-2021/Hackathon Project 2/test.png", img)

api.add_resource(SaveImage, '/image')

if __name__ == '__main__':
    app.run(debug=True)