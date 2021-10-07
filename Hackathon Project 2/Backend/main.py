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

        main(img)
        return "Success"
        
api.add_resource(SaveImage, '/image')

if __name__ == '__main__':
    app.run(debug=True)