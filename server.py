from flask import Flask, request, Response
from flask_restful import Resource, Api
from flask_cors import CORS
import json
import insightface
from insightface.app import FaceAnalysis
import cv2
import numpy as np
import base64
import io
from PIL import Image


app = Flask(__name__)
api = Api(app)
CORS(app, origins=["http://localhost:5000"])


def load_image_from_base64(base64_str):
    try:
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return image
    except Exception as e:
        raise ValueError(f"Error in decoding image: {str(e)}")


class TestModel(Resource):

    def __init__(self):
        try:
            self.model1 = FaceAnalysis(name='buffalo_l')
            self.model1.prepare(ctx_id=0, det_size=(480, 480))

            self.model2 = FaceAnalysis(name='buffalo_s')
            self.model2.prepare(ctx_id=0, det_size=(480, 480))
        except Exception as e:
            raise RuntimeError("Model initialization failed")

    def post(self):
        try:
            data = request.get_json()

            if 'img1' not in data or 'img2' not in data or 'model' not in data:
                return Response(json.dumps({'message': 'Invalid input data'}), status=400)

            img1 = load_image_from_base64(data['img1'])
            img2 = load_image_from_base64(data['img2'])
            sent_model = data['model']

            if sent_model == 'buffalo_l':
                model = self.model1
            elif sent_model == 'buffalo_s':
                model = self.model2
            else:
                return Response(json.dumps({'message': 'unknown model'}), status=400)

            faces1 = model.get(img1)
            faces2 = model.get(img2)

            if len(faces1) > 0 and len(faces2) > 0:
                embedding1 = faces1[0].embedding
                embedding2 = faces2[0].embedding
                sim = np.dot(embedding1, embedding2) / \
                    (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
                match_result = bool(sim > 0.2)
                return Response(json.dumps({'match': match_result}), status=200)
            else:
                return Response(json.dumps({'match': False}), status=200)
        except ValueError as ve:
            return Response(json.dumps({'message': str(ve)}), status=400)
        except Exception as e:
            return Response(json.dumps({'message': 'internal server error'}), status=500)


api.add_resource(TestModel, '/testModel')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
