from flask import Flask, request, jsonify, render_template
import torch
from model.Encoder import Resnet101
from model.Decoder import DecoderWithAttention
from model.CombinedModel import ImageCaptioningModel
from PIL import Image, UnidentifiedImageError
from utils.utils import load_model
from inference.caption_generator import generate_caption
from utils.vocab import Vocabulary
from utils.image_utils import image_transforms
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app, resources={
    r"/caption": {
        "origins": ["http://localhost:5173", "http://192.168.29.178:5173"],
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
        "supports_credentials": True
    }
})

vocab = Vocabulary()
vocab.__dict__.update(torch.load("checkpoints/vocab_dict.pth"))
encoder = Resnet101()
decoder = DecoderWithAttention(attention_dim=256, embed_dim=300, hidden_dim=512, vocab=vocab)
model = ImageCaptioningModel(encoder, decoder)

model_state_dict, device = load_model()
model.load_state_dict(model_state_dict)
model = model.to(device)
model.eval()

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = 'http://localhost:5173'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response


@app.route('/caption', methods=['OPTIONS'])
def handle_options():
    response = jsonify({'message': 'Preflight Accepted'})
    response.headers.add("Access-Control-Allow-Origin", "http://localhost:5173")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Allow-Methods", "POST")
    return response

@app.route('/caption', methods=['POST'])
@cross_origin() 
def caption():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    try:
        image = Image.open(request.files['image']).convert("RGB")
    except UnidentifiedImageError:
        return jsonify({'error': 'Uploaded file is not a valid image'}), 400

    try:
        transform = image_transforms()
        image = transform(image)
        caption = generate_caption(model, image, vocab)
        return jsonify({'caption': caption})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
