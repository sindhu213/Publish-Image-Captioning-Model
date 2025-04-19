from flask import Flask, request, jsonify, render_template
import torch
from model.Encoder import VGG16Encoder
from model.Decoder import Decoder
from model.CombinedModel import ImageCaptioningModel
from PIL import Image, UnidentifiedImageError
from utils.utils import load_vocab, load_model
from inference.caption_generator import generate_caption
from utils.vocab import Vocabulary
from utils.image_utils import image_transforms

app = Flask(__name__)
vocab = Vocabulary()
vocab.__dict__.update(torch.load("checkpoints/vocab_dict.pth"))
encoder = VGG16Encoder()
decoder = Decoder(embed_dim=300, hidden_size=512, vocab=vocab)
model = ImageCaptioningModel(encoder, decoder)

model = load_model()
model.eval()

# Routes
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/caption', methods=['POST'])
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