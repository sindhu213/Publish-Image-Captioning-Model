import os
import gdown
import torch
from utils.vocab import Vocabulary

def load_vocab():
    if not os.path.exists("vocab.pth"):
        print("Downloading vocab...")
        file_id = "1iedHly1ZGTAK1JnW-vH0NQXUug6TKF2A"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, "vocab.pth",quiet=False)
    
    vocab = torch.load("vocab.pth", weights_only=False)
    return vocab

def load_model():
    if not os.path.exists("saved_model.pth"):
        print("Downloading model...")
        file_id = "1BYpkwtPNnLGClJnkQ0nzguz3FIaIeVU2"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, "saved_model.pth", quiet=False)

    model = torch.load("saved_model.pth", map_location=torch.device("cpu"))
    return model