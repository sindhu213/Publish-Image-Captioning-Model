import torch
from PIL import Image
from utils.image_utils import image_transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def beam_search(model, image, vocab, beam_size=5, max_len=20):
    model.eval()
    with torch.no_grad():
        image = image.to(device)

        features = model.encoder(image)
        features = features.permute(0, 2, 3, 1)
        features = features.view(features.size(0), -1, features.size(-1))
        h, c = model.decoder.init_hidden_state(features)

        sequences = [[[], 0.0, h, c]]

        for _ in range(max_len):
            all_candidates = []

            for seq, score, h, c in sequences:
                if len(seq) > 0 and seq[-1] == vocab.word2idx['<eos>']:
                    all_candidates.append((seq, score, h, c))
                    continue

                prev_token = torch.tensor(
                    [seq[-1]] if seq else [vocab.word2idx['<sos>']]
                ).to(device)

                embedding = model.decoder.embedding(prev_token)
                context, _ = model.decoder.attention(features, h)
                lstm_input = torch.cat([embedding, context], dim=1)

                h, c = model.decoder.decode_step(lstm_input, (h, c))
                output = model.decoder.fc(model.decoder.dropout_layer(h))
                log_probs = torch.log_softmax(output, dim=1)

                top_log_probs, top_indices = torch.topk(log_probs, beam_size)

                for i in range(beam_size):
                    token = top_indices[0, i].item()
                    new_seq = seq + [token]
                    new_score = score + top_log_probs[0, i].item()
                    all_candidates.append((new_seq, new_score, h, c))

            sequences = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)[:beam_size]

        best_seq = sequences[0][0]
        caption = [vocab.idx2word[idx] for idx in best_seq if idx not in {
            vocab.word2idx['<sos>'], vocab.word2idx['<eos>'], vocab.word2idx['<pad>']
        }]
        return ' '.join(caption)
    
def generate_caption(model, image, vocab, beam_size=5, max_len=20):
    model.eval()
    image_tensor = image.unsqueeze(0).to(device)

    caption = beam_search(
        model=model,
        image=image_tensor,
        vocab=vocab,
        beam_size=beam_size,
        max_len=max_len
    )
    return caption

