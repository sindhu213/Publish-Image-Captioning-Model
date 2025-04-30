import torch
import torch.nn as nn

class Attention(nn.Module):
  def __init__(self,feature_dim, hidden_dim, attn_dim):
    super(Attention, self).__init__()
    self.encoder_attn = nn.Linear(feature_dim, attn_dim)
    self.hidden_attn = nn.Linear(hidden_dim, attn_dim)
    self.full_attn = nn.Linear(attn_dim, 1)
    self.relu = nn.ReLU()

  def forward(self, encoder_out, hidden):
    # encoder_out = [batch_size, 49, 2048]
    # hidden = [batch_size, hidden_dim]

    # attn1 = [batch_size, 49, attn_dim]
    attn1 = self.encoder_attn(encoder_out)
    # attn2 = [batch_size, 1, attn_dim]
    attn2 = self.hidden_attn(hidden).unsqueeze(1)
    # attn = [batch_size, 49, attn_dim]
    attn = self.relu(attn1 + attn2)
    # out = [batch_size, 49]
    out = self.full_attn(attn).squeeze(2)
    # alpha = [batch_size, 49]
    alpha = torch.softmax(out, dim=1)
    # context = [batch_size, 2048]
    context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
    return context, alpha
  
class DecoderWithAttention(nn.Module):
    def __init__(self, attention_dim, embed_dim, hidden_dim, vocab, encoder_dim=2048, dropout=0.5):
        super(DecoderWithAttention, self).__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.encoder_dim = encoder_dim

        self.attention = Attention(encoder_dim, hidden_dim, attention_dim)
        self.embedding = nn.Embedding(self.vocab_size, embed_dim)
        self.embedding.weight = nn.Parameter(self.vocab.embeddings, requires_grad=False)

        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, hidden_dim, bias=True)
        self.init_h = nn.Linear(encoder_dim, hidden_dim)
        self.init_c = nn.Linear(encoder_dim, hidden_dim)
        self.f_beta = nn.Linear(hidden_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(hidden_dim, self.vocab_size)
        self.dropout_layer = nn.Dropout(p=dropout)

        self.init_weights()


    def init_weights(self):
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)


    def init_hidden_state(self, encoder_out):
        # encoder_out: [batch_size, 49, 2048]

        # mean_encoder_out = [batch_size, 2048]
        mean_encoder_out = encoder_out.mean(dim=1)
        # h = [batch_size, hidden_dim]
        h = self.init_h(mean_encoder_out)
        # c = [batch_size, hidden_dim]
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        # encoder_out = [batch_size, 2048, 7, 7]
        # encoder_captions = [batch_size, max_seq_len]
        # caption_lengths(List) = batch_size

        batch_size = encoder_out.size(0)
        # encoder_out = [batch_size, 2048, 49]
        encoder_out = encoder_out.view(batch_size, self.encoder_dim, -1)
        # encoder_out = [batch_size, 49, 2048]
        encoder_out = encoder_out.permute(0, 2, 1)
        # num_pixels = 49
        num_pixels = encoder_out.size(1)

        # caption_lengths = [batch_size]
        # sort_ind = [batch_size]
        caption_lengths, sort_ind = torch.sort(torch.tensor(caption_lengths), dim=0, descending=True)
        # encoder_out = [batch_size, 49, 2048]
        encoder_out = encoder_out[sort_ind]
        # encoded_captions = [batch_size,max_seq_len]
        encoded_captions = encoded_captions[sort_ind]
        # embeddings = [batch_size, max_seq_len, embed_dim]
        embeddings = self.embedding(encoded_captions)
        # h = [batch_size, hidden_dim]
        # c = [batch_size, hidden_dim]
        h, c = self.init_hidden_state(encoder_out)

        decode_lengths = (caption_lengths - 1).tolist()
        max_decode_len = max(decode_lengths)
        # predictions = [batch_size, max_decode_len, vocab_size]
        # alphas = [batch_size, max_decode_len, vocab_size]
        predictions = torch.zeros(batch_size, max_decode_len, self.vocab_size).to(encoder_out.device)
        alphas = torch.zeros(batch_size, max_decode_len, num_pixels).to(encoder_out.device)

        for t in range(max_decode_len):
            batch_size_t = sum([l > t for l in decode_lengths])
            # attention_weighted_encoding = [batch_size_t, 2048]
            # alpha = [batch_size_t, 49]
            attention_weighted_encoding, alpha = self.attention(
                encoder_out[:batch_size_t], h[:batch_size_t]
            )
            # gate = [batch_size_t, 2048]
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            # attention_weighted_encoding = [batch_size_t, 2048]
            attention_weighted_encoding = gate * attention_weighted_encoding
            # lstm_input = [batch_size_t, embed_dim + 2048]
            lstm_input = torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1)
            # h = [batch_size_t, hidden_dim]
            # c = [batch_size_t, hidden_dim]
            h, c = self.decode_step(lstm_input, (h[:batch_size_t], c[:batch_size_t]))
            # preds = [batch_size_t, vocab_size]
            preds = self.fc(self.dropout_layer(h))
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind