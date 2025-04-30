import torch.nn as nn

class ImageCaptioningModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images, captions, lengths):
        # images = [batch_size, 3, 224, 224]
        # captions = [batch_size, max_seq_len]
        # lengths(List) = batch_size

        # encoder_out = [batch_size, 2048, 7, 7]
        encoder_out = self.encoder(images)
        # predictions = [batch_size, max_decode_len, vocab_size]
        # captions_sorted = [batch_size,max_seq_len]
        # decode_lengths = List(int)
        # alphas = [batch_size, max_decode_len, 49]
        # sort_ind = [batch_size]
        predictions, captions_sorted, decode_lengths, alphas, sort_ind = self.decoder(
            encoder_out, captions, lengths
        )
        return predictions, captions_sorted, decode_lengths, alphas, sort_ind