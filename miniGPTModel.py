from data_dict import *
from myTrans.gpt_layer import *

class GPTDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder_layers = nn.ModuleList([
            GPTLayer() for _ in range(GPT_LAYER_NUM)
        ])
        self.embedding = nn.Embedding(VOCAB_SIZE, D_MODEL, padding_idx=PAD_ID)
        self.pos_embedding = nn.Embedding(BLOCK_SIZE, D_MODEL)

    def forward(self, x, mask=None):
        _, T = x.shape
        #1
        token_embedding = self.embedding(x)
        pos_embedding = self.pos_embedding(torch.arange(T, device=token_embedding.device))
        x = token_embedding + pos_embedding
        #2
        weights = []
        for layer in self.decoder_layers:
            x, w = layer(x, mask=mask)
            weights.append(w)
        return x, weights


class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.dec = GPTDecoder()
        self.fc = nn.Linear(D_MODEL, VOCAB_SIZE)
        self.norm = nn.LayerNorm(D_MODEL)

    def forward(self, x, mask_x):
        o, w= self.dec(x, mask=mask_x)
        o = self.norm(o)
        o = self.fc(o)
        return o, w