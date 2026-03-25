from myTrans.ffn import *
from myTrans.multi_att import *

class GPTLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask_att = MultiHeadAttention()
        self.ffn = FFN()
        self.norm1 = nn.LayerNorm(D_MODEL)
        self.norm2 = nn.LayerNorm(D_MODEL)
        self.dropout = nn.Dropout(DROPOUT_RATE)

    def forward(self, x, mask=None):
        # 1
        x_norm1 = self.norm1(x)
        # 2
        o1, w = self.mask_att(x_norm1, x_norm1, x_norm1, mask=mask)
        # 3
        x = x + self.dropout(o1)
        # 4
        x_norm2 = self.norm2(x)
        # 5
        o2 = self.ffn(x_norm2)
        # 6
        o = x + self.dropout(o2)

        return o, w
