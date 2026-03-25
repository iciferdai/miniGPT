from myTrans.base_params import *

# 第一步：定义带标签平滑的交叉熵损失函数
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, vocab_size, ignore_index=PAD_ID, smoothing=LABEL_SMOOTH):
        super().__init__()
        self.vocab_size = vocab_size  # 词表大小
        self.ignore_index = ignore_index  # 忽略的索引
        self.smoothing = smoothing  # 平滑系数，业界主流0.1（目标token 90%概率）

    def forward(self, pred, target):
        # 1. 计算log_softmax（数值更稳定）
        log_prob = F.log_softmax(pred, dim=-1)

        # 2. 构建软标签分布：目标token占(1-smoothing)，其余均匀分配smoothing/(vocab_size-1)
        # 初始化全部分布为 smoothing/(vocab_size-1)
        soft_target = torch.full_like(log_prob, self.smoothing / (self.vocab_size - 1))
        # 把目标token的位置替换为 (1 - self.smoothing)
        soft_target.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)

        # 3. 计算损失（带ignore_index处理）
        if self.ignore_index is not None:
            # 屏蔽ignore_index的位置
            mask = (target != self.ignore_index).unsqueeze(1)
            soft_target = soft_target * mask
            log_prob = log_prob * mask

        # 4. 交叉熵损失：-Σ(soft_target * log_prob) / 有效token数
        loss = -torch.sum(soft_target * log_prob)
        # 计算有效token数（避免除以0）
        num_tokens = mask.sum() if self.ignore_index is not None else target.numel()
        loss = loss / num_tokens.clamp(min=1)

        return loss