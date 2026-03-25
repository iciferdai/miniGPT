from data_dict import *
#from data_dict_addition import *
from torch.utils.data import DataLoader, TensorDataset, random_split, ConcatDataset
import re
import random

def generate_src_mask(src_token_ids, pad_id = PAD_ID):
    src_mask = (src_token_ids == pad_id)
    # [batch_size, 1, 1, src_seq_len] -> [batch_size, n_heads, seq_len_q, seq_len_k]
    src_mask_4d = src_mask.unsqueeze(1).unsqueeze(1)
    logging.debug(f"Mask: {src_mask_4d.shape} -> {src_mask_4d.device}")
    return src_mask_4d

def generate_tgt_mask(tgt_token_ids, pad_id = PAD_ID):
    # [batch_size, tgt_seq_len]
    batch_size, tgt_seq_len = tgt_token_ids.shape

    # ahead mask -> [batch_size, tgt_seq_len, tgt_seq_len]
    ahead_mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len, dtype=torch.bool), diagonal=1)
    ahead_mask = ahead_mask.unsqueeze(0).repeat(batch_size, 1, 1)

    # pad mask -> [batch_size, tgt_seq_len, tgt_seq_len]
    tgt_pad_mask = (tgt_token_ids == pad_id)
    tgt_pad_mask_3d = tgt_pad_mask.unsqueeze(1).repeat(1, tgt_seq_len, 1)

    # combine
    tgt_mask = ahead_mask | tgt_pad_mask_3d
    # [batch_size, 1, tgt_seq_len, tgt_seq_len] -> [batch_size, n_heads, seq_len_q, seq_len_k]
    tgt_mask_4d = tgt_mask.unsqueeze(1)
    logging.debug(f"Mask: {tgt_mask_4d.shape} -> {tgt_mask_4d.device}")
    return tgt_mask_4d


def generate_gpu_mask(block_size, n_heads):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ahead_mask = torch.triu(
        torch.ones(block_size, block_size, dtype=torch.bool, device=device),
        diagonal=1  # 对角线1以上的部分为True（遮挡未来token）
    )
    # 2. 扩展维度适配多头注意力：[1, n_heads, block_size, block_size]
    # 注意：这里只生成[1, n_heads, ...]，训练时会自动广播到batch_size维度
    mask = ahead_mask.unsqueeze(0).unsqueeze(0)  # [1,1,block_size,block_size]
    mask = mask.repeat(1, n_heads, 1, 1)  # [1,n_heads,block_size,block_size]
    return mask

"""
class GPT_Train_Data():
    def __init__(self, d):
        self.data = d

    def get_batches(self, batch_size, block_size):
        max_start_idx = len(self.data) - block_size - 1
        start_indices = []
        special_ids = [SEP_ID, EOS_ID, token2idx['…'], token2idx['！'], token2idx['，'], token2idx['：'], token2idx['—'], token2idx['“'], token2idx['、'], token2idx['。']]
        for _ in range(batch_size):
            idx = random.randint(0, max_start_idx)
            search_idx = 1
            while True:
                tmp_id = idx - search_idx
                if tmp_id <= 0:
                    start_indices.append(0)
                    break

                if self.data[tmp_id] in special_ids:
                    start_indices.append(tmp_id+1)
                    #print(f'random find id {idx}|{tmp_id}: {self.data[tmp_id]}')
                    break

                search_idx += 1
                if search_idx > IGNORE_INDEX:
                    start_indices.append(tmp_id)
                    #print(f'random not find id {idx}|{tmp_id}: {self.data[tmp_id]}')
                    break

        batch_x = []
        batch_y = []
        for start_idx in start_indices:
            chunk = self.data[start_idx: start_idx + block_size + 1]
            x = chunk[:-1]
            y = chunk[1:]
            batch_x.append(x)
            batch_y.append(y)

        tensor_x = torch.tensor(batch_x, dtype=torch.long)
        tensor_y = torch.tensor(batch_y, dtype=torch.long)
        mask_x = generate_tgt_mask(tensor_x)

        return tensor_x, tensor_y, mask_x
"""

class GPT_Train_Data_GPU():
    def __init__(self, d, special_ids):
        self.data = torch.tensor(d, dtype=torch.long).cuda()  # 数据直接放GPU
        self.special_ids = special_ids
        # 都是block，而且没有PAD，所以都一样，算一次就行
        self.mask = generate_gpu_mask(BLOCK_SIZE, NUM_HEADS)
        # 预计算所有合法的起始索引
        self.valid_starts = self._precompute_valid_starts()

    def _precompute_valid_starts(self):
        valid_starts = []
        data_np = self.data.cpu().numpy()  # 只在预处理时拷贝一次
        for i in range(len(data_np)):
            if data_np[i] in self.special_ids and i + 1 < len(data_np):
                valid_starts.append(i + 1)
        return torch.tensor(valid_starts, dtype=torch.long).cuda()  # 预计算结果也放GPU

    def get_batches(self, batch_size, block_size):
        # 直接从预计算的合法位置中随机采样
        start_indices = self.valid_starts[torch.randint(len(self.valid_starts), (batch_size,))]

        # 在GPU上直接截取数据
        max_len = block_size + 1
        batch_x = []
        batch_y = []
        for start_idx in start_indices:
            end_idx = start_idx + max_len
            if end_idx > len(self.data):
                # 处理边界情况
                pad_len = end_idx - len(self.data)
                chunk = torch.cat([self.data[start_idx:], torch.zeros(pad_len, dtype=torch.long, device='cuda')])
            else:
                chunk = self.data[start_idx:end_idx]
            x = chunk[:-1]
            y = chunk[1:]
            batch_x.append(x)
            batch_y.append(y)

        tensor_x = torch.stack(batch_x)
        tensor_y = torch.stack(batch_y)

        return tensor_x, tensor_y, self.mask

def process_data():
    str_list = []
    id_list = []
    for chapter in demo_data:
        for s in chapter:
            one_str_list = list(s)
            for i in one_str_list:
                str_list.append(i)
                if i in token2idx.keys():
                    id_list.append(token2idx[i])
                else:
                    id_list.append(UNK_ID)
            str_list.append('\n')
            id_list.append(SEP_ID)
        str_list.append('\n')
        id_list.append(EOS_ID)
    #print(f'length: {len(str_list)}|{len(id_list)}')
    #d = GPT_Train_Data(id_list)
    d = GPT_Train_Data_GPU(id_list, [SEP_ID, EOS_ID, token2idx['…'], token2idx['！'], token2idx['，'], token2idx['：'],
                                     token2idx['—'], token2idx['“'], token2idx['、'], token2idx['。']])
    return d

if __name__ == '__main__':
    d = process_data()
    print(d.get_batches(8, 128))