import signal
from miniGPTModel import *
from processData import *
import torch.optim as optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, ConstantLR
from torch.amp import autocast, GradScaler
import pickle
import time
import matplotlib.pyplot as plt
from SoftLoss import LabelSmoothingCrossEntropy

class ModelManagement():
    def __init__(self, model, train_dataloader, device=torch.device('cpu')):
        # === static ===
        self.STEP_PROGRESS_COUNT = 100
        self.STEP_CHECKPOINT_COUNT = 5000
        self.STEP_IGNORE_CHECKPOINT = 50
        self.WARMUP_STEPS = 2000
        self.COS_STEPS = 18000
        # === init ===
        self.model = model
        self.train_dl = train_dataloader
        self.device = device
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.scaler = GradScaler('cuda') if torch.cuda.is_available() else None
        # === dynamic related with model===
        self.train_loss = float('inf')
        self.best_train_loss = float('inf')
        # === dynamic related with mgmt===
        self.step_count = 0
        self.train_loss_list = []
        self.best_checkpoints = dict()
        # === tmp & plt & flags ===
        self.step = 0
        self.steps = 0
        self.monitor_flag = []
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.train_line1 = None
        self.train_line2 = None
        # for manual exit
        self._register_signal_handler()

    def _register_signal_handler(self):
        #  SIGINT(Ctrl+C)   SIGTERM
        self.original_sigint = signal.getsignal(signal.SIGINT)
        self.original_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, self._handle_termination)
        signal.signal(signal.SIGTERM, self._handle_termination)

    def _handle_termination(self, signum, frame):
        print(f"\n!!! CATCH SIGNAL: {signum}, SAVING BEFORE TERMINATING!!!\n...")
        try:
            self.save_checkpoint()
            print("SAVED SUCCESS, EXIT NOW")
        except Exception as e:
            print(f"SAVED FAILED: {e}, EXIT")
        finally:
            signal.signal(signal.SIGINT, self.original_sigint)
            signal.signal(signal.SIGTERM, self.original_sigterm)
            exit(0)

    def init_weights(self):
        for name, param in self.model.named_parameters():
            if 'weight' in name and 'emb' in name:
                # 嵌入层初始化
                param.data.normal_(0.0, 1.0 / math.sqrt(D_MODEL))
            elif 'weight' in name and 'linear' in name:
                # 注意力层Linear
                if 'attn' in name:
                    nn.init.xavier_uniform_(param)
                # FFN层Linear
                elif 'ffn' in name:
                    nn.init.kaiming_uniform_(param, nonlinearity='relu')
            elif 'bias' in name:
                # 偏置初始化为0
                nn.init.zeros_(param)
            elif 'norm' in name:
                # 层归一化初始化为1（weight）和0（bias）
                if 'weight' in name:
                    nn.init.ones_(param)
                else:
                    nn.init.zeros_(param)

    def init_train(self):
        self.model.train()
        self.model.to(self.device)
        self.init_dashboard()
        # 分组设置weight_decay
        param_groups = [
            # 权重参数：应用weight_decay
            {'params': [p for n, p in self.model.named_parameters() if 'weight' in n and 'norm' not in n],
             'weight_decay': 0.03},
            # 偏置/归一化参数：无weight_decay
            {'params': [p for n, p in self.model.named_parameters() if 'bias' in n or 'norm' in n],
             'weight_decay': 0.0}
        ]
        self.optimizer = optim.AdamW(param_groups, lr=5e-5, betas=(0.9, 0.98), eps=1e-8)
        self.criterion = LabelSmoothingCrossEntropy(VOCAB_SIZE)

    def init_eval(self):
        self.model.eval()
        self.model.to(self.device)

    def progress_info(self, force=False):
        if self.monitor_flag:
            logging.info(
                f"[{self.step + 1}/{self.steps}]|Step_{self.step_count}] -> Loss: {self.train_loss:.4f}, "
                f"Best loss: {self.best_train_loss:.4f}, lr: {self.optimizer.param_groups[0]['lr']:.6f}; "
                f"\n ---> Monitor: {','.join(self.monitor_flag)}")
            self.monitor_flag = []
            self.update_dashboard()
        elif (self.step + 1) % self.STEP_PROGRESS_COUNT == 0:
            logging.info(
                f"[{self.step + 1}/{self.steps}]|Step_{self.step_count}] -> Loss: {self.train_loss:.4f}, "
                f"Best loss: {self.best_train_loss:.4f}, lr: {self.optimizer.param_groups[0]['lr']:.6f}")
            self.update_dashboard()
        elif force:
            logging.info(
                f"[{self.step + 1}/{self.steps}]|Step_{self.step_count}] -> Loss: {self.train_loss:.4f}, "
                f"Best loss: {self.best_train_loss:.4f}")
            logging.info(f'Best checkpoints: {self.best_checkpoints}')
            self.update_dashboard()

        if not force and (self.step_count % self.STEP_CHECKPOINT_COUNT == 0):
            self.save_checkpoint()

    def save_checkpoint(self, ckp_name=''):
        checkpoint = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'train_loss': self.train_loss,
            'best_train_loss': self.best_train_loss
        }
        if ckp_name:
            weight_path = './saves/' + ckp_name
        else:
            weight_path = f'./saves/CheckPoint_Ep{self.step_count}_{self.train_loss:.4f}.pth'
        torch.save(checkpoint, weight_path)
        logging.info(f"checkpoint: {weight_path} Saved")

    def load_checkpoint(self, ckp_name='', only_weights=False):
        if not ckp_name:
            print('No checkpoint provided.')
            return
        weight_path = './saves/' + ckp_name
        try:
            ckpt = torch.load(weight_path, map_location=self.device)
            self.model.load_state_dict(ckpt["state_dict"])
            if not only_weights:
                self.optimizer.load_state_dict(ckpt["optimizer"])
                #self.scheduler.load_state_dict(ckpt["scheduler"])  # 调度器每次根据传入训练步数重新设置，所以不load
                self.train_loss = ckpt["train_loss"]
                self.best_train_loss = ckpt["best_train_loss"]
            logging.info(f"checkpoint: {weight_path} Loaded")
        except Exception as e:
            logging.error(f"load_checkpoint Error: {e}", exc_info=True)

    def save_state(self, state_name=''):
        manager_state = {
            'step_count': self.step_count,
            'train_loss_list': self.train_loss_list,
            'best_checkpoints': self.best_checkpoints
        }
        if state_name:
            state_path = './saves/' + state_name
        else:
            state_path = f'./saves/State_Ep{self.step_count}_{self.best_train_loss:.4f}.pkl'
        with open(state_path, "wb") as f:
            pickle.dump(manager_state, f)
        logging.info(f"State saved at {state_path}")

    def load_state(self, state_name=''):
        if not state_name:
            print('No state provided.')
            return
        state_path = './saves/' + state_name
        try:
            with open(state_path, 'rb') as f:
                manager_state = pickle.load(f)
                self.step_count = manager_state['step_count']
                self.train_loss_list = manager_state['train_loss_list']
                self.best_checkpoints = manager_state['best_checkpoints']
                logging.info(f"State: {state_path} Loaded")
        except Exception as e:
            logging.error(f"load_state Error: {e}", exc_info=True)

    def clear_state(self):
        self.train_loss = float('inf')
        self.best_train_loss = float('inf')
        self.step_count = 0
        self.train_loss_list = []
        self.best_checkpoints = dict()

    def save_best(self):
        self.save_checkpoint('best_loss_cpt.pth')
        self.best_checkpoints[self.step_count] = self.train_loss
        self.save_state('best_state.pkl')

    def load_best(self):
        self.load_checkpoint('best_loss_cpt.pth', False)
        self.load_state('best_state.pkl')
        logging.info(f'Load Best; Best checkpoints: {self.best_checkpoints}')

    def roll_back(self, with_state=False):
        self.load_checkpoint('best_loss_cpt.pth', False)
        if with_state: self.load_state('best_state.pkl')

    def trans_data2dev(self, *args):
        transferred_args = []
        for arg in args:
            try:
                transferred_arg = arg.to(self.device, non_blocking=True)
                transferred_args.append(transferred_arg)
            except Exception as e:
                logging.error(f"trans_data2dev Error: {e}", exc_info=True)
                transferred_args.append(arg)

        if len(transferred_args) == 1:
            return transferred_args[0]
        return tuple(transferred_args)

    def init_dashboard(self):
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('PIL').setLevel(logging.WARNING)
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        # ion
        plt.switch_backend('TkAgg')
        plt.ion()
        # draw
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(20, 6), num="Loss Dashboard")
        self.ax1.set_xlabel('Steps')
        self.ax1.set_ylabel('Loss')
        self.ax1.set_title('Linear Scale')
        self.ax1.grid(alpha=0.2)
        self.ax2.set_yscale('log')
        self.ax2.set_xlabel('Steps')
        self.ax2.set_ylabel('Loss')
        self.ax2.set_title('Log Scale')
        self.ax2.grid(alpha=0.2)
        # init
        self.train_line1, = self.ax1.plot(range(1, self.step_count+1),
                                        self.train_loss_list,
                                        label='Train Loss',
                                        marker='o',
                                        markersize=2)
        self.train_line2, = self.ax2.plot(range(1, self.step_count + 1),
                                        self.train_loss_list,
                                        label='Train Loss',
                                        marker='o',
                                        markersize=2)
        # update
        self.ax1.legend()
        self.ax2.legend()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update_dashboard(self):
        self.train_line1.set_xdata(range(1, len(self.train_loss_list) + 1))
        self.train_line1.set_ydata(self.train_loss_list)
        self.train_line2.set_xdata(range(1, len(self.train_loss_list) + 1))
        self.train_line2.set_ydata(self.train_loss_list)
        # auto-set
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()
        # update
        #self.ax1.legend()
        #self.ax2.legend()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        # optional
        #time.sleep(0.01)

    def show_dashboard(self):
        plt.ioff()
        plt.show()

    def loss_algorithm(self):
        if self.train_loss < self.best_train_loss and self.step_count > self.STEP_IGNORE_CHECKPOINT:
            self.save_best()
            self.monitor_flag.append(f'Save Best Loss! ({self.best_train_loss:.4f}->{self.train_loss:.4f})')
            self.best_train_loss = self.train_loss

    def get_batch_loss(self, one_pack_data, is_fp16=False):
        # unpack: Source -> processData
        batch_x, batch_y, mask_x = one_pack_data
        batch_x, batch_y, mask_x = self.trans_data2dev(batch_x, batch_y, mask_x)
        # forward
        if is_fp16:
            with autocast(device_type='cuda', dtype=torch.float16):
                output, _ = self.model(batch_x, mask_x)
                # 输出截取：[BS, BLOCK, VOCAB_SIZE] → [BS, BLOCK-IGNORE_INDEX, VOCAB_SIZE]
                out_ignore = output[:, IGNORE_INDEX:, :]
                # 目标值截取：[BS, BLOCK] → [BS, BLOCK-IGNORE_INDEX]
                tgt_ignore = batch_y[:, IGNORE_INDEX:]
                # 展平计算损失
                out_flat = out_ignore.reshape(-1, VOCAB_SIZE)
                tgt_flat = tgt_ignore.reshape(-1)
                loss = self.criterion(out_flat, tgt_flat)
            loss = loss.float()
        else:
            output, _ = self.model(batch_x, mask_x)
            out_ignore = output[:, IGNORE_INDEX:, :]
            tgt_ignore = batch_y[:, IGNORE_INDEX:]
            out_flat = out_ignore.reshape(-1, VOCAB_SIZE)
            tgt_flat = tgt_ignore.reshape(-1)
            loss = self.criterion(out_flat, tgt_flat)

        return loss

    def get_batch_output(self, one_pack_data):
        # unpack: Source -> processData
        batch_x, batch_y, mask_x = one_pack_data
        batch_x, batch_y, mask_x = self.trans_data2dev(batch_x, batch_y, mask_x)
        # forward
        output, _ = self.model(batch_x, mask_x)
        return output.cpu()

    def train_step(self, is_fp16=False):
        one_pack_data = self.train_dl.get_batches(DEFAULT_BATCH_SIZE, BLOCK_SIZE)
        loss = self.get_batch_loss(one_pack_data, is_fp16)
        if is_fp16:
            self.scaler.scale(loss).backward()  # 缩放loss并反向传播
            self.scaler.unscale_(self.optimizer)  # 反缩放梯度（用于梯度裁剪）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # 梯度裁剪
            self.scaler.step(self.optimizer)  # 优化器更新（自动处理梯度缩放）
            self.scaler.update()  # 更新scaler的缩放因子
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # 梯度裁剪
            self.optimizer.step()

        return loss.item()

    def train_steps(self, input_steps, is_fp16=False):
        if self.optimizer is None:
            logging.error('NEED INIT TRAIN FIRST!!!')
            return
        self.steps = input_steps

        if input_steps > self.WARMUP_STEPS + self.COS_STEPS:
            warm_steps = self.WARMUP_STEPS
            cos_steps = self.COS_STEPS
            last_step = input_steps - self.WARMUP_STEPS - self.COS_STEPS
        else:
            warm_steps = 1
            cos_steps = 1
            last_step = input_steps-2

        print(f'Strat Training with policy: Warm-{warm_steps}，Cos-{cos_steps}, Last-{last_step}')
        warmup = LinearLR(self.optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warm_steps)
        cos = CosineAnnealingLR(self.optimizer,T_max=cos_steps,eta_min=1e-5)
        constant = ConstantLR(self.optimizer,factor=0.02,total_iters=last_step)
        self.scheduler = SequentialLR(self.optimizer,schedulers=[warmup, cos, constant],milestones=[warm_steps, warm_steps+cos_steps])

        for one_step in range(self.steps):
            self.optimizer.zero_grad()
            self.step = one_step
            self.train_loss = self.train_step(is_fp16)
            if self.scheduler is not None: self.scheduler.step()
            self.train_loss_list.append(self.train_loss)
            self.loss_algorithm()
            self.step_count += 1
            self.progress_info()

        self.save_checkpoint()

    def predict_step(self, id_dict):
        tgt_tensor = torch.tensor([id_dict], dtype=torch.long)
        # infer
        t_mask = generate_tgt_mask(tgt_tensor)
        tgt_tensor, t_mask = self.trans_data2dev(tgt_tensor, t_mask)
        output, _ = self.model(tgt_tensor, t_mask)
        last_token_logits = output[:, -1, :]
        return last_token_logits

    def predict_best(self, id_dict):
        pred_text = []
        prob_list = []
        prob_log_list = []
        for i in range(BLOCK_SIZE - len(id_dict) - 1):
            last_token_logits = self.predict_step(id_dict)
            # BEST
            next_token_idx = torch.argmax(last_token_logits, dim=-1, keepdim=True)
            probs = F.softmax(last_token_logits, dim=-1)
            max_prob = torch.gather(probs, dim=-1, index=next_token_idx)
            token_log_prob = torch.log(max_prob + 1e-10).item()
            # next id
            next_id = next_token_idx[0].item()
            id_dict.append(next_id)
            pred_text.append(idx2token[next_id])
            prob_list.append(max_prob[0].item())
            prob_log_list.append(token_log_prob)
            # EOS
            if next_id == EOS_ID:
                break

        output_text = ''.join(pred_text)
        return output_text, prob_list, prob_log_list

    def predict_top_k(self, id_dict, temperature, k):
        pred_text = []
        prob_list = []
        prob_log_list = []
        for i in range(BLOCK_SIZE - len(id_dict) - 1):
            last_token_logits = self.predict_step(id_dict)
            # temperature
            logits = last_token_logits / temperature
            probs = F.softmax(logits, dim=-1)
            # top-k
            topk_probs, topk_ids = torch.topk(probs, k=k, dim=-1)
            topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
            sample_idx = torch.multinomial(topk_probs, num_samples=1)
            # next id
            sample_idx_scalar = sample_idx[0].item()
            next_id_tensor = topk_ids[0][sample_idx_scalar].unsqueeze(0).unsqueeze(0)
            next_id = next_id_tensor.item()
            sample_prob = torch.gather(probs, dim=-1, index=next_id_tensor)
            token_log_prob = torch.log(sample_prob + 1e-10).item()
            id_dict.append(next_id)
            pred_text.append(idx2token[next_id])
            prob_list.append(sample_prob[0].item())
            prob_log_list.append(token_log_prob)

            if next_id == EOS_ID:
                break

        output_text = ''.join(pred_text)
        return output_text, prob_list, prob_log_list

    def predict_top_p(self, id_dict, temperature, p):
        pred_text = []
        prob_list = []
        prob_log_list = []
        for i in range(BLOCK_SIZE - len(id_dict) - 1):
            last_token_logits = self.predict_step(id_dict)
            # temperature
            logits = last_token_logits / temperature
            probs = F.softmax(logits, dim=-1)
            # 1.sort
            sorted_probs, sorted_ids = torch.sort(probs, dim=-1, descending=True)
            # 2.
            cum_probs = torch.cumsum(sorted_probs, dim=-1)
            # 3.
            mask = cum_probs <= p
            mask[0, 0] = True
            filtered_probs = sorted_probs[mask].unsqueeze(0)
            filtered_ids = sorted_ids[mask].unsqueeze(0)
            # 4.
            filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
            # 5.
            sample_idx = torch.multinomial(filtered_probs, num_samples=1).item()

            # next id
            next_id_tensor = filtered_ids[0, sample_idx].unsqueeze(0).unsqueeze(0)
            next_id = filtered_ids[0, sample_idx].item()
            sample_prob = torch.gather(probs, dim=-1, index=next_id_tensor)
            token_log_prob = torch.log(sample_prob + 1e-10).item()
            id_dict.append(next_id)
            pred_text.append(idx2token[next_id])
            prob_list.append(sample_prob[0].item())
            prob_log_list.append(token_log_prob)

            if next_id == EOS_ID:
                break

        output_text = ''.join(pred_text)
        return output_text, prob_list, prob_log_list

    def predict_manual(self, txt, mode='BEST', temperature = 1.0, k_p = 1.0, is_prob=True):
        # pre-process
        id_dict = []
        cn_list = list(txt)
        for i in cn_list:
            id_dict.append(token2idx[i])

        with torch.no_grad():
            if mode == 'BEST':
                print(f'Use mode BEST Predict:')
                output_text, prob_list, prob_log_list = self.predict_best(id_dict)
            elif mode == 'TOP_K':
                print(f'Use mode TOP-k({temperature}|{k_p}) Predict:')
                output_text, prob_list, prob_log_list = self.predict_top_k(id_dict, temperature, k_p)
            elif mode == 'TOP_P':
                print(f'Use mode TOP-p({temperature}|{k_p}) Predict:')
                output_text, prob_list, prob_log_list = self.predict_top_p(id_dict, temperature, k_p)
            else:
                logging.warning(f'Unexpected mode: {mode}, use BEST as default!')
                output_text, prob_list, prob_log_list = self.predict_best(id_dict)

            print(f'{txt + output_text}')
            if is_prob:
                print(','.join([f"{num:.2f}" for num in prob_list]))
                print(f"Sum prob: {(sum(prob_log_list) / len(prob_log_list)):.2f}")


if __name__ == '__main__':
    print('init model...')
    model = MiniGPT()
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('init ModelManagement...')
    m_mgmt = ModelManagement(model, None, dev)
    print('Empty, Do nothing, Exit...')