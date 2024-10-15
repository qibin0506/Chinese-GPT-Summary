import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from tokenizer import Tokenizer
from ddp import DDPHelper
import math

# 梯度积累步数
gradient_accumulation_steps = 0

ddp_helper = DDPHelper()
tokenizer = Tokenizer()

"""
'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params

'vocab_size':   50257 # always 50257 for GPT model checkpoints
'block_size':   1024 # always 1024 for GPT model checkpoints
"""

CFG = {
    'vocab_size': tokenizer.vocab_size,
    'ctx_len': 256,
    'embed_dim': 1024,
    'n_heads': 16,
    'n_layers': 24,
    'drop_rate': 0.1
}


def padding_fn(batch_data):
    inputs = pad_sequence(batch_data, batch_first=True, padding_value=0)
    # crossEntropy默认的ignore_index是-100
    labels = pad_sequence(batch_data, batch_first=True, padding_value=-100)

    return inputs, labels


class CosineAnnealingWarmupScheduler:
    def __init__(self, warmup_iters, initial_lr, min_lr, max_lr, total_iters):
        super().__init__()

        self.warmup_iters = warmup_iters
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.total_iters = total_iters
        self.lr_increment = (max_lr - initial_lr) / warmup_iters
        self.steps = -1

        print(f'warmup_iters: {self.warmup_iters},'
              f' initial_lr: {self.initial_lr},'
              f' min_lr: {self.min_lr},'
              f' max_lr: {self.max_lr},'
              f'total_iters: {self.total_iters},'
              f'lr_increment: {self.lr_increment}')

    def update_steps(self, steps):
        self.steps = steps

    def incr_steps(self):
        self.steps += 1

    def can_clip_grad(self):
        return self.steps > self.warmup_iters

    def update_lr(self, optimizer: torch.optim.Optimizer):
        if self.steps <= self.warmup_iters:
            # Warmup: adjust learning rate linearly
            lr = self.initial_lr + self.steps * self.lr_increment
        else:
            # Cosine annealing phase
            progress = (self.steps - self.warmup_iters) / (self.total_iters - self.warmup_iters)
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def pretrain_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits = logits.reshape(-1, logits.shape[-1])
    targets = labels.reshape(-1)

    return F.cross_entropy(logits, targets, ignore_index=-100)


def calc_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # logits shape (batch, seq_len, vocab_size)
    # labels shape (batch, seq_len)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    logits = shift_logits.reshape(-1, logits.shape[-1])
    targets = shift_labels.reshape(-1)

    return F.cross_entropy(logits, targets, ignore_index=-100)


def generate_text(model, tokens, ctx_len, max_new_tokens, temperature, topk=None, token_item_callback=None):
    for _ in range(max_new_tokens):
        t = tokens[:, -ctx_len:]
        with torch.no_grad():
            with torch.autocast(device_type=ddp_helper.device_type, dtype=torch.bfloat16):
                # (batch, seq_len, vocab_size)
                logits = model(t)

        # (batch, vocab_size)
        logits = logits[:, -1, :]
        # 抑制[UNK]输出
        logits[..., tokenizer.unk] = torch.tensor(-torch.inf)

        if topk is not None:
            topk_logits, _ = torch.topk(logits, k=topk)
            min_val: torch.Tensor = topk_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(-torch.inf).to(ddp_helper.device), logits)

        if temperature > 0:
            logits /= temperature
            prob = logits.softmax(dim=-1)
            # 返回下标
            next_token = torch.multinomial(prob, num_samples=1)
        else:
            # 返回下标
            next_token = logits.argmax(dim=-1, keepdim=True)

        if token_item_callback is not None:
            token_item_callback(next_token)

        tokens = torch.cat([tokens, next_token], dim=-1)
        if next_token.item() == tokenizer.eot:
            break

    return tokens


def generate(model, prompt, max_new_tokens, temperature=1.25, topk=3, item_callback=None):
    model.eval()

    if item_callback is not None:
        token_item_callback = lambda token: item_callback(tokenizer.decode_to_text(token))
    else:
        token_item_callback = None

    encoded = tokenizer.encode_to_token(prompt).to(ddp_helper.device)
    output = generate_text(model, encoded, CFG['ctx_len'], max_new_tokens, temperature, topk, token_item_callback)
    decoded = tokenizer.decode_to_text(output)

    return decoded