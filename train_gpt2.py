from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken



class CausalSelfAttention(nn.Module):
  bias: torch.Tensor

  def __init__(self, config):
    super().__init__()
    assert config.n_embd % config.n_head == 0
    self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
    self.c_proj = nn.Linear(config.n_embd, config.n_embd)
    self.c_proj.NANOGPT_SCALE_INIT = True  # type: ignore # 让权重初始化时乘上一个系数，避免大模型梯度爆炸
    self.n_embd = config.n_embd
    self.n_head = config.n_head
    self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                         .view(1, 1, config.block_size, config.block_size))

  def forward(self, x):
    B, T, C = x.size()
    qkv = self.c_attn(x)
    q, k, v = qkv.split(self.n_embd, dim=2)
    k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
    att = F.softmax(att, dim=-1)
    y = att @ v
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    y = self.c_proj(y)
    return y

class MLP(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
    self.gelu = nn.GELU(approximate='tanh')
    self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
    self.c_proj.NANOGPT_SCALE_INIT = True  # type: ignore # 让权重初始化时乘上一个系数，避免大模型梯度爆炸

  def forward(self, x):
    x = self.c_fc(x)
    x = self.gelu(x)
    x = self.c_proj(x)
    return x

class Block(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.ln_1 = nn.LayerNorm(config.n_embd)
    self.attn = CausalSelfAttention(config)
    self.ln_2 = nn.LayerNorm(config.n_embd)
    self.mlp = MLP(config)

  def forward(self, x):
    x = x + self.attn(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))
    return x

@dataclass
class GPTConfig:
  block_size: int = 1024  # maximum context length
  vocab_size: int = 50257 # GPT-2 small has 50257
  n_layer: int = 12
  n_head: int = 12
  n_embd: int = 768 # GPT-2 small has 768

class GPT(nn.Module):
  def __init__(self,config):
    super().__init__()
    self.config = config

    self.transformer = nn.ModuleDict(dict(
      wte = nn.Embedding(config.vocab_size, config.n_embd), # token embedding
      wpe = nn.Embedding(config.block_size, config.n_embd), # position embedding
      h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # transformer blocks
      ln_f = nn.LayerNorm(config.n_embd),
    ))

    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    self.transformer.wte.weight = self.lm_head.weight  # type: ignore # weight tying
    self.apply(self._init_weights)  # type: ignore # initialize weights

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      std = 0.02
      if hasattr(module, 'NANOGPT_SCALE_INIT'):
        std *= (2 * self.config.n_layer) ** -0.5
      torch.nn.init.normal_(module.weight, mean=0.0, std=std)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def forward(self, idx, targets=None):
    B, T = idx.size()
    assert T <= self.config.block_size, "Cannot forward, model block size is exhausted."
    pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
    pos_emb = self.transformer.wpe(pos)
    tok_emb = self.transformer.wte(idx)
    x = tok_emb + pos_emb
    for block in self.transformer.h:
      x = block(x)
    x = self.transformer.ln_f(x)
    logits = self.lm_head(x)  # （B, T, n_embd）--> (B, T, vocab_size)
    loss = None
    if targets is not None:
      loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    return logits, loss
  
  @classmethod
  def from_pretrained(cls, model_type):
    """从 huggingface 加载预训练 GPT-2 权重"""
    assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
    from transformers import GPT2LMHeadModel
    print("loading weights from pretrained gpt: %s" % model_type)

    # 不同型号对应不同的层数/头数/维度
    config_args = {
      'gpt2':        dict(n_layer=12, n_head=12, n_embd=768),   # 124M
      'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # 350M
      'gpt2-large':  dict(n_layer=36, n_head=20, n_embd=1280),  # 774M
      'gpt2-xl':     dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M
    }[model_type]
    config_args['vocab_size'] = 50257   # GPT-2 都是 50257
    config_args['block_size'] = 1024    # GPT-2 都是 1024

    # 1) 造一个我们自己的空壳模型
    config = GPTConfig(**config_args)
    model = GPT(config)
    sd = model.state_dict()
    sd_keys = sd.keys()
    sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]  # 掩码buffer，不是权重，跳过

    # 2) 加载 huggingface 官方模型
    model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    sd_hf = model_hf.state_dict()

    # 3) 对拷，注意名字和形状要对齐
    sd_keys_hf = sd_hf.keys()
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
    transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
    assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
    for k in sd_keys_hf:
      if any(k.endswith(w) for w in transposed):
        # Conv1D 的权重要转置
        assert sd_hf[k].shape[::-1] == sd[k].shape
        with torch.no_grad():
          sd[k].copy_(sd_hf[k].t())
      else:
        # 其余直接拷
        assert sd_hf[k].shape == sd[k].shape
        with torch.no_grad():
          sd[k].copy_(sd_hf[k])

    return model
  
class DataloaderLite:
  """一个轻量级的 dataloader，直接从文本文件中读取数据"""
  def __init__(self, B, T):
    self.B = B
    self.T = T
    with open('input.txt', 'r') as f:
      text = f.read()
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(text)
    self.tokens = torch.tensor(tokens)
    print(f"Loaded {len(self.tokens)} tokens from input.txt")
    print(f"1 epoch = {len(self.tokens) // (B * T)} batches")
    # 记录读到哪了
    self.current_position = 0

  def next_batch(self):
    """返回一个 batch 的数据"""
    B, T = self.B, self.T
    buf = self.tokens[self.current_position: self.current_position + B * T + 1]
    x = buf[:-1].view(B, T) # 输入
    y = buf[1:].view(B, T)  # 目标（右移一位）
    self.current_position += B * T  # 光标前进B * T
    # 如果下一批会越界， 就从头开始
    if self.current_position + B * T + 1 > len(self.tokens):
      self.current_position = 0
    return x, y




if __name__ == "__main__":
  num_return_sequences = 5  # 生成的文本数量
  max_length = 30          # 生成的最大长度

  # choose a device (mps, cuda, cpu)
  device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
  print(f"Using device: {device}")
  
  # -------------------------------------------------------
  # 不同设备都选择相同的随机种子
  torch.manual_seed(42)
  if device == 'cuda':
      torch.cuda.manual_seed(42)

  train_loader = DataloaderLite(B=4, T=32)

  # model = GPT.from_pretrained('gpt2')
  model = GPT(GPTConfig())
  model.eval()  # 切换到评估模式
  model.to(device)  # 将模型移动到设备上

  optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
  for i in range(50):
      x, y = train_loader.next_batch()
      x, y = x.to(device), y.to(device)
      optimizer.zero_grad()
      logits, loss = model(x, y)
      loss.backward()
      optimizer.step()
      print(f"Step {i}, Loss: {loss.item()}")



  import sys; sys.exit(0)

  # -------------------------------------------------------

  # 用tikenizer将输入文本转换为模型的输入格式
  import tiktoken
  enc = tiktoken.get_encoding("gpt2")
  tokens = enc.encode("Hello, I'm a language model,") # 将输入文本编码为token
  tokens = torch.tensor(tokens, dtype=torch.long) # 将token转换为tensor
  tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # 重复输入以生成多条文本
  x = tokens.to(device) # 将输入tensor移动到设备上

  # 自回归生成： 依次添加一个token，直到达到最大长度
  torch.manual_seed(42)  # 设置随机种子以获得可重复的结果
  for _ in range(max_length):
    with torch.no_grad():
      x_cond = x[:, -model.config.block_size:]  # 取最后block_size个token作为输入
      logits, _ = model(x_cond)  # 前向传播得到logits
      logits = logits[:, -1, :] # 取最后一个token的logits B, T, C --> B, C
      probs = F.softmax(logits, dim=-1) # 计算概率分布
      topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1) # 取top-k个概率最大的token
      next_token = torch.multinomial(topk_probs, num_samples=1) # 从top-k中采样一个token
      next_token = torch.gather(topk_indices, -1, next_token) # 将采样的token映射回原始的token id
      x = torch.cat((x, next_token), dim=1) # 将采样的token添加到输入中，继续生成下一个token

  # 将生成的token转换回文本
  for i in range(num_return_sequences):
      tokens = x[i, :max_length].tolist()
      print(">", enc.decode(tokens))

