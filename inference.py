import os
import torch
from utils import generate

summary = True

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"

os.environ["TOKENIZERS_PARALLELISM"] = "false"
gpt = torch.load('modeling.pth')
gpt.to(device)

while True:
    inputs = input('\n输入内容：').strip()
    print(f'字数：{len(inputs)}')
    if summary:
        inputs = f'###总结以下内容###输入:{inputs}###输出:'

    print('总结: ', end='')
    generate(gpt, inputs, 256, item_callback=lambda item: print(item, end=''))

