import json
import time

from tokenizer import Tokenizer
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch

tokenizer = Tokenizer()

# 1470769
train_file = './LCSTS/train.json'
count_map = {}
all_content = []

"""
###总结以下内容
###输入:新闻内容
###输出:总结
"""

with open(train_file, 'r') as f:
    json_arr = json.loads(f.read())

    for idx in range(len(json_arr)):
        item = json_arr[idx]

        if idx % 10000 == 0:
            print(f'cur: {idx}, all: {len(json_arr)}')

        content = f"###总结以下内容###输入:{item['content']}###输出:{item['summary']}"
        all_content.append(tokenizer.encode_to_token(f'{content}[SEP]', False, covert_tensor=False))


print('dump pickle.')

with open(f'{train_file}_sft.pkl', 'wb') as pkl:
    pickle.dump(all_content, pkl)