import json
import time

from tokenizer import Tokenizer
import pickle

tokenizer = Tokenizer()

# 1470769
train_file = './LCSTS/train.json'
count_map = {}
all_content = []
max_len = 0

with open(train_file, 'r') as f:
    json_arr = json.loads(f.read())

    for idx in range(len(json_arr)):
        item = json_arr[idx]

        if idx % 10000 == 0:
            print(f'cur: {idx}, all: {len(json_arr)}')

        content = item['content']
        len_content = len(content)
        if len_content > max_len:
            max_len = len_content

        # summary = item['summary']

        all_content.append(tokenizer.encode_to_token(f'{content}[SEP]', False, covert_tensor=False))


print(f'dump pickle. max_len: {max_len}')

with open(f'{train_file}.pkl', 'wb') as pkl:
    pickle.dump(all_content, pkl)
