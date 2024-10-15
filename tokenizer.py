from transformers import BertTokenizerFast
import torch


class Tokenizer:
    def __init__(self, vocab_path='./vocab.txt'):
        super().__init__()

        self.tokenizer = BertTokenizerFast(vocab_path)
        self.vocab_size = self.tokenizer.vocab_size
        self.eot = self.tokenizer.sep_token_id
        self.unk = self.tokenizer.unk_token_id

    def encode_to_token(self, text: str, unsqueeze=True, covert_tensor=True):
        # 将原始空格替换成[PAD]
        text = text.replace(' ', '[PAD]')

        # [1, 2]
        encoded = self.tokenizer.encode(text, add_special_tokens=False)

        if unsqueeze:
            # tensor: [[1, 2]]
            return torch.tensor(encoded).long().unsqueeze(0)
        else:
            # tensor: [1, 2]
            if covert_tensor:
                return torch.tensor(encoded).long()

            return encoded

    def decode_to_text(self, token: torch.Tensor) -> str:
        return (self.tokenizer.decode(token.squeeze(0))
                .replace(' ', '')
                .replace('[PAD]', ' '))


if __name__ == '__main__':
    tokenizer = Tokenizer()
    print(tokenizer.eot)
    print(tokenizer.unk)
    print(tokenizer.encode_to_token('hello 你好啊[UNK][SEP]', unsqueeze=False))
    print(tokenizer.decode_to_text(tokenizer.encode_to_token('hello 你好啊')))

