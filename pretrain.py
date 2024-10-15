import torch
from torch.utils.data import Dataset
import torch.distributed as dist
import os, time
import pickle
from gpt import GPT
from utils import (
    gradient_accumulation_steps,
    CFG,
    CosineAnnealingWarmupScheduler,
    generate,
    ddp_helper,
    calc_loss,
    padding_fn
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
pre_train_file_path = './LCSTS/train.json.pkl'


class PreTrainDataset(Dataset):
    def __init__(self, ctx_len):
        super().__init__()
        self.ctx_len = ctx_len

        start_load_time = time.time()
        with open(pre_train_file_path, 'rb') as f:
            self.tokens = pickle.load(f)

        print(f'load data time: {time.time() - start_load_time}')

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, item):
        inputs = self.tokens[item]
        inputs = inputs[:self.ctx_len]
        return torch.tensor(inputs).long()


def train(n_epochs, batch_size):
    pass_epoch_count = 0

    model = ddp_helper.process_model(GPT(CFG), 'gpt.pth')

    train_data_loader = ddp_helper.create_dataloader(
        dataset=PreTrainDataset(CFG['ctx_len']),
        batch_size=batch_size,
        collate_fn=padding_fn
    )

    batch_count = len(train_data_loader)
    train_iters = batch_count * n_epochs

    warmup_iters = int(0.2 * train_iters)
    # 学习率要根据GPU的数量进行倍增：
    # 在训练的过程中，损失梯度决定下降的方向，学习率决定下降的步长。如果有两块gpu，前进的综合步长为：平均学习率*2
    initial_lr = 1e-5 * ddp_helper.world_size()
    min_lr = 0.1 * initial_lr
    max_lr = 5e-4 * ddp_helper.world_size()

    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=0.1)
    lr_scheduler = CosineAnnealingWarmupScheduler(warmup_iters, initial_lr, min_lr, max_lr, train_iters)

    if pass_epoch_count != 0:
        lr_scheduler.update_steps(pass_epoch_count * len(train_data_loader))
        lr_scheduler.update_lr(optimizer)

    if ddp_helper.is_main_process():
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total number of parameters: {total_params:,}")

        total_size_bytes = total_params * 4
        total_size_mb = total_size_bytes / (1024 * 1024)
        print(f"Total size of the model: {total_size_mb:.2f} MB")

    for epoch in range(n_epochs):
        if epoch < pass_epoch_count:
            print(f'pass epoch {epoch}')
            continue

        loss_accumulation = torch.tensor(0.0, device=ddp_helper.device)
        ddp_helper.on_epoch(epoch)
        model.train()

        for batch, (inputs, labels) in enumerate(train_data_loader):
            # 是否需要更新梯度
            if gradient_accumulation_steps > 1:
                need_update_grad = (batch + 1) % gradient_accumulation_steps == 0 or batch == batch_count - 1
            else:
                need_update_grad = True

            try:
                inputs, labels = inputs.to(ddp_helper.device), labels.to(ddp_helper.device)

                if ddp_helper.ddp:
                    # in DDP training we only need to sync gradients at the last micro step.
                    # the official way to do this is with model.no_sync() context manager, but
                    # I really dislike that this bloats the code and forces us to repeat code
                    # looking at the source of that context manager, it just toggles this variable
                    model.require_backward_grad_sync = need_update_grad

                with torch.autocast(device_type=ddp_helper.device_type, dtype=torch.bfloat16):
                    logits = model(inputs)

                # calc loss
                loss = calc_loss(logits, labels)
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                loss_accumulation += loss.detach()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                lr_scheduler.incr_steps()

                if need_update_grad:
                    lr_scheduler.update_lr(optimizer)
                    optimizer.step()
                    optimizer.zero_grad()

                if ddp_helper.is_main_process():
                    if batch % 100 == 0:
                        print(f"epoch: {epoch}, batch: {batch}/{len(train_data_loader)}")
                        with open('./batch.txt', 'a') as f:
                            f.write(f"epoch: {epoch}, batch: {batch}/{len(train_data_loader)}, need_update_grad:{need_update_grad}\n")

                    if batch % 1000 == 0:
                        gen1 = generate(model, '一辆小轿车，一名女司机，竟造成9死24伤。', 100)
                        with open('./gen.txt', 'a') as f:
                            f.write(f"{gen1}\n")

                        model.train()
                        ckpt = {'model': ddp_helper.raw_model.state_dict()}
                        torch.save(ckpt, 'gpt.pth')
            except KeyboardInterrupt:
                if ddp_helper.is_main_process():
                    model.train()
                    ckpt = {'model': ddp_helper.raw_model.state_dict()}
                    torch.save(ckpt, 'gpt.pth')
                    exit(0)
            except Exception as e:
                if ddp_helper.is_main_process():
                    with open('./batch.txt', 'a') as f:
                        f.write(f"epoch: {epoch}, batch: {batch}/{len(train_data_loader)}, {e}\n")

        if ddp_helper.ddp:
            dist.all_reduce(loss_accumulation, dist.ReduceOp.AVG)

        ddp_helper.end_epoch(epoch)
        if ddp_helper.is_main_process():
            if os.path.exists('gpt.pth'):
                os.rename('gpt.pth', 'gpt_backup.pth')

            ckpt = {'model': ddp_helper.raw_model.state_dict()}
            torch.save(ckpt, 'gpt.pth')

            # test_loss = test_loop(model, test_data_loader)
            print(f'train_loss: {loss_accumulation.item()/batch_count}')
            with open('./batch.txt', 'a') as f:
                f.write(f"epoch: {epoch}, loss: {loss_accumulation.item()/batch_count}, need_update_grad:{need_update_grad}\n")

            gen1 = generate(model, '一辆小轿车，一名女司机，竟造成9死24伤。', 100)
            with open('./gen.txt', 'a') as f:
                f.write(f"{gen1}\n")

    ddp_helper.destroy()


if __name__ == '__main__':
    train(10, 4)
