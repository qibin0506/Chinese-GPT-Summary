# GPTSummary
Training GPT Summary Model from Scratch

## Inference
Download the [modeling.pth](https://github.com/qibin0506/ChineseGPTSummary/releases/tag/release) and run inferrence.py
``` python
python inferrence.py
```
*Note: As the maximum length of the model context is 256, try to keep the input content length within 150 words*

<img width="1356" alt="inference" src="https://github.com/user-attachments/assets/feda028b-7732-4e17-92b9-55ac7d9a106b">


## Train
Training includes pretrain and sft.

### Pretrain
Process your dataset by modifying the data_preprocess.py file, then modify the relevant PreTrainDataset in pertrain.py to fit your dataset, and run pretrain.py for pretrain.

### SFT
Process your dataset by modifying the data_preprocess_sft.py file, then modify the relevant LCSTSDataset in sft.py to fit your dataset, and run sft.py for SFT.
