import os
os.environ['CUDA_VISIBLE_DEVICES']='2'
from transformers import AutoTokenizer, EsmForMaskedLM
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
import csv
from amp.utils import basic_model_serializer
import amp.data_utils.sequence as du_sequence
import seaborn as sns
import matplotlib.pyplot as plt
import random
import time

start_time = time.time()
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
# 修改字符串
def str_cat(s, index, a):
    new_seq = list(s)
    new_seq[index] = a
    new_seq1 = ''.join(new_seq)
    return new_seq1

# 数据加载
data = pd.read_csv("/geniusland/home/wanglijuan/sci_proj/ESM2_opt/src_datasets/case3.csv")
seqs = data['Sequence']


# 加载打分模型 
bms = basic_model_serializer.BasicModelSerializer()
amp_classifier = bms.load_model('/geniusland/home/wanglijuan/sci_proj/models/amp_classifier')
amp_classifier_model = amp_classifier()
mic_classifier = bms.load_model('/geniusland/home/wanglijuan/sci_proj/models/mic_classifier/')
mic_classifier_model = mic_classifier() 
# 加载预测模型
tokenizer = AutoTokenizer.from_pretrained("/geniusland/home/wanglijuan/sci_proj/facebookesm2_t33_650M_UR50D")
model = EsmForMaskedLM.from_pretrained("/geniusland/home/wanglijuan/sci_proj/facebookesm2_t33_650M_UR50D")
model = model.cuda()
model.eval()
r_list = []
esm2_list = []
r_amp_list = []
r_mic_list = []
esm_amp_list = []
esm_mic_list = []
cnt = 5
# k = 5
for seq in tqdm(seqs):
    r_dict = {}
    e_dict = {}
    l = len(seq)
    if l < cnt:
        continue
    for j in range(l):
        r_seq = list(seq)
        esm_seq = list(seq)
        positions = random.sample([h for h in range(l)], cnt)
        for i in positions:
            r_seq[i] = random.choice(AMINO_ACIDS)
            esm_seq[i] = tokenizer.mask_token  # 用掩码标记替换
            masked_sequence = ''.join(esm_seq)
        # sequences.append(masked_sequence)

        # 对所有序列进行标记化
        inputs = tokenizer(masked_sequence, return_tensors="pt", padding=True)
        inputs = inputs.to('cuda')

        with torch.no_grad():
            logits = model(**inputs).logits  # [batch_size, seq_len, vocab_size]
            logits = logits.detach().cpu()
        mask_token_index = (inputs.to('cpu').input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]  # 找到序列中mask的id
        predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
        # import pdb; pdb.set_trace()
        pred_token = tokenizer.convert_ids_to_tokens(predicted_token_id)
        # if pred_token[0] not in AMINO_ACIDS:
        #     pred_token[0] = 'K'
        # esm_seq[j] = pred_token[0]
        # print(pred_token)
        for i in (range(cnt)):
            if pred_token[i] not in AMINO_ACIDS:
                pred_token[i] = 'K'
        k = 0
        for i in positions:
            esm_seq[i] = pred_token[k]
            k += 1
        tmp = []
        tmp.append(''.join(r_seq))
        tmp.append(''.join(esm_seq))
        # 预测mic值
        pad_seq = du_sequence.pad(du_sequence.to_one_hot(tmp))
        pred_amp = amp_classifier_model.predict(pad_seq)
        pred_mic = mic_classifier_model.predict(pad_seq)
        r_dict[tmp[0]] = pred_mic[0][0] + pred_amp[0][0]
        e_dict[tmp[1]] = pred_mic[1][0] + pred_amp[1][0]
    r_max_seq = max(r_dict, key=r_dict.get)
    e_max_seq = max(e_dict, key=e_dict.get)
    tmp = []
    tmp.append(r_max_seq)
    tmp.append(e_max_seq)
    r_list.append(r_max_seq)
    esm2_list.append(e_max_seq)
    # 预测mic值
    pad_seq = du_sequence.pad(du_sequence.to_one_hot(tmp))
    pred_amp = amp_classifier_model.predict(pad_seq)
    pred_mic = mic_classifier_model.predict(pad_seq)
    r_amp_list.append(pred_amp[0][0])
    r_mic_list.append(pred_mic[0][0])
    esm_amp_list.append(pred_amp[1][0])
    esm_mic_list.append(pred_mic[1][0])

data1 = {
    'Sequence': r_list,
    'AMP': r_amp_list,
    'MIC': r_mic_list
}
data2 = {
    'Sequence': esm2_list,
    'AMP': esm_amp_list,
    'MIC': esm_mic_list
}
df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)
df1.to_csv('/geniusland/home/wanglijuan/sci_proj/ESM2_opt/AMP_Methods_Compare/case3/random/m5_uu.csv', index=False)
df2.to_csv('/geniusland/home/wanglijuan/sci_proj/ESM2_opt/AMP_Methods_Compare/case3/esm2/m5_uu.csv', index=False)

end_time = time.time() 
print(f"程序运行时间：{end_time - start_time} 秒")  