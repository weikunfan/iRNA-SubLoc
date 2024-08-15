import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import os
import seaborn as sns
from util import util_file
from transformers.models.bert.configuration_bert import BertConfig
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
tokenlen = 864 # [seq_len:4096]
cuda=True

class MyDataSet(Data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.label)


    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]
def construct_dataset(sequences, labels, cuda, batch_size):
    if cuda:
        labels = torch.cuda.LongTensor(labels)
    else:
        labels = torch.LongTensor(labels)
    dataset = MyDataSet(sequences, labels)
    data_loader = Data.DataLoader(dataset,
                                  batch_size=batch_size,
                                  drop_last=False,
                                  shuffle=True)
    print('len(data_loader)', len(data_loader))
    return data_loader
# test_dataset, test_label, len_test = util_file.load_tsv_format_data('../data/lncRNA_sublocation_TestSet.tsv')
# test_dataset, test_label, len_test = util_file.load_tsv_format_data('../data/mRNA_test_data.tsv')
# test_dataset, test_label, len_test = util_file.load_tsv_format_data('../data/mRNA_cyto_only.tsv')
test_dataset, test_label, len_test = util_file.load_tsv_format_data('../data/mRNA_nuc_only.tsv')
batch_size = 1
test_dataloader = construct_dataset(test_dataset, test_label, True,
                                              batch_size)


'''DNAbert2 model'''
class DNABERT2(nn.Module):
    def __init__(self):
        super(DNABERT2, self).__init__()
        # 加载预训练模型参数
        self.pretrainpath = '../pretrain/DNABERT2_attention'

        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrainpath)
        # self.bert = AutoModel.from_pretrained(self.pretrainpath, config=BertConfig.from_pretrained(self.pretrainpath))
        # 改了这里 输出注意力
        self.bert = AutoModel.from_pretrained(self.pretrainpath, trust_remote_code=True, output_attentions=True)

        self.classification = nn.Sequential(
            nn.Linear(768, 20),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(20, 2)
        )
    def forward(self, seqs):
        # print(seqs)
        seqs = list(seqs)

        token_seq = self.tokenizer(
            seqs,
            add_special_tokens=True,
            max_length=tokenlen,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        # print(token_seq)
        input_ids, token_type_ids, attention_mask = token_seq['input_ids'], token_seq['token_type_ids'], token_seq[
            'attention_mask']
        # 以下三个的维度都是[batchsize,toeknlen]
        # print('input_ids:',input_ids.shape)
        # print('token_type_ids:',token_type_ids.shape)
        # print('attention_mask:',attention_mask.shape)
        if cuda:
            # _,representation,_,_ = self.bert(input_ids.cuda(), token_type_ids.cuda(), attention_mask.cuda())
            _,representation,all_attention_weights, all_attention_probs = self.bert(input_ids.cuda(), token_type_ids.cuda(), attention_mask.cuda())

        else:
            # representation = self.bert(input_ids, token_type_ids, attention_mask)["pooler_output"]
            _,representation,all_attention_weights, all_attention_probs = self.bert(input_ids, token_type_ids, attention_mask)["pooler_output"]

        output = self.classification(representation)
        # print('output:',output)
        # print('representation:',representation)
        return output, representation ,all_attention_weights, all_attention_probs

# output_file = 'motifs_lncRNA.txt'
# output_file = 'motifs_mRNA.txt'
# output_file = 'mrna_cyto_motifs.txt'
output_file = 'mrna_nuc_motifs.txt'

def predict():
    model = DNABERT2()
    state_dict = torch.load('mRNA, ACC[0.768].pt')
    # state_dict = torch.load('lncRNA_ACC[0.771].pt')

    # 修改键名
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.bert.bert'):
            new_key = k.replace('module.bert.bert', 'bert')
        elif k.startswith('module.classification'):
            new_key = k.replace('module.classification', 'classification')
        elif k.startswith('bert.bert'):
            new_key = k.replace('bert.bert', 'bert')
        else:
            new_key = k
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)

    model.eval()
    model.cuda()

    with torch.no_grad():
        loop = tqdm((test_dataloader), total=len(test_dataloader), desc="testing")
        with open(output_file, 'w', encoding='utf-8') as f:
            for batch in loop:
                data, label = batch
                logits, representation,all_attention_weights, all_attention_probs = model(data)
                # print('all_attention_probs.shape',all_attention_probs.shape) # (batch_size,12,864,864) 12层，输出最后一层

                for i, seq in enumerate(data):
                    token_seq = model.tokenizer(
                        seq,
                        add_special_tokens=True,
                        max_length=tokenlen,
                        padding='max_length',
                        return_token_type_ids=True,
                        truncation=True,
                        return_attention_mask=True,
                        return_tensors='pt'
                    )
                    input_ids = token_seq['input_ids'][0]

                    attention_scores = all_attention_probs[-1][i].cpu().detach().numpy()  # (12,864,864)

                    # 合并每个位置的注意力分数
                    summed_attention_scores = np.sum(attention_scores, axis=0)  # (864,864)
                    summed_scores = np.sum(summed_attention_scores, axis=0).reshape(1, 864)
                    # print('summed_attention_scores.shape:',summed_attention_scores.shape)
                    # print('summed_scores.shape:',summed_scores.shape)
                    # print("max_attention_score.shape:",max_attention_score.shape)
                    sorted_indices = np.argsort(summed_scores)[0][::-1]  # 从大到小排序
                    top_tokens = []

                    # max_position = sorted_indices[0]
                    # max_attention_token = model.tokenizer.convert_ids_to_tokens(input_ids[max_position].item())

                    # 获取最大注意力分数的token
                    for idx in sorted_indices:
                        max_position = idx
                        max_attention_token = model.tokenizer.convert_ids_to_tokens(input_ids[max_position].item())
                        if max_attention_token not in ['[SEP]', '[CLS]'] and len(max_attention_token) >= 3 and summed_scores[0][max_position] > 700:
                            top_tokens.append(max_attention_token)
                            print(summed_scores[0][max_position])
                            break  # 只需要最大注意力分数的token

                    # 写入文件
                    for token in top_tokens:
                        if token:
                            f.write(token + '\n')






if __name__ == '__main__':
    predict()