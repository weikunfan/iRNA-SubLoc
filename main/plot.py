import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix, \
    accuracy_score, recall_score, f1_score
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import os
import matplotlib.pyplot as plt
from util import util_file
import seaborn as sns
from transformers.models.bert.configuration_bert import BertConfig

from util.util_plot import visualize_attention

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
test_dataset, test_label, len_test = util_file.load_tsv_format_data('lncRNA_sublocation_TestSet.tsv')
# test_dataset, test_label, len_test = util_file.load_tsv_format_data('lncRNA_mus_test_data.tsv')
# test_dataset, test_label, len_test = util_file.load_tsv_format_data('mRNA_mus_test_data.tsv')
batch_size = 1
test_dataloader = construct_dataset(test_dataset, test_label, True,
                                              batch_size)
def __caculate_metric(pred_prob, label_pred, label_real):
    test_num = len(label_real)
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for index in range(test_num):
        if label_real[index] == 1:
            if label_real[index] == label_pred[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if label_real[index] == label_pred[index]:
                tn = tn + 1
            else:
                fp = fp + 1

    # Accuracy
    ACC = float(tp + tn) / test_num
    # Precision
    if tp + fp == 0:
        Precision = 0
    else:
        Precision = float(tp) / (tp + fp)

    # Sensitivity
    if tp + fn == 0:
        Recall = Sensitivity = 0
    else:
        Recall = Sensitivity = float(tp) / (tp + fn)

    # Specificity
    if tn + fp == 0:
        Specificity = 0
    else:
        Specificity = float(tn) / (tn + fp)

    # MCC
    if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) == 0:
        MCC = 0
    else:
        MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

    # F1-score
    if Recall + Precision == 0:
        F1 = 0
    else:
        F1 = 2 * Recall * Precision / (Recall + Precision)

    # ROC and AUC
    FPR, TPR, thresholds = roc_curve(label_real, pred_prob, pos_label=1)  # Default 1 is positive sample
    AUC = auc(FPR, TPR)

    # PRC and AP
    precision, recall, thresholds = precision_recall_curve(label_real, pred_prob, pos_label=1)
    AP = average_precision_score(label_real, pred_prob, average='macro', pos_label=1, sample_weight=None)


    performance = [ACC, Sensitivity, Specificity, AUC, MCC, F1]
    roc_data = [FPR, TPR, AUC]
    prc_data = [recall, precision, AP]
    return performance, roc_data, prc_data

def plot_roc_curve(label_real, pred_prob):
    # 计算FPR和TPR
    fpr, tpr, _ = roc_curve(label_real, pred_prob)
    roc_auc = auc(fpr, tpr)

    # 绘制ROC曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


def plot_confusion_matrix_with_metrics(label_real, label_pred):
    # Calculate the confusion matrix
    cm = confusion_matrix(label_real, label_pred)

    # Calculate metrics
    accuracy = accuracy_score(label_real, label_pred)
    sensitivity = recall_score(label_real, label_pred)
    specificity = recall_score(label_real, label_pred, pos_label=0)
    f1 = f1_score(label_real, label_pred)

    # Plot the confusion matrix as a heatmap
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',vmin=0,vmax=1)
    plt.xlabel('Predicted Label',fontweight='bold')
    plt.ylabel('True Label',fontweight='bold')
    plt.yticks(rotation=0,fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.title("iRNA-SubLoc")
    plt.savefig('iRNA-SubLoc_confusion_matrix.png',dpi=300)
    plt.savefig('iRNA-SubLoc_confusion_matrix.pdf',dpi=300)
    # plt.show()
def save_data_to_file(roc_data, prc_data, roc_filename='roc_data.csv', prc_filename='prc_data.csv'):
    # Ensure the dimensions match for ROC data
    fpr, tpr, auc_value = roc_data
    auc_array = np.full_like(fpr, auc_value)  # Create an array of the same length as fpr filled with auc_value
    roc_data_to_save = np.column_stack((fpr, tpr, auc_array))

    # Save ROC data
    np.savetxt(roc_filename, roc_data_to_save, delimiter=',', header='FPR,TPR,AUC', comments='')

    # Ensure the dimensions match for PRC data
    recall, precision, ap_value = prc_data
    ap_array = np.full_like(recall, ap_value)  # Create an array of the same length as recall filled with ap_value
    prc_data_to_save = np.column_stack((recall, precision, ap_array))

    # Save PRC data
    np.savetxt(prc_filename, prc_data_to_save, delimiter=',', header='Recall,Precision,AP', comments='')

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

def predict():
    model = DNABERT2()
    # state_dict = torch.load('mRNA, ACC[0.768].pt')
    state_dict = torch.load('lncRNA_ACC[0.771].pt')

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
    corrects = 0
    test_batch_num = 0
    test_sample_num = 0
    pred_prob = []
    label_pred = []
    label_real = []

    repres_list = []
    label_list = []
    with torch.no_grad():
        loop = tqdm((test_dataloader), total=len(test_dataloader), desc="testing")
        for batch in loop:
            data, label = batch
            logits, representation,all_attention_weights, all_attention_probs = model(data)
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
                tokens = model.tokenizer.convert_ids_to_tokens(input_ids.tolist())
                attention_scores = all_attention_probs[-1][i].cpu().detach().numpy()  # (12, 864, 864)

                # Visualize attention scores for the first token
                # visualize_attention(tokens, attention_scores[:, 0, :])
            repres_list.extend(representation.cpu().detach().numpy())
            label_list.extend(label.cpu().detach().numpy())

            pred_prob_all = F.softmax(logits, dim=1)  # 预测概率 [batch_size, class_num]
            pred_prob_positive = pred_prob_all[:, 1]  # 注意，极其容易出错
            pred_prob_sort = torch.max(pred_prob_all, 1)  # 每个样本中预测的最大的概率 [batch_size]
            pred_class = pred_prob_sort[1]  # 每个样本中预测的最大的概率所在的位置（类别） [batch_size]

            corrects += (pred_class == label).sum()
            test_sample_num += len(label)
            test_batch_num += 1
            pred_prob = pred_prob + pred_prob_positive.tolist()
            label_pred = label_pred + pred_class.tolist()
            label_real = label_real + label.tolist()

        performance, ROC_data, PRC_data = __caculate_metric(pred_prob, label_pred, label_real)

        avg_acc = 100.0 * corrects / test_sample_num
        print('Evaluation -   ACC: {:.4f}%({}/{})'.format(avg_acc, corrects, test_sample_num))
        print('\n[ACC,\tSE,\t\tSP,\t\tAUC,\tMCC,\tF1]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
                    performance[0], performance[1], performance[2], performance[3],
                    performance[4], performance[5]))

        # plot_roc_curve(label_real,pred_prob)
        plot_confusion_matrix_with_metrics(label_real, label_pred)
        # save_data_to_file(ROC_data, PRC_data, 'roc_data.csv', 'prc_data.csv')
        return performance, ROC_data, PRC_data, repres_list, label_list


if __name__ == '__main__':
    predict()