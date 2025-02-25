from configuration import config_init
from transformers import AutoTokenizer, AutoModel
import os
from transformers.models.bert.configuration_bert import BertConfig
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def load_easy_fasta(filename, skip_first=False):
    with open(filename, 'r') as file:
        content = file.read()
    content_split = content.split('\n')

    seqs = []
    for index, record in enumerate(content_split):
        if index % 2 == 1:
            seqs.append(content_split[index])

    return seqs


def load_fasta(filename, skip_first=False):
    with open(filename, 'r') as file:
        content = file.read()
    content_split = content.split('\n')

    train_dataset = []
    train_label = []
    test_dataset = []
    test_label = []
    for index, record in enumerate(content_split):
        if index % 2 == 1:
            continue
        recordsplit = record.split('|')
        if recordsplit[-1] == 'training':
            train_label.append(int(recordsplit[-2]))
            train_dataset.append(content_split[index + 1])
        if recordsplit[-1] == 'testing':
            test_label.append(int(recordsplit[-2]))
            test_dataset.append(content_split[index + 1])
    return train_dataset, train_label, test_dataset, test_label


def load_tsv_format_data(filename, skip_head=True):
    sequences = []
    labels = []

    with open(filename, 'r') as file:
        if skip_head:
            next(file)
        for line in file:
            if line[-1] == '\n':
                line = line[:-1]
            list = line.split('\t')
            sequences.append(list[2])
            labels.append(int(list[3]))
        length = [len(seq) for seq in sequences]
        # sequences = [padding_truncate_seq(seq) for seq in sequences]
        sequences = [truncate_sequence(seq,max_length) for seq in sequences]
        max_len = getmaxtokenizerlen(sequences)
        print("max_len:",max_len)
    return sequences, labels, length

path = "../pretrain/DNABERT2-pretrain"
def getmaxtokenizerlen(X):
    maxlen = 0
    max_sequence = None
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    for each in X:
        templen=tokenizer(each,return_tensors='pt')["input_ids"].shape[1]
        if templen > maxlen:
            maxlen=templen
            max_sequence = each
    print('Max length of tokenizer:')
    print(maxlen)
    # print('The longest sequence:')
    # print(max_sequence)
    return maxlen

def read_csv(filename, skip_head=True):
    sequences = []
    labels = []
    with open(filename, 'r') as file:
        if skip_head:
            next(file)
        for line in file:
            if line[-1] == '\n':
                line = line[:-1]
            list = line.split(',')
            feature = []
            for i in range(len(list) - 1):
                feature.append(int(list[i]))
            sequences.append(feature)
            labels.append(int(list[-1]))

    return sequences, labels


def load_txt_data(filename, skip_head=True):
    sequences = []

    with open(filename, 'r') as file:
        if skip_head:
            next(file)
        for line in file:
            if line[-1] == '\n':
                line = line[:-1]
            sequences.append(line)

    return sequences


def load_tsv_po_ne_data(filename, skip_head=True):
    negsequences = []
    possequences = []
    labels = []

    with open(filename, 'r') as file:
        if skip_head:
            next(file)
        for line in file:
            if line[-1] == '\n':
                line = line[:-1]
            list = line.split('\t')
            if list[1] == '1':
                possequences.append(list[2])
            elif list[1] == '0':
                negsequences.append(list[2])

    return possequences, negsequences


max_length = 4096
# max_length = 3000
#
def truncate_sequence(sequence, max_length):
    # print(len(sequence))
    if len(sequence) <= max_length:
        return sequence
    else:
        return sequence[:max_length]

def padding_truncate_seq(seq):
    length = len(seq)
    base = 1024
    if length > base:
        seq = seq[:base]
    else:
        seq = seq + (base-length)*'N'
    return seq

if __name__ == '__main__':
    config = config_init.get_config()

