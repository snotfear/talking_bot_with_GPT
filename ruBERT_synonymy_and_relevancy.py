import json

import torch.utils.data
import torch
import torch.nn as nn
import torch.utils.data
import os
import random
import transformers
import replace_pronounce

'''Основной родительский класс, от него будет унаследован следующий'''
class RubertRelevancyDetector0(nn.Module):
    def __init__(self, device, arch, max_len, sent_emb_size):
        super(RubertRelevancyDetector0, self).__init__()
        self.max_len = max_len
        self.arch = arch

        if self.arch == 1:
            self.norm = torch.nn.BatchNorm1d(num_features=sent_emb_size)
            self.fc1 = nn.Linear(sent_emb_size*2, 20)
            self.fc2 = nn.Linear(20, 1)
        elif self.arch == 2:
            self.rnn1 = nn.LSTM(input_size=sent_emb_size, hidden_size=sent_emb_size, num_layers=1, batch_first=True)
            self.rnn2 = nn.LSTM(input_size=sent_emb_size, hidden_size=sent_emb_size, num_layers=1, batch_first=True)
            self.fc1 = nn.Linear(in_features=sent_emb_size*4, out_features=20)
            self.fc2 = nn.Linear(in_features=20, out_features=1)
        elif self.arch == 3:
            cnn_size = 100
            self.conv1 = nn.Conv1d(sent_emb_size, out_channels=cnn_size, kernel_size=3)
            self.conv2 = nn.Conv1d(sent_emb_size, out_channels=cnn_size, kernel_size=3)
            self.fc1 = nn.Linear(in_features=cnn_size*4, out_features=1)
        else:
            raise NotImplementedError()

        self.device = device
        self.to(device)

    def save_weights(self, weights_path):
        # !!! Не сохраняем веса rubert, так как они не меняются при обучении и одна и та же rubert используется
        # несколькими моделями !!!
        state = dict((k, v) for (k, v) in self.state_dict().items() if not k.startswith('bert_model'))
        torch.save(state, weights_path)

    def load_weights(self, weights_path):
        self.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.eval()
        return

    def forward_0(self, b1, b2):
        """b1 и b2 это результат инференса в rubert"""

        if self.arch == 1:
            w1 = b1.sum(dim=-2)
            w2 = b2.sum(dim=-2)

            z1 = self.norm(w1)
            z2 = self.norm(w2)

            #merged = torch.cat((z1, z2, torch.abs(z1 - z2)), dim=-1)
            #merged = torch.cat((z1, z2, torch.abs(z1 - z2), z1 * z2), dim=-1)
            merged = torch.cat((z1, z2), dim=-1)

            merged = self.fc1(merged)
            merged = torch.relu(merged)
            merged = self.fc2(merged)
            output = torch.sigmoid(merged)

        elif self.arch == 2:
            out1, (hidden1, cell1) = self.rnn1(b1)
            v1 = out1[:, -1, :]

            out2, (hidden2, cell2) = self.rnn2(b2)
            v2 = out2[:, -1, :]

            v_sub = torch.sub(v1, v2)
            v_mul = torch.mul(v1, v2)

            merged = torch.cat((v1, v2, v_sub, v_mul), dim=-1)

            merged = self.fc1(merged)
            merged = torch.sigmoid(merged)
            merged = self.fc2(merged)
            output = torch.sigmoid(merged)

        elif self.arch == 3:
            z1 = b1.transpose(1, 2).contiguous()
            z2 = b2.transpose(1, 2).contiguous()

            v1 = self.conv1(z1)
            v1 = torch.relu(v1).transpose(1, 2).contiguous()

            v2 = self.conv2(z2)
            v2 = torch.relu(v2).transpose(1, 2).contiguous()

            v_sub = torch.sub(v1, v2)
            v_mul = torch.mul(v1, v2)

            merged = torch.cat((v1, v2, v_sub, v_mul), dim=-1)
            net, _ = torch.max(merged, 1)
            net = self.fc1(net)
            output = torch.sigmoid(net)
        else:
            raise NotImplementedError()

        return output

    def pad_tokens(self, tokens):
        l = len(tokens)
        if l < self.max_len:
            return tokens + [0] * (self.max_len - l)
        elif l > self.max_len:
            return tokens[:self.max_len]
        else:
            return tokens


class RubertRelevancyDetector(RubertRelevancyDetector0):
    """Вариант с внутренним вызовом rubert"""
    def __init__(self, device, arch, max_len, sent_emb_size):
        super(RubertRelevancyDetector, self).__init__(device, arch, max_len, sent_emb_size)
        self.bert_tokenizer = None
        self.bert_model = None

    def forward(self, x1, x2):
        with torch.no_grad():
            b1 = self.bert_model(x1)[0]
            b2 = self.bert_model(x2)[0]

        return self.forward_0(b1, b2)

    def calc_relevancy1(self, premise, query, **kwargs):
        tokens1 = self.pad_tokens(self.bert_tokenizer.encode(premise))
        tokens2 = self.pad_tokens(self.bert_tokenizer.encode(query))

        z1 = torch.unsqueeze(torch.tensor(tokens1), 0).to(self.device)
        z2 = torch.unsqueeze(torch.tensor(tokens2), 0).to(self.device)
        y = self.forward(z1, z2)[0].item()
        return y

    def get_most_relevant(self, query, premises, nb_results=1):
        query_t1 = self.pad_tokens(self.bert_tokenizer.encode(query))

        res = []

        batch_size = 100
        while premises:
            premises_batch = premises[:batch_size]
            premises = premises[batch_size:]
            premises_tx = [self.pad_tokens(self.bert_tokenizer.encode(premise)) for premise, _ in premises_batch]
            query_tx = [query_t1 for _ in range(len(premises_batch))]

            z1 = torch.tensor(premises_tx).to(self.device)
            z2 = torch.tensor(query_tx).to(self.device)

            y = self.forward(z1, z2).squeeze()
            if len(y.shape) == 0:
                res.append((premises_batch[0][0], y.item()))
            else:
                delta = [(premise[0], yi.item()) for (premise, yi) in zip(premises_batch, y)]
                res.extend(delta)
        res = sorted(res, key=lambda z: -z[1])[:nb_results]
        return [x[0] for x in res], [x[1] for x in res]

def load_bert(bert_path, bert_tokenizer, bert_model):
    bert_tokenizer = transformers.BertTokenizer.from_pretrained(bert_path, do_lower_case=False)
    bert_model = transformers.BertModel.from_pretrained(bert_path)
    bert_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    bert_model.eval()

def load_model(bert_model, bert_tokenizer):
    models_dir = 'ruBert-base'
    with open(os.path.join(models_dir, 'pq_relevancy_rubert_model.cfg'), 'r') as f:
        cfg = json.load(f)
        relevancy_detector = RubertRelevancyDetector(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), **cfg)
        relevancy_detector.load_weights(os.path.join(models_dir, 'pq_relevancy_rubert_model.pt'))
        relevancy_detector.bert_model = bert_model
        relevancy_detector.bert_tokenizer = bert_tokenizer

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_path = 'ruBert-base'
    bert_tokenizer = transformers.BertTokenizer.from_pretrained(bert_path, do_lower_case=False)
    bert_model = transformers.BertModel.from_pretrained(bert_path)
    bert_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    bert_model.eval()

    with open(os.path.join(bert_path, 'pq_relevancy_rubert_model.cfg'), 'r') as f:
        cfg = json.load(f)
        relevancy_detector = RubertRelevancyDetector(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), **cfg)
        relevancy_detector.load_weights(os.path.join(bert_path, 'pq_relevancy_rubert_model.pt'))
        relevancy_detector.bert_model = bert_model
        relevancy_detector.bert_tokenizer = bert_tokenizer


    with open('facts.txt', 'r') as facts:
        memory_phrases = []
        for ind, i in enumerate(facts.readlines()):
            i = i[:-1]
            if len(i) > 2:
                if '|' in i:
                    sep_i = i.split(' | ')
                    i = random.choice(sep_i)
                memory_phrases.append((i, ind))

    text = 'расскажи о себе?'
    normalized_phrase_1 = replace_pronounce.total_replace(text)

    premises = []
    rels = []
    premises0, rels0 = relevancy_detector.get_most_relevant(normalized_phrase_1, memory_phrases, nb_results=3)

    for premise, premise_rel in zip(premises0, rels0):
        if premise_rel >= 0.8:
            premises.append(premise)
            rels.append(premise_rel)
    print(premises, 'premises')
    print(rels, 'rels')

