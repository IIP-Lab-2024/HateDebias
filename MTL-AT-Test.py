import json
import torch
import random
import gc
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertConfig, BertTokenizerFast, get_linear_schedule_with_warmup
import pandas
import os
import warnings
import csv
import evaluator
import glob
warnings.filterwarnings('ignore')


class GlobalConfig:
    def __init__(self):
        self.seed = 2020
        self.path = Path('./data/')
        self.max_length = 32
        self.bert_path = 'bert-base-uncased'  # @param
        self.num_workers = os.cpu_count()
        self.batch_size = 32
        self.steps_show = 100
        self.accum_steps = 1
        num_epochs = 10  # @param
        self.epochs = num_epochs
        self.warmup_steps = 0
        lr = 2e-5  # @param
        self.lr = lr  # modified from 3e-5
        self.weight = 0.05
        run_id = "MTL-AT"  # @param
        self.offline = True
        self.saved_model_path = run_id


def seed_everything(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def move_to_device(x, device):
    if callable(getattr(x, 'to', None)): return x.to(device)
    if isinstance(x, (tuple, list)):
        return [move_to_device(o, device) for o in x]
    elif isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    return x


class TweetBertDataset(Dataset):
    def __init__(self, df, tokenizer, max_length, is_testing=False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_testing = is_testing

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, ix):
        text = self.df.iloc[ix]['text']
        input_ids = tokenizer.encode(text)[:-1]
        input_ids = input_ids[:self.max_length - 1]
        input_ids = input_ids + [102]
        attn_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)

        # PAD
        pad_len = self.max_length - len(input_ids)
        input_ids += [0] * pad_len
        attn_mask += [0] * pad_len
        token_type_ids += [0] * pad_len

        input_ids, attn_mask, token_type_ids = map(torch.LongTensor,
                                                   [input_ids, attn_mask, token_type_ids])
        encoded_dict = {
            'input_ids': input_ids,
            'attn_mask': attn_mask,
            'token_type_ids': token_type_ids,
        }
        encoded_dict['label'] = torch.tensor(all_labels.index(self.df.iloc[ix]['label']), dtype=torch.long)

        if not self.is_testing:
            encoded_dict['gender'] = torch.tensor(all_labels.index(self.df.iloc[ix]['gender']), dtype=torch.long)
            encoded_dict['age'] = torch.tensor(all_labels.index(self.df.iloc[ix]['age']), dtype=torch.long)
            encoded_dict['country'] = torch.tensor(all_labels.index(self.df.iloc[ix]['country']), dtype=torch.long)
            encoded_dict['ethnicity'] = torch.tensor(all_labels.index(self.df.iloc[ix]['ethnicity']), dtype=torch.long)
        return encoded_dict


class SentimentBertModel(nn.Module):
    def __init__(self, bert_path):
        super().__init__()

        bert_config = BertConfig.from_pretrained(bert_path)
        bert_config.output_hidden_states = True
        self.bert = BertModel.from_pretrained(bert_path, config=bert_config)
        self.dropout = nn.Dropout(0.5)

        self.classifier = nn.Linear(bert_config.hidden_size, 2)
        torch.nn.init.normal_(self.classifier.weight, std=0.02)

        self.gender_at = nn.Linear(bert_config.hidden_size, 2)
        self.age_at = nn.Linear(bert_config.hidden_size, 2)
        self.country_at = nn.Linear(bert_config.hidden_size, 2)
        self.ethnicity_at = nn.Linear(bert_config.hidden_size, 2)

    def forward(self, input_ids, attn_mask, token_type_ids):
        pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids
        )['last_hidden_state']
        all_sentence_v = pooled_output[:, 0, :]
        all_sentence_v = self.dropout(all_sentence_v)
        start_logits = self.classifier(all_sentence_v)

        gender_logits = self.gender_at(all_sentence_v)
        age_logits = self.age_at(all_sentence_v)
        country_logits = self.country_at(all_sentence_v)
        ethnicity_logits = self.ethnicity_at(all_sentence_v)
        return start_logits, all_sentence_v, gender_logits, age_logits, country_logits, ethnicity_logits


def test_eval(val_iter, model):
    model.eval()
    logits_list = []
    label_list = []
    for batch in val_iter:
        input_ids, attn_mask, token_type_ids, label = batch['input_ids'], batch['attn_mask'], batch['token_type_ids'], batch['label']

        if torch.cuda.is_available():
            input_ids, attn_mask, token_type_ids, label = input_ids.to(DEVICE), attn_mask.to(DEVICE), token_type_ids.to(DEVICE), label.to(DEVICE)
        with torch.no_grad():
            logits, all_sentence_v, gender_logits, age_logits, country_logits, ethnicity_logits = model(input_ids, attn_mask, token_type_ids)

        logits = torch.max(logits.data, 1)[1].cpu()
        label = label.cpu()

        logits_list.extend(logits.tolist())
        label_list.extend(label.tolist())
    f1 = f1_score(logits_list, label_list, average='macro')
    model.train()
    return f1


def train_function(model, age_train_dl, country_train_dl, gender_train_dl, ethnicity_train_dl, test_iter):
    if torch.cuda.is_available():
        model.to(DEVICE)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_parameters = [
        {
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': 1e-3
        },
        {
            'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.
        }
    ]
    optimizer = optim.AdamW(optimizer_parameters, lr=GCONF.lr)
    train_steps = ((len(age_train_dl) + len(country_train_dl) + len(gender_train_dl) + len(ethnicity_train_dl)) * GCONF.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(GCONF.warmup_steps * train_steps),
        num_training_steps=train_steps
    )
    steps = 0
    best_f1 = 0
    model.train()

    for epoch in range(1, GCONF.epochs + 1):
        for batch in age_train_dl:
            input_ids, attn_mask, token_type_ids, label, gender, age, country, ethnicity = batch['input_ids'], batch[
                'attn_mask'], batch['token_type_ids'], batch['label'], batch['gender'], batch['age'], batch['country'], batch['ethnicity']

            if torch.cuda.is_available():
                input_ids, attn_mask, token_type_ids, label, gender, age, country, ethnicity = input_ids.to(DEVICE), attn_mask.to(DEVICE), token_type_ids.to(DEVICE), label.to(DEVICE), gender.to(DEVICE), age.to(DEVICE), country.to(DEVICE), ethnicity.to(DEVICE)

            logits, all_sentence_v, gender_logits, age_logits, country_logits, ethnicity_logits = model(input_ids, attn_mask, token_type_ids)
            loss = nn.CrossEntropyLoss()
            loss_at = nn.CrossEntropyLoss()
            loss = loss(logits, label)
            loss_at = loss_at(age_logits, age)
            loss = loss - GCONF.weight * loss_at
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            steps += 1
            logits = torch.max(logits.data, 1)[1].cpu()
            label = label.cpu()

            if steps % GCONF.steps_show == 0:
                logits_list = logits.tolist()
                label_list = label.tolist()
                f1 = f1_score(logits_list, label_list, average='macro')

                print('epoch:%d\t\t\tsteps:%d\t\t\tloss:%.6f\t\t\tf1_score:%.4f' % (epoch, steps, loss.item(), f1))

        for batch in country_train_dl:
            input_ids, attn_mask, token_type_ids, label, gender, age, country, ethnicity = batch['input_ids'], batch[
                'attn_mask'], batch['token_type_ids'], batch['label'], batch['gender'], batch['age'], batch['country'], batch['ethnicity']

            if torch.cuda.is_available():
                input_ids, attn_mask, token_type_ids, label, gender, age, country, ethnicity = input_ids.to(DEVICE), attn_mask.to(DEVICE), token_type_ids.to(DEVICE), label.to(DEVICE), gender.to(DEVICE), age.to(DEVICE), country.to(DEVICE), ethnicity.to(DEVICE)

            logits, all_sentence_v, gender_logits, age_logits, country_logits, ethnicity_logits = model(input_ids, attn_mask, token_type_ids)
            loss = nn.CrossEntropyLoss()
            loss_at = nn.CrossEntropyLoss()
            loss = loss(logits, label)
            loss_at = loss_at(country_logits, country)
            loss = loss - GCONF.weight * loss_at
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            steps += 1
            logits = torch.max(logits.data, 1)[1].cpu()
            label = label.cpu()

            if steps % GCONF.steps_show == 0:
                logits_list = logits.tolist()
                label_list = label.tolist()
                f1 = f1_score(logits_list, label_list, average='macro')

                print('epoch:%d\t\t\tsteps:%d\t\t\tloss:%.6f\t\t\tf1_score:%.4f' % (epoch, steps, loss.item(), f1))

        for batch in gender_train_dl:
            input_ids, attn_mask, token_type_ids, label, gender, age, country, ethnicity = batch['input_ids'], batch[
                'attn_mask'], batch['token_type_ids'], batch['label'], batch['gender'], batch['age'], batch['country'], batch['ethnicity']

            if torch.cuda.is_available():
                input_ids, attn_mask, token_type_ids, label, gender, age, country, ethnicity = input_ids.to(DEVICE), attn_mask.to(DEVICE), token_type_ids.to(DEVICE), label.to(DEVICE), gender.to(DEVICE), age.to(DEVICE), country.to(DEVICE), ethnicity.to(DEVICE)

            logits, all_sentence_v, gender_logits, age_logits, country_logits, ethnicity_logits = model(input_ids, attn_mask, token_type_ids)
            loss = nn.CrossEntropyLoss()
            loss_at = nn.CrossEntropyLoss()
            loss = loss(logits, label)
            loss_at = loss_at(gender_logits, gender)
            loss = loss - GCONF.weight * loss_at
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            steps += 1
            logits = torch.max(logits.data, 1)[1].cpu()
            label = label.cpu()

            if steps % GCONF.steps_show == 0:
                logits_list = logits.tolist()
                label_list = label.tolist()
                f1 = f1_score(logits_list, label_list, average='macro')

                print('epoch:%d\t\t\tsteps:%d\t\t\tloss:%.6f\t\t\tf1_score:%.4f' % (epoch, steps, loss.item(), f1))

        for batch in ethnicity_train_dl:
            input_ids, attn_mask, token_type_ids, label, gender, age, country, ethnicity = batch['input_ids'], batch[
                'attn_mask'], batch['token_type_ids'], batch['label'], batch['gender'], batch['age'], batch['country'], batch['ethnicity']

            if torch.cuda.is_available():
                input_ids, attn_mask, token_type_ids, label, gender, age, country, ethnicity = input_ids.to(DEVICE), attn_mask.to(DEVICE), token_type_ids.to(DEVICE), label.to(DEVICE), gender.to(DEVICE), age.to(DEVICE), country.to(DEVICE), ethnicity.to(DEVICE)

            logits, all_sentence_v, gender_logits, age_logits, country_logits, ethnicity_logits = model(input_ids, attn_mask, token_type_ids)
            loss = nn.CrossEntropyLoss()
            loss_at = nn.CrossEntropyLoss()
            loss = loss(logits, label)
            loss_at = loss_at(ethnicity_logits, ethnicity)
            loss = loss - GCONF.weight * loss_at
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            steps += 1
            logits = torch.max(logits.data, 1)[1].cpu()
            label = label.cpu()

            if steps % GCONF.steps_show == 0:
                logits_list = logits.tolist()
                label_list = label.tolist()
                f1 = f1_score(logits_list, label_list, average='macro')

                print('epoch:%d\t\t\tsteps:%d\t\t\tloss:%.6f\t\t\tf1_score:%.4f' % (epoch, steps, loss.item(), f1))

        print('dev')
        dev_f1 = test_eval(test_iter, model)
        print('dev\nf1:%.6f' % (dev_f1))
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            torch.save(model, GCONF.saved_model_path + '/model.pth')
            print('save best model\t\tf1:%.6f' % best_f1)


def test_eval(val_iter, model):
    model.eval()
    logits_list = []
    label_list = []
    probs_list = []
    for batch in val_iter:
        input_ids, attn_mask, token_type_ids, label = batch['input_ids'], batch['attn_mask'], batch['token_type_ids'], \
                                                      batch['label']

        if torch.cuda.is_available():
            input_ids, attn_mask, token_type_ids, label = input_ids.to(DEVICE), attn_mask.to(DEVICE), token_type_ids.to(DEVICE), label.to(DEVICE)
        with torch.no_grad():
            logits, all_sentence_v, gender_logits, age_logits, country_logits, ethnicity_logits = model(input_ids, attn_mask, token_type_ids)

        probs_list.extend(logits.data[:,1].cpu().tolist())
        logits = torch.max(logits.data, 1)[1].cpu()
        label = label.cpu()

        logits_list.extend(logits.tolist())
        label_list.extend(label.tolist())

    with open('Answer.tsv', 'w') as wfile:
        with open('dataset/test.tsv') as dfile:
            wfile.write(
                dfile.readline().strip() + '\tpred\tpred_prob\n')
            for idx, line in enumerate(dfile):
                wfile.write(line.strip() + '\t' + str(logits_list[idx]) + '\t' + str(probs_list[idx]) + '\n')
    evaluator.eval('Answer.tsv')
    f1 = f1_score(logits_list, label_list, average='macro')
    model.train()
    return f1

GCONF = GlobalConfig()
if not os.path.exists(GCONF.saved_model_path):
    os.mkdir(GCONF.saved_model_path)

seed_everything(GCONF.seed)
all_labels = [0, 1]
tokenizer = BertTokenizerFast.from_pretrained(GCONF.bert_path, do_lower_case=True, add_prefix_space=True,
                                              is_split_into_words=True, truncation=True)

age_train_df = pd.read_csv('dataset/age.tsv', sep='\t')
age_train_ds = TweetBertDataset(age_train_df, tokenizer, GCONF.max_length,is_testing=False)
age_train_dl = DataLoader(age_train_ds, batch_size=GCONF.batch_size, shuffle=True)

country_train_df = pd.read_csv('dataset/country.tsv', sep='\t')
country_train_ds = TweetBertDataset(country_train_df, tokenizer, GCONF.max_length,is_testing=False)
country_train_dl = DataLoader(country_train_ds, batch_size=GCONF.batch_size, shuffle=True)

gender_train_df = pd.read_csv('dataset/gender.tsv', sep='\t')
gender_train_ds = TweetBertDataset(gender_train_df, tokenizer, GCONF.max_length,is_testing=False)
gender_train_dl = DataLoader(gender_train_ds, batch_size=GCONF.batch_size, shuffle=True)

ethnicity_train_df = pd.read_csv('dataset/ethnicity.tsv', sep='\t')
ethnicity_train_ds = TweetBertDataset(ethnicity_train_df, tokenizer, GCONF.max_length,is_testing=False)
ethnicity_train_dl = DataLoader(ethnicity_train_ds, batch_size=GCONF.batch_size, shuffle=True)

val_df = pd.read_csv('dataset/test.tsv', sep='\t', na_values='x', quoting=csv.QUOTE_NONE)
val_ds = TweetBertDataset(val_df, tokenizer, GCONF.max_length, is_testing=True)
val_dl = DataLoader(val_ds, batch_size=GCONF.batch_size, shuffle=False)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for filename in glob.glob('Sequence-AT-SensitiveReplay-0.15-0.1*'):
    GCONF.saved_model_path = filename
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(GCONF.saved_model_path+'/model.pth')
    model.to(DEVICE)
    print(filename,end='\t')
    test_eval(val_dl,model)
