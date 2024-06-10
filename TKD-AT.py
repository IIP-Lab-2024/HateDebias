"""
Usage:
    main.py [options]

Options:
    -h --help                         show this screen
    --attribute1=<str>              attribute1 [default: country]
    --attribute2=<str>              attribute2 [default: age]
    --attribute3=<str>              attribute3 [default: gender]
    --attribute4=<str>              attribute4 [default: ethnicity]
    --save_path=<str>               save_path [default: Sequence-AT-SensitiveReplay-0.05-0.05cage]
    --weight=<float>                weight [default: 0.05]
    --proportion=<float>            proportion [default: 0.05]
    --task_weight=<float>                weight [default: 0.05]
"""
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
import heapq
from docopt import docopt
warnings.filterwarnings('ignore')
args = docopt(__doc__)


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
        num_epochs = 5  # @param
        self.epochs = num_epochs
        self.warmup_steps = 0
        self.attribute1 = args['--attribute1']
        self.attribute2 = args['--attribute2']
        self.attribute3 = args['--attribute3']
        self.attribute4 = args['--attribute4']
        lr = 2e-5  # @param
        self.weight = float(args['--weight'])
        self.task_weight = float(args['--task_weight'])
        self.proportion = float(args['--proportion'])
        self.lr = lr  # modified from 3e-5
        run_id = args['--save_path']  # @param
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
            encoded_dict['task_id'] = torch.tensor(self.df.iloc[ix]['task_id'], dtype=torch.long)
        return encoded_dict


class SentimentBertModel(nn.Module):
    def __init__(self, bert_path):
        super().__init__()

        bert_config = BertConfig.from_pretrained(bert_path)
        bert_config.output_hidden_states = True
        self.bert = BertModel.from_pretrained(bert_path, config=bert_config)
        self.dropout = nn.Dropout(0.5)

        self.General_Encoder = nn.Sequential(
            nn.Linear(768, bert_config.hidden_size),
            nn.Tanh()
        )

        self.Specific_Encoder = nn.Sequential(
            nn.Linear(768, bert_config.hidden_size),
            nn.Tanh()
        )

        self.classifier = nn.Linear(bert_config.hidden_size * 2, 2)
        self.task_classifier = nn.Linear(bert_config.hidden_size, 4)
        torch.nn.init.normal_(self.classifier.weight, std=0.02)

        self.gender_at = nn.Linear(bert_config.hidden_size * 2, 2)
        self.age_at = nn.Linear(bert_config.hidden_size * 2, 2)
        self.country_at = nn.Linear(bert_config.hidden_size * 2, 2)
        self.ethnicity_at = nn.Linear(bert_config.hidden_size * 2, 2)

    def forward(self, input_ids, attn_mask, token_type_ids):
        pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids
        )['last_hidden_state']
        all_sentence_v = pooled_output[:, 0, :]
        all_sentence_v = self.dropout(all_sentence_v)

        general_features = self.General_Encoder(all_sentence_v)
        specific_features = self.Specific_Encoder(all_sentence_v)

        task_pred = self.task_classifier(specific_features)
        features = torch.cat([general_features, specific_features], dim=1)

        start_logits = self.classifier(features)

        gender_logits = self.gender_at(features)
        age_logits = self.age_at(features)
        country_logits = self.country_at(features)
        ethnicity_logits = self.ethnicity_at(features)
        return start_logits, all_sentence_v, gender_logits, age_logits, country_logits, ethnicity_logits, task_pred, general_features,specific_features


def test_eval(val_iter, model):
    model.eval()
    logits_list = []
    label_list = []
    for batch in val_iter:
        input_ids, attn_mask, token_type_ids, label = batch['input_ids'], batch['attn_mask'], batch['token_type_ids'], batch['label']

        if torch.cuda.is_available():
            input_ids, attn_mask, token_type_ids, label = input_ids.to(DEVICE), attn_mask.to(DEVICE), token_type_ids.to(DEVICE), label.to(DEVICE)
        with torch.no_grad():
            logits, all_sentence_v, gender_logits, age_logits, country_logits, ethnicity_logits, task_pred, general_features,specific_features = model(input_ids, attn_mask, token_type_ids)

        logits = torch.max(logits.data, 1)[1].cpu()
        label = label.cpu()

        logits_list.extend(logits.tolist())
        label_list.extend(label.tolist())
    f1 = f1_score(logits_list, label_list, average='macro')
    model.train()
    return f1


def train_function(model, train_dl,  pre_task_train_iter, test_iter, attribute, pre_model):
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
    if pre_task_train_iter != None:
        train_steps = ((len(train_dl) + sum([len(pre_task_train_iter[_]) for _ in pre_task_train_iter])) * GCONF.epochs)
    else:
        train_steps = (len(train_dl) * GCONF.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(GCONF.warmup_steps * train_steps),
        num_training_steps=train_steps
    )
    steps = 0
    best_f1 = 0
    model.train()

    for epoch in range(1, GCONF.epochs + 1):
        for batch in train_dl:
            input_ids, attn_mask, token_type_ids, label, gender, age, country, ethnicity, task_id = batch['input_ids'], batch[
                'attn_mask'], batch['token_type_ids'], batch['label'], batch['gender'], batch['age'], batch['country'], batch['ethnicity'], batch['task_id']

            if torch.cuda.is_available():
                input_ids, attn_mask, token_type_ids, label, gender, age, country, ethnicity, task_id = input_ids.to(DEVICE), attn_mask.to(DEVICE), token_type_ids.to(DEVICE), label.to(DEVICE), gender.to(DEVICE), age.to(DEVICE), country.to(DEVICE), ethnicity.to(DEVICE),task_id.to(DEVICE)

            logits, all_sentence_v, gender_logits, age_logits, country_logits, ethnicity_logits, task_pred, general_features,specific_features = model(input_ids, attn_mask, token_type_ids)
            loss = nn.CrossEntropyLoss()
            loss_at = nn.CrossEntropyLoss()
            loss = loss(logits, label)
            if attribute == 'age':
                loss_at = loss_at(age_logits, age)
            if attribute == 'gender':
                loss_at = loss_at(gender_logits,gender)
            if attribute == 'country':
                loss_at = loss_at(country_logits, country)
            if attribute == 'ethnicity':
                loss_at = loss_at(ethnicity_logits, ethnicity)

            if pre_model != None:
                pre_model.eval()
                pre_logits, pre_all_sentence_v, gender_logits, age_logits, country_logits, ethnicity_logits, pre_task_pred, pre_general_features, pre_specific_features = pre_model(input_ids, attn_mask, token_type_ids)
                sp_loss = torch.nn.functional.mse_loss(specific_features, pre_specific_features)
                ge_loss = torch.nn.functional.mse_loss(general_features, pre_general_features)
                task_loss = nn.CrossEntropyLoss()
                task_loss = task_loss(task_pred,task_id)
                double_id_loss = nn.CrossEntropyLoss()
                double_id_loss = double_id_loss(logits, pre_logits)
                loss = loss + GCONF.task_weight * (sp_loss + ge_loss + task_loss + double_id_loss * 0.01)

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

        if pre_task_train_iter != None:
            for pre_task in pre_task_train_iter:
                pre_task_train_dl = pre_task_train_iter[pre_task]
                for batch in pre_task_train_dl:
                    input_ids, attn_mask, token_type_ids, label, gender, age, country, ethnicity, task_id = batch['input_ids'], batch['attn_mask'], batch['token_type_ids'], batch['label'], batch['gender'], batch['age'], batch['country'], batch['ethnicity'], batch['task_id']

                    if torch.cuda.is_available():
                        input_ids, attn_mask, token_type_ids, label, gender, age, country, ethnicity, task_id = input_ids.to(DEVICE), attn_mask.to(DEVICE), token_type_ids.to(DEVICE), label.to(DEVICE), gender.to(DEVICE), age.to(DEVICE), country.to(DEVICE), ethnicity.to(DEVICE), task_id.to(DEVICE)

                    logits, all_sentence_v, gender_logits, age_logits, country_logits, ethnicity_logits, task_pred, general_features,specific_features = model(input_ids,attn_mask,token_type_ids)
                    loss = nn.CrossEntropyLoss()
                    loss_at = nn.CrossEntropyLoss()
                    loss = loss(logits, label)
                    if pre_task == 'age':
                        loss_at = loss_at(age_logits, age)
                    if pre_task == 'gender':
                        loss_at = loss_at(gender_logits, gender)
                    if pre_task == 'country':
                        loss_at = loss_at(country_logits, country)
                    if pre_task == 'ethnicity':
                        loss_at = loss_at(ethnicity_logits, ethnicity)

                    if pre_model != None:
                        pre_model.eval()
                        pre_logits, pre_all_sentence_v, gender_logits, age_logits, country_logits, ethnicity_logits, pre_task_pred, pre_general_features, pre_specific_features = pre_model(
                            input_ids, attn_mask, token_type_ids)
                        sp_loss = torch.nn.functional.mse_loss(specific_features, pre_specific_features)
                        ge_loss = torch.nn.functional.mse_loss(general_features, pre_general_features)
                        task_loss = nn.CrossEntropyLoss()
                        task_loss = task_loss(task_pred, task_id)
                        double_id_loss = nn.CrossEntropyLoss()
                        double_id_loss = double_id_loss(logits, pre_logits)
                        loss = loss + GCONF.task_weight * (sp_loss + ge_loss + task_loss + double_id_loss * 0.01)

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

        dev_f1 = test_eval(test_iter, model)
        print('dev\nf1:%.6f' % (dev_f1))
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            torch.save(model, GCONF.saved_model_path + '/model.pth')
            print('save best model\t\tf1:%.6f' % best_f1)
    return model

GCONF = GlobalConfig()
if not os.path.exists(GCONF.saved_model_path):
    os.mkdir(GCONF.saved_model_path)

seed_everything(GCONF.seed)
all_labels = [0, 1]
tokenizer = BertTokenizerFast.from_pretrained(GCONF.bert_path, do_lower_case=True, add_prefix_space=True,
                                              is_split_into_words=True, truncation=True)

age_train_df = pd.read_csv('dataset/age.tsv', sep='\t')
age_train_df['task_id'] = 0
age_train_ds = TweetBertDataset(age_train_df, tokenizer, GCONF.max_length,is_testing=False)
age_train_dl = DataLoader(age_train_ds, batch_size=GCONF.batch_size, shuffle=True)

country_train_df = pd.read_csv('dataset/country.tsv', sep='\t')
country_train_df['task_id'] = 1
country_train_ds = TweetBertDataset(country_train_df, tokenizer, GCONF.max_length,is_testing=False)
country_train_dl = DataLoader(country_train_ds, batch_size=GCONF.batch_size, shuffle=True)

gender_train_df = pd.read_csv('dataset/gender.tsv', sep='\t')
gender_train_df['task_id'] = 2
gender_train_ds = TweetBertDataset(gender_train_df, tokenizer, GCONF.max_length,is_testing=False)
gender_train_dl = DataLoader(gender_train_ds, batch_size=GCONF.batch_size, shuffle=True)

ethnicity_train_df = pd.read_csv('dataset/ethnicity.tsv', sep='\t')
ethnicity_train_df['task_id'] = 3
ethnicity_train_ds = TweetBertDataset(ethnicity_train_df, tokenizer, GCONF.max_length,is_testing=False)
ethnicity_train_dl = DataLoader(ethnicity_train_ds, batch_size=GCONF.batch_size, shuffle=True)

val_df = pd.read_csv('dataset/valid.tsv', sep='\t', na_values='x', quoting=csv.QUOTE_NONE)
val_ds = TweetBertDataset(val_df, tokenizer, GCONF.max_length, is_testing=True)
val_dl = DataLoader(val_ds, batch_size=GCONF.batch_size, shuffle=True)

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = SentimentBertModel(GCONF.bert_path)
model.to(DEVICE)

dataset = {
    'dl': {
        'age': age_train_dl,
        'country': country_train_dl,
        'gender': gender_train_dl,
        'ethnicity': ethnicity_train_dl
    },
    'df': {
        'age': age_train_df,
        'country': country_train_df,
        'gender': gender_train_df,
        'ethnicity': ethnicity_train_df
    },
}

attribute_list = [GCONF.attribute1,GCONF.attribute2,GCONF.attribute3,GCONF.attribute4]
for attribute_num, attribute in enumerate(attribute_list):
    if attribute_num == 0:
        pre_model = train_function(model, dataset['dl'][attribute], None, val_dl, attribute, None)
    else:
        sample_dataset_dict = {}
        for pred_num in range(attribute_num):
            print(attribute_list[pred_num])
            sample_dataset = dataset['df'][attribute_list[pred_num]].sample(frac=GCONF.proportion)
            sample_dataset_ds = TweetBertDataset(sample_dataset, tokenizer, GCONF.max_length, is_testing=False)
            sample_dataset_dl = DataLoader(sample_dataset_ds, batch_size=GCONF.batch_size, shuffle=True)
            sample_dataset_dict[attribute_list[pred_num]] = sample_dataset_dl

        pre_model = train_function(model, dataset['dl'][attribute], sample_dataset_dict, val_dl, attribute, pre_model)



