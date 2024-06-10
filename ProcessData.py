import csv
import pandas
from sklearn.model_selection import StratifiedKFold

datas = pandas.read_csv('data/split/English/train.tsv',sep='\t',na_values='x',quoting=csv.QUOTE_NONE)[['tid','uid','text','date','gender','age','country','ethnicity','label']]
headers = datas.columns
print(len(datas))
datas = datas.dropna().values.tolist()
print(len(datas))
print(datas[:3])
labels = []
for data in datas:
    label = data[4] * 10000 + data[5] * 1000 + data[6] * 100 + data[7] * 10 + data[8]
    labels.append(label)
print(labels)

kfolder = StratifiedKFold(n_splits=4,random_state=2023,shuffle=True)

for fold, (train_index, test_index) in enumerate(kfolder.split(datas, labels)):
    print('TRAIN:', train_index, "TEST:", test_index)
    X_test = [datas[_] for _ in test_index]
    X_test = pandas.DataFrame(X_test)
    X_test.to_csv('dataset/' + str(fold)+'.tsv',sep='\t',header=headers,index=None)
