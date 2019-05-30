import pandas as pd
from tqdm import tqdm
import re as reg

# Permet de fusionner le path des images avec une list de classes en un seul fichier
'''
df=pd.read_csv('./HARRISON/data_list.txt')
df2=pd.read_csv('./HARRISON/tag_list.txt')
df3 = pd.concat( [df, df2], axis=1)
df3.columns = ["path", "labels"]
for index, row in tqdm(df3.iterrows(), total=df3.shape[0]):
    temp = row['labels'].replace(" ", ",")
    temp = temp[:-1]
    df3.at[index, 'labels'] = temp

df3.to_csv('./HARRISON/dataTest.txt', header=["path", "labels"], index=None, sep=',', mode='w')
'''

# Permet de lister l'ensemble des classes depuis le fichier d'origine de HARISSOn
# Phase de netoyage obligatoire pour avoir un format de dataset lisible
colnames=['classe']
df=pd.read_csv('./HARRISON/vocab_index.txt', names=colnames, header=None)
pattern=reg.compile(r"(.)\1{1,}",reg.DOTALL)

for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    temp = row['classe'].replace(" ", ",")
    print(pattern.sub(r"\1",temp))
    df.at[index, 'classe'] = pattern.sub(r"\1",temp)

df.to_csv('./HARRISON/listClass.txt', header=["classe"], index=None, sep=',', mode='w')


