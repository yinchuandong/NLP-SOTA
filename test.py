# %%

from collections import Counter
import pandas as pd

import seaborn as sns
import matplotlib.pylab as plt

# %%
df = pd.read_csv('./data_raw/data.tsv', sep='\t', encoding='latin1')

df['sent_token_count'] = df['Sentence'].apply(lambda x: len(x.split(' ')))
df_doc_token_count = df.groupby('DocumentID').sum()['sent_token_count'].reset_index(name='doc_token_count')
df = df.merge(df_doc_token_count, on='DocumentID')
# %%

# we first count the labels, and find the dataset is very imbalanced.
df_label_count = df.groupby(['Label']).size().reset_index(name='count')
sns.barplot(x=df_label_count['Label'], y=df_label_count['count'])

# %%
df_sent_count = df.groupby(['DocumentID']).count()['Label'].reset_index(name='sent_count')
df_sent_count = df_sent_count.sort_values('sent_count', ascending=False).reset_index(drop=True)
df_sent_count.head()
df_sent_count.describe()
# %%


df_ans_count = df.groupby('DocumentID').sum()['Label'].reset_index(name='ans_count')
df_ans_count = df_ans_count.sort_values('ans_count', ascending=False).reset_index(drop=True)
df_ans_count

# %%

df_ans_doc_count = df_ans_count.groupby('ans_count').size().reset_index(name='doc_count')
# %%

df_ans_doc_count
sns.barplot(x=df_ans_doc_count['ans_count'], y=df_ans_doc_count['doc_count'])

# %%

# %%
df['sent_token_count'].describe()
df['sent_token_count'].hist()


df['doc_token_count'].describe()

df['doc_token_count'].hist()
# %%
