# %%

from collections import Counter
import pandas as pd


# %%
df = pd.read_csv('./data/data.tsv', sep='\t', encoding='latin1')

# %%

Counter(df['Label'])

df_s = df.groupby(['DocumentID']).count()['Label'].reset_index(name='sentences')
df_s = df_s.sort_values('sentences', ascending=False)
df_s.describe()
# %%


df_a = df.groupby('DocumentID').sum()['Label'].reset_index(name='answers')

df_a = df_a.sort_values('answers', ascending=False)
df_a.head()
# %%
df_a[df_a['answers'] == 1].count()
df_a[df_a['answers'] > 1].count()
=
df_a[df_a['answers'] < 1].count()
# %%
df['DocumentID'].unique().shape

# %%
