import pandas as pd
pd.set_option('display.max_colwidth', None)

df = pd.read_csv('data/tweets_vader.csv', usecols=['Cleaned_Text','Vader_Sentiment_Type'])
df.dropna(inplace=True)
df['Cleaned_Text'] = df['Cleaned_Text'].str.replace('’',"'")

df.rename(columns={"Vader_Sentiment_Type":"sentiment","Cleaned_Text":"text"}, inplace=True)

def redefine_label(label):
    if label == "NEUTRAL":
        return 0
    elif label == "NEGATIVE":
        return 1
    elif label == "POSITIVE":
        return 2

df["label"] = df["sentiment"].apply(redefine_label)
df.drop(columns=["sentiment"], inplace=True)

from sklearn.model_selection import train_test_split
#split the data into train and test set

train,test = train_test_split(df, test_size=0.20, random_state=0)
train,val = train_test_split(train, test_size=0.25, random_state=1)

#save the data
train.to_csv('data/bert_train.csv',index=False)
val.to_csv('data/bert_val.csv',index=False)
test.to_csv('data/bert_test.csv',index=False)

import spacy
from spacymoji import Emoji

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("emoji", first=True)
nlp.pipe_names

from tqdm import tqdm

docs = []
for doc in tqdm(nlp.pipe(df.text), total=len(df)):
    docs.append(doc)

def extract_tokens_plus_meta(doc:spacy.tokens.doc.Doc):
    """Extract tokens and metadata from individual spaCy doc."""
    return [
        (i.text, i.i, i.lemma_, i.ent_type_, i.tag_, 
         i.dep_, i.pos_, i.is_stop, i.is_alpha, 
         i.is_digit, i.is_punct, i._.is_emoji) for i in doc
    ]

def tidy_tokens(docs):
    """Extract tokens and metadata from list of spaCy docs."""
    
    cols = [
        "doc_id", "token", "token_order", "lemma", 
        "ent_type", "tag", "dep", "pos", "is_stop", 
        "is_alpha", "is_digit", "is_punct", "is_emoji"
    ]
    
    meta_df = []
    for ix, doc in tqdm(enumerate(docs), total=len(docs)):
        meta = extract_tokens_plus_meta(doc)
        meta = pd.DataFrame(meta)
        meta.columns = cols[1:]
        meta = meta.assign(doc_id = ix).loc[:, cols]
        meta_df.append(meta)
        
    return pd.concat(meta_df)    

tidy_docs = tidy_tokens(docs)
tidy_docs = tidy_docs[tidy_docs.groupby('doc_id').doc_id.transform(len) > 3]
tidy_docs.query("is_emoji == True")

emoji_list = tidy_docs.query("is_emoji == True").token.unique()
    
emoji_df = pd.DataFrame({'emoji':emoji_list})

emoji_df.to_csv('emoji.csv')

d= {}

# tidy_docs.query("ent_type != ''").ent_type.value_counts()

for e in tidy_docs.query("ent_type != ''").ent_type:
		d[e] = 0

for e in tidy_docs.query("ent_type != ''").ent_type:
	d[e] +=1

print(d)

import matplotlib.pyplot as plt 

tidy_docs.query("ent_type != '' & pos != 'SPACE' & is_stop == False & \
is_punct == False & is_emoji == False & is_digit == False & is_alpha == True").lemma.value_counts().head(30).plot(kind="barh", figsize=(24, 14), alpha=.7)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)

q = tidy_docs.query("ent_type != '' & pos != 'SPACE' & is_stop == False & \
is_punct == False & is_emoji == False & is_digit == False & is_alpha == True").lemma.value_counts().head(30)

r= {}

# tidy_docs.query("ent_type != ''").ent_type.value_counts()

# for e in q:
# 		r[e] = q[e]

# for e in q:
# 	r[e] +=1
for count,e, value in enumerate(q):
    print(count,e ,value)



alphanumeric_list = tidy_docs.query("ent_type != '' & pos != 'SPACE' & is_stop == False & \
is_punct == False & is_emoji == False & is_digit == False").lemma.unique()
alpha_list = tidy_docs.query("ent_type != '' & pos != 'SPACE' & is_stop == False & \
is_punct == False & is_emoji == False & is_digit == False & is_alpha == True").lemma.unique()

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

alphanumeric_list = set(alphanumeric_list) - set(tokenizer.vocab.keys())
alpha_list = set(alpha_list) - set(tokenizer.vocab.keys())

# import pyenchant

# d = pyenchant.Dict("en_US")

# english_words = set()

# for new_token in sorted(alphanumeric_list):
#     if d.check(new_token) == True:
#         english_words.add(new_token)


# vocab_list = sorted(alpha_list-english_words)
# words_to_add = ['2nite','2lgbtqias','bagus','covid-19','fair1','furrie','gta','inb4','ppls','proship','proshpped0','rated18','sama','su1c1de','l3sbians','lgbt2','lgbtabc123xyz','lgbtq2','lgbtqia2','lgbt÷','ww2']
# for word in words_to_add:
#     vocab_list.append(word)
# print(sorted(vocab_list))

# from flask import Flask, send_file, make_response

# app = Flask(__name__)


# # dataset = load_dataset('csv', data_files={'train': base_url+'bert_train.csv','validation': base_url+'bert_val.csv','test': base_url+'bert_test.csv'})

# @app.route('/pp', methods=['GET'])
# def main():
#     return d


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)

import jsons

json_string = jsons.dumps(d)

with open('sample.json', 'w') as outfile:
    outfile.write(json_string)