import random
import tensorflow as tf
import numpy as np

tf.keras.utils.set_random_seed(17)  # sets seeds for base-python, numpy and tf

tf.config.list_physical_devices('GPU')

import pandas as pd
df = pd.read_csv("/Users/vinhkhaitruong/Documents/CE4045/npm_project/src/NLP_preprocessing/data/tweets.csv")

df[df.duplicated()]

df.rename(columns={"Vader_Sentiment_Type":"sentiment","Text":"text"}, inplace=True)
df = df[['text','sentiment']]

import re
def get_emoji_regexp():
    # Sort emoji by length to make sure multi-character emojis are
    # matched first
    emojis = sorted(emoji.EMOJI_DATA, key=len, reverse=True)
    pattern = u'(' + u'|'.join(re.escape(u) for u in emojis) + u')'
    return re.compile(pattern)

    import re
import string
import emoji

##CUSTOM DEFINED FUNCTIONS TO CLEAN THE TWEETS

#Clean emojis from text
def strip_emoji(text):
    return re.sub(get_emoji_regexp(), r"", text) #remove emoji

#Remove punctuations, links, mentions and \r\n new line characters
def strip_all_entities(text): 
    text = text.replace('\r', '').replace('\n', ' ').replace('\n', ' ').lower() #remove \n and \r and lowercase
    text = re.sub(r"(?:\@|https?\://)\S+", "", text) #remove links and mentions
    text = re.sub(r'[^\x00-\x7f]',r'', text) #remove non utf8/ascii characters such as '\x9a\x91\x97\x9a\x97'
    banned_list= string.punctuation + 'Ã'+'±'+'ã'+'¼'+'â'+'»'+'§'
    table = str.maketrans('', '', banned_list)
    text = text.translate(table)
    return text

#clean hashtags at the end of the sentence, and keep those in the middle of the sentence by removing just the # symbol
def clean_hashtags(tweet):
    new_tweet = " ".join(word.strip() for word in re.split('#(?!(?:hashtag)\b)[\w-]+(?=(?:\s+#[\w-]+)*\s*$)', tweet)) #remove last hashtags
    new_tweet2 = " ".join(word.strip() for word in re.split('#|_', new_tweet)) #remove hashtags symbol from words in the middle of the sentence
    return new_tweet2

#Filter special characters such as & and $ present in some words
def filter_chars(a):
    sent = []
    for word in a.split(' '):
        if ('$' in word) | ('&' in word):
            sent.append('')
        else:
            sent.append(word)
    return ' '.join(sent)

def remove_mult_spaces(text): # remove multiple spaces
    return re.sub("\s\s+" , " ", text)

def clean_text(text,stop_words): 
    delete_dict = {sp_character: '' for sp_character in string.punctuation} 
    delete_dict[' '] = ' ' 
    table = str.maketrans(delete_dict)
    text1 = text.translate(table)
    #print('cleaned:'+text1)
    textArr= text1.split()
    text_new = []
    for token in textArr:
        if token in stop_words:
            a = 1
        else:
            text_new.append(token)
    text2 = ' '.join([w for w in text_new if ( not w.isdigit() and  ( not w.isdigit() and len(w)>2))]) 
    
    return text2.lower()

    manual_stop_words = ['the',
 'to',
 'and',
 'is',
 'of',
 'i',
 'it',
 'in',
 'you',
 'are',
 'that',
 '’',
 'for',
 'people',
 'they',
 'this',
 'with',
 'be',
 'on',
 'have',
 's',
 'about',
 'so',
 'we',
 'or',
 'if',
 'who',
 't',
 'was',
 'an',
 'from',
 'can',
 'their',
 'my',
 'what',
 'at',
 'by',
 'your',
 'am',
 'being',
 'no',
 'them',
 'will',
 'me',
 'he',
 'more',
 'because',
 'has',
 'out',
 'would',
 'there',
 'when',
 'how',
 'why',
 'get',
 'think',
 'one',
 '...',
 'up',
 'want',
 '️',
 'know',
 'our',
 'she',
 'also',
 'us',
 'some',
 'other',
 'been',
 'see',
 'women',
 'her',
 'say',
 'does',
 'any',
 'make',
 'then',
 'were',
 'going',
 'its',
 're',
 '…',
 'way',
 'time',
 'did',
 'these',
 '2',
 'children',
 'where',
 'm',
 'those',
 'part',
 'said',
 'need',
 'here',
 'go',
 'which',
 'his',
 'into',
 'something',
 'men',
 '3',
 'person',
 'had',
 'thing',
 '"',
 'things',
 '.',
 'group',
 'got',
 'first',
 'could',
 'someone',
 'characters',
 'saying',
 "that's",
 'made',
 'show',
 'u',
 'trying',
 '..',
 'take',
 'school',
 'man',
 ',',
 'let',
 'after',
 'ppl',
 'etc',
 '2022',
 'life',
 'live',
 'today',
 'world',
 'him',
 'day',
 'years',
 'book',
 'feel',
 'use',
 'look',
 'stuff',
 'media',
 'under',
 'while',
 'point',
 'read',
 'again',
 'country',
 'since',
 'getting',
 "there's",
 'doing',
 'such',
 'come',
 'making',
 've',
 'friends',
 'having',
 'without',
 'own',
 'stonewall',
 'members',
 'work',
 'makes',
 'better',
 '‘',
 'state',
 'before',
 'tell',
 'family',
 'twitter',
 'around',
 'woman',
 'fact',
 'give',
 'find',
 'groups',
 'reason',
 'character',
 'using',
 'keep',
 'folks',
 'via',
 'both',
 'call',
 'two',
 'used',
 'states',
 'means',
 'put',
 'schools',
 '5']

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stops = stopwords.words('english')

texts_new = []
for t in df.text:
    texts_new.append(clean_text(remove_mult_spaces(filter_chars(clean_hashtags(strip_all_entities(strip_emoji(t))))),manual_stop_words))
df['text_clean'] = texts_new

import transformers
from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

token_lens = []

for i,txt in enumerate(df['text_clean'].values):
    tokens = tokenizer.encode(txt, max_length=512, truncation=True)
    token_lens.append(len(tokens))
    if len(tokens)>80:
        print(f"INDEX: {i}, TEXT: {txt}") 