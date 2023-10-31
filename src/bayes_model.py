import pandas as pd
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import matplotlib 
from matplotlib import pyplot
import nltk
import re
import string
from nltk.corpus import stopwords
# nltk.download('punkt')
# nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# stop_words = stopwords.words()
matplotlib.pyplot.switch_backend('Agg') 

def cleaning(text):        
    # converting to lowercase, removing URL links, special characters, punctuations...
    text = text.lower() # converting to lowercase
    text = re.sub('https?://\S+|www\.\S+', '', text) # removing URL links
    text = re.sub(r"\b\d+\b", "", text) # removing number 
    text = re.sub('<.*?>+', '', text) # removing special characters, 
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text) # punctuations
    text = re.sub('\n', '', text)
    text = re.sub('[’“”…]', '', text)
   
    #removing emoji: 
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)   

   # removing short form: 
    
    text=re.sub("isn't",'is not',text)
    text=re.sub("he's",'he is',text)
    text=re.sub("wasn't",'was not',text)
    text=re.sub("there's",'there is',text)
    text=re.sub("couldn't",'could not',text)
    text=re.sub("won't",'will not',text)
    text=re.sub("they're",'they are',text)
    text=re.sub("she's",'she is',text)
    text=re.sub("There's",'there is',text)
    text=re.sub("wouldn't",'would not',text)
    text=re.sub("haven't",'have not',text)
    text=re.sub("That's",'That is',text)
    text=re.sub("you've",'you have',text)
    text=re.sub("He's",'He is',text)
    text=re.sub("what's",'what is',text)
    text=re.sub("weren't",'were not',text)
    text=re.sub("we're",'we are',text)
    text=re.sub("hasn't",'has not',text)
    text=re.sub("you'd",'you would',text)
    text=re.sub("shouldn't",'should not',text)
    text=re.sub("let's",'let us',text)
    text=re.sub("they've",'they have',text)
    text=re.sub("You'll",'You will',text)
    text=re.sub("i'm",'i am',text)
    text=re.sub("we've",'we have',text)
    text=re.sub("it's",'it is',text)
    text=re.sub("don't",'do not',text)
    text=re.sub("that´s",'that is',text)
    text=re.sub("I´m",'I am',text)
    text=re.sub("it’s",'it is',text)
    text=re.sub("she´s",'she is',text)
    text=re.sub("he’s'",'he is',text)
    text=re.sub('I’m','I am',text)
    text=re.sub('I’d','I did',text)
    text=re.sub("he’s'",'he is',text)
    text=re.sub('there’s','there is',text)
    
     
    return text

def bayes(sentence):
	stop_words = stopwords.words()
	
	df = pd.read_csv('./text-query-tweets-sentiment-analysis-textblob.csv')
	df.head()
   
	# from io import StringIO
	col = ['Cleaned_Text', 'Blob_Sentiment_Type']
	df = df[col]
	df = df[pd.notnull(df['Cleaned_Text'])]
	df.columns = ['Cleaned_Text', 'Blob_Sentiment_Type']
	df['category_id'] = df['Blob_Sentiment_Type'].factorize()[0]
	category_id_df = df[['Blob_Sentiment_Type', 'category_id']].drop_duplicates().sort_values('category_id')
	category_to_id = dict(category_id_df.values)
	id_to_category = dict(category_id_df[['category_id', 'Blob_Sentiment_Type']].values)
	df.head()

	# Show count of each class
	# import matplotlib.pyplot as plt
	# fig = plt.figure(figsize=(8,6))
	# df.groupby('Blob_Sentiment_Type').Cleaned_Text.count().plot.bar(ylim=0)
	# plt.show()

	# Check actual count of each class
	positive_count, neutral_count, negative_count = df['Blob_Sentiment_Type'].value_counts()

	# Separate class
	positive_class = df[df['Blob_Sentiment_Type'] == "POSITIVE"]
	neutral_class = df[df['Blob_Sentiment_Type'] == "NEUTRAL"] # print the shape of the class
	negative_class = df[df['Blob_Sentiment_Type'] == "NEGATIVE"]
	print('class 0:', positive_count)
	print('class 1:', neutral_count)
	print('class 2:', negative_count)

	# Balance class (i used underfitting here)
	new_positive_count = positive_class.sample(negative_count)
	new_neutral_count = neutral_class.sample(negative_count)

	df = pd.concat([new_positive_count,new_neutral_count,negative_class], axis=0)

	print("Total count for each class: \n",df['Blob_Sentiment_Type'].value_counts())# plot the count after under-sampeling
	df['Blob_Sentiment_Type'].value_counts().plot(kind='bar', title='count (target)')

	# Clean data and set features
	# from sklearn.feature_extraction.text import TfidfVectorizer
	tfidf = TfidfVectorizer(sublinear_tf=True, min_df=2, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
	features = tfidf.fit_transform(df.Cleaned_Text).toarray()
	labels = df.category_id                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
	# features.shape

	# Find the terms that are the most correlated with each class
	# from sklearn.feature_selection import chi2
	# import numpy as np
	# Number that can be adjusted to show top words in each category
	N = 2
	for Blob_Sentiment_Type, category_id in sorted(category_to_id.items()):
		features_chi2 = chi2(features, labels == category_id)
		indices = np.argsort(features_chi2[0])
		feature_names = np.array(tfidf.get_feature_names_out())[indices]
		unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
		bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
	print("# '{}':".format(Blob_Sentiment_Type))
	print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
	print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))

	# Using MultinomialNB
	# from sklearn.model_selection import train_test_split
	# from sklearn.feature_extraction.text import CountVectorizer
	# from sklearn.feature_extraction.text import TfidfTransformer
	# from sklearn.naive_bayes import MultinomialNB

	X_train, X_test, y_train, y_test = train_test_split(df['Cleaned_Text'], df['Blob_Sentiment_Type'], random_state = 0)
	count_vect = CountVectorizer()
	X_train_counts = count_vect.fit_transform(X_train)
	tfidf_transformer = TfidfTransformer()
	X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

	clf = MultinomialNB().fit(X_train_tfidf, y_train)
	sentence = cleaning(sentence)
	# a = []
	# a = ( word for word in sentence.split() if word not in (stop_words) )
	# Testing predictions
	result = clf.predict(count_vect.transform([sentence]))
	print(sentence.split(" "))
	# print(a)

	return [result,sentence.split(" ")] 



from flask import Flask, send_file, make_response
from flask import Flask, request, render_template
app = Flask(__name__)


# dataset = load_dataset('csv', data_files={'train': base_url+'bert_train.csv','validation': base_url+'bert_val.csv','test': base_url+'bert_test.csv'})

@app.route('/test', methods = ['POST'])
def main():
   inp = request.get_json()['input']
   print(inp)
   store = bayes(inp) 
   return {"result":store[0][0],"tokenize":store[1]}


if __name__ == '__main__':
    app.run(host="localhost",port=5000)


# Try and compare different models
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import LinearSVC
# from sklearn.model_selection import cross_val_score
# models = [
#     #RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0,),
#     # LinearSVC(),
#     MultinomialNB(),
#     #LogisticRegression(random_state=0),
# ]
# CV = 5
# cv_df = pd.DataFrame(index=range(CV * len(models)))
# entries = []
# for model in models:
#   model_name = model.__class__.__name__
#   accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
#   for fold_idx, accuracy in enumerate(accuracies):
#     entries.append((model_name, fold_idx, accuracy))
# cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
# import seaborn as sns
# sns.boxplot(x='model_name', y='accuracy', data=cv_df)
# sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
#               size=8, jitter=True, edgecolor="gray", linewidth=2)
# plt.show()
