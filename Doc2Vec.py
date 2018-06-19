# SENTIMENT ANALYSIS OF AMAZON FOOD REVIEWS USING DOC2VEC MODEL
# BY: OMKAR VIVEK SABNIS - 17/06/2018

# IMPORTING ALL REQUIRED MODULES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import re
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier as RFC
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pickle

np.random.seed(0)

# FUNCTION TO READ THE FILE
def read_text_file(f):
    df_complete = pd.read_csv(f)
    df = df_complete.loc[:,["Text","Score"]]
    df.dropna(how="any", inplace=True)    
    return df
df = read_text_file("../input/Reviews.csv")
print (df.head())

# VISUALIZING THE DATASET AND ITS SHAPE
def sampling_dataset(df):
    count = 5000
    class_df_sampled = pd.DataFrame(columns = ["Score","Text"])
    temp = []
    for c in df.Score.unique():
        class_indexes = df[df.Score == c].index
        random_indexes = np.random.choice(class_indexes, count, replace=False)
        temp.append(df.loc[random_indexes])
        for each_df in temp:
            class_df_sampled = pd.concat([class_df_sampled,each_df],axis=0)
    return class_df_sampled
df = sampling_dataset(df)
df.reset_index(drop=True,inplace=True)
print (df.head())
print (df.shape)

lmtzr = WordNetLemmatizer()
w = re.compile("\w+",re.I)

def label_sentences(df):
    labeled_sentences = []
    for index, datapoint in df.iterrows():
        tokenized_words = re.findall(w,datapoint["Text"].lower())
        labeled_sentences.append(LabeledSentence(words=tokenized_words, tags=['SENT_%s' %index]))
    return labeled_sentences

def train_doc2vec_model(labeled_sentences):
    model = Doc2Vec(alpha=0.025, min_alpha=0.025)
    model.build_vocab(labeled_sentences)
    for epoch in range(10):
        model.train(labeled_sentences)
        model.alpha -= 0.002 
        model.min_alpha = model.alpha
    return model

sen = label_sentences(df)
model = train_doc2vec_model(sen)

def vectorize_comments(df,d2v_model):
    y = []
    comments = []
    for i in range(0,df.shape[0]):
        label = 'SENT_%s' %i
        comments.append(d2v_model.docvecs[label])
    df['vectorized_comments'] = comments
    return df
df = vectorize_comments(df,model)
print (df.head(2))

def train_classifier(X,y):
    n_estimators = [200,400]
    min_samples_split = [2]
    min_samples_leaf = [1]
    bootstrap = [True]

    parameters = {'n_estimators': n_estimators, 'min_samples_leaf': min_samples_leaf,
                  'min_samples_split': min_samples_split}

    clf = GridSearchCV(RFC(verbose=1,n_jobs=4), cv=4, param_grid=parameters)
    clf.fit(X, y)
    return clf

X_train, X_test, y_train, y_test = cross_validation.train_test_split(df["vectorized_comments"].T.tolist(), df["Score"], test_size=0.02, random_state=17)
classifier = train_classifier(X_train,y_train)
print (classifier.best_score_, "----------------Best Accuracy score on Cross Validation Sets")
print (classifier.score(X_test,y_test))

f = open("Output.txt","w")
f.write("Best Accuracy score on Cross Validation Sets %f" %classifier.best_score_,)
f.write("Score on Test Set %f" %classifier.score(X_test,y_test))
f.close()
