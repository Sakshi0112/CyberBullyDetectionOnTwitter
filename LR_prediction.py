import nltk
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re
from nltk.stem import WordNetLemmatizer


data = pd.read_csv(r"C:\Users\LENOVO\PycharmProjects\FinalProject\PreprocessedDataset3.csv")
data.dropna(inplace=True)
x = data['text']
y = data['label']


lemmatizer = WordNetLemmatizer()


# lemmatize string
def lemmatize_word(text):
    
    lemmas = [lemmatizer.lemmatize(word, pos='v') for word in text]
    return lemmas



def process_text(text):
    # 1
    stop = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
            'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
            'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
            'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'if', 'because',
            'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
            'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
            'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'only', 'own', 'same', 'so', 'than',
            'too', 'very', 's', 't', 'can', 'will', 'just', 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're',
            've', 'y']

    text = text.lower()
    text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())
    clean_words = [word for word in text.split() if word not in stop and len(word) > 2]
    sentence = " ".join(lemmatize_word(clean_words))
    return sentence


pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2))), 
    ('classifier', LogisticRegression())  
])
pipeline.fit(x, y)


def lg_prediction(sentence):
    preprocessed = process_text(sentence)
    predictions = pipeline.predict([preprocessed])
    return predictions[0]


