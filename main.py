import os
import math
import string
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


stop_words = set(stopwords.words('english'))
stop_words = stop_words.union(set(string.punctuation))
porter = PorterStemmer()
tf_idf = pd.DataFrame(dtype='float64')


def tokenize_line(line) -> list:
    tokens = nltk.word_tokenize(line)
    stemmed = []
    for token in tokens:
        if token.lower() not in stop_words:
            stemmed.append(porter.stem(token))
    return stemmed


def preprocess_kd():
    global tf_idf
    tf = pd.DataFrame(dtype='float64')
    idf = pd.DataFrame(dtype='float64')

    for (dirPath, dirNames, fileNames) in os.walk('KnowledgeDatabase/'):
        for fileName in fileNames:
            filePath = dirPath + "/" + fileName
            print(filePath[18:])
            doc = pd.read_csv(filePath, sep='`', names=['Line'], encoding='utf-8')
            doc = doc.applymap(lambda x: tokenize_line(x))
            all_tokens = doc['Line'].explode()
            total = all_tokens.shape[0]
            unique_tokens = all_tokens.unique()
            for token in unique_tokens:
                count = all_tokens[all_tokens == token].shape[0]
                tf.loc[token, filePath[18:]] = count / total
    tf.fillna(0, inplace=True)
    N = tf.shape[1]
    idf = tf.apply(lambda row: math.log(N / (row[row > 0].shape[0]+1)), axis=1)
    tf_idf = tf.mul(idf, axis=0)
    print(tf_idf.describe())


def preprocess_questions(questions: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    totals = questions.iloc[0]
    questions = questions.drop(0).reset_index(drop=True)
    
    q_processed = questions.applymap(lambda x: tokenize_line(x), na_action='ignore')
    
    return totals, q_processed


def main():
    preprocess_kd()
    
    # questions = pd.read_csv('QuestionsCorpus/AllQuestions.csv', sep=';')

    # totals, q_processed = preprocess_questions(questions)
    # print(totals)
    # print(q_processed)


if __name__ == "__main__":
    main()