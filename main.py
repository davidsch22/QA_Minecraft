import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


stop_words = set(stopwords.words('english'))
porter = PorterStemmer()


def process_file(file: pd.DataFrame, keyword_dict: pd.DataFrame):
    file = file.applymap(lambda x: nltk.word_tokenize(x))
    file = file.applymap(lambda x: [w for w in x if not w.lower() in stop_words])
    file = file.applymap(lambda x: [porter.stem(w) for w in x])
    file = file.applymap(lambda x: list(set(x)))
    return file


def preprocess_info() -> pd.DataFrame:
    keyword_dict = pd.DataFrame(columns=['Keyword', 'Docs'])

    for (dirPath, dirNames, fileNames) in os.walk('KnowledgeDatabase/'):
        for fileName in fileNames:
            file = pd.read_csv(dirPath + "/" + fileName, sep="~", names=['Line'], encoding='utf-8')
            keyword_dict = process_file(file, keyword_dict)
            print(keyword_dict)
            return file
    
    return keyword_dict


def preprocess_questions(questions: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    totals = questions.iloc[0]
    questions = questions.drop(0).reset_index(drop=True)
    
    q_tokens = questions.applymap(lambda x: nltk.word_tokenize(x), na_action='ignore')
    q_pos = q_tokens.applymap(lambda x: nltk.pos_tag(x), na_action='ignore')
    q_processed = q_tokens.applymap(lambda x: [w for w in x if not w.lower() in stop_words], na_action='ignore')
    q_processed = q_processed.applymap(lambda x: [porter.stem(w) for w in x], na_action='ignore')
    
    return totals, q_pos, q_processed


def main():
    keyword_dict = preprocess_info()

    # questions = pd.read_csv('QuestionsCorpus/AllQuestions.csv', sep=';')
    # totals, q_pos, q_processed = preprocess_questions(questions)
    # print(totals)
    # print(q_pos)
    # print(q_processed)


if __name__ == "__main__":
    main()