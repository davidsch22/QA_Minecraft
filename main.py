import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def preprocess_info(knowledge: pd.DataFrame):
    pass


def preprocess_questions(questions: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    totals = questions.iloc[0]
    questions = questions.drop(0).reset_index(drop=True)
    
    q_tokens = questions.applymap(lambda x: nltk.word_tokenize(x), na_action='ignore')

    q_pos = q_tokens.applymap(lambda x: nltk.pos_tag(x), na_action='ignore')

    stop_words = set(stopwords.words('english'))
    q_processed = q_tokens.applymap(lambda x: [w for w in x if not w.lower() in stop_words], na_action='ignore')

    porter = PorterStemmer()
    q_processed = q_processed.applymap(lambda x: [porter.stem(w) for w in x], na_action='ignore')
    
    return totals, q_pos, q_processed


def main():
    questions = pd.read_csv('QuestionsCorpus/AllQuestions.csv', sep=';')

    totals, q_pos, q_processed = preprocess_questions(questions)
    print(totals)
    print(q_pos)
    print(q_processed)


if __name__ == "__main__":
    main()