import os
import pandas as pd
import nltk
from ast import literal_eval
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer


stop_words = set(stopwords.words('english'))
porter = PorterStemmer()
keyword_dict = pd.DataFrame(columns=['Keyword', 'Docs'])

pos_map = {
    "NN": wordnet.NOUN, "NNS": wordnet.NOUN, "NNP": wordnet.NOUN, "NNPS": wordnet.NOUN,
    "VB": wordnet.VERB, "VBD": wordnet.VERB, "VBG": wordnet.VERB, "VBN": wordnet.VERB, "VBP": wordnet.VERB, "VBZ": wordnet.VERB,
    "JJ": wordnet.ADJ, "JJR": wordnet.ADJ, "JJS": wordnet.ADJ,
    "RB": wordnet.ADV, "RBR": wordnet.ADV, "RBS": wordnet.ADV
}


def preprocess_line(line: str) -> tuple[list, list]:
    tokens = nltk.word_tokenize(line)
    pos = nltk.pos_tag(tokens)
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.lower() in stop_words:
            tokens.pop(i)
            pos.pop(i)
        else:
            stemmed = porter.stem(token)
            tokens[i] = stemmed
            temp = list(pos[i])
            temp[0] = stemmed
            pos[i] = tuple(temp)
            i += 1
    return tokens, pos


def process_doc(doc: pd.DataFrame, fileName: str) -> None:
    doc = doc.applymap(lambda x: preprocess_line(x)[0])
    keywords = doc['Line'].explode().unique()
    for keyword in keywords:
        if keyword not in keyword_dict['Keyword'].values:
            keyword_dict.loc[len(keyword_dict.index)] = [keyword, [fileName]]
        else:
            keyword_dict.loc[keyword_dict['Keyword'] == keyword, 'Docs'].values[0].append(fileName)
    return


def preprocess_info() -> None:
    for (dirPath, dirNames, fileNames) in os.walk('KnowledgeDatabase/'):
        for fileName in fileNames:
            print(fileName)
            doc = pd.read_csv(dirPath + "/" + fileName, sep="`", names=['Line'], encoding='utf-8')
            process_doc(doc, fileName)
    return


def preprocess_questions(questions: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    totals = questions.iloc[0]
    questions = questions.drop(0).reset_index(drop=True)

    q_processed = questions.applymap(lambda x: preprocess_line(x), na_action='ignore')
    q_pos = q_processed.applymap(lambda x: x[1], na_action='ignore')
    q_processed = q_processed.applymap(lambda x: x[0], na_action='ignore')
    
    return totals, q_pos, q_processed


def rank_docs(keywords: list) -> list:
    # Get all docs where all question keywords appear
    keywords.remove("?") 
    intersect_docs = None
    for keyword in keywords:
        if intersect_docs == None:
            intersect_docs = keyword_dict.loc[keyword_dict['Keyword'] == keyword, 'Docs'].values[0]
        else:
            keyword_docs = keyword_dict.loc[keyword_dict['Keyword'] == keyword, 'Docs'].values[0]
            intersect_docs = list(set(intersect_docs) & set(keyword_docs))
    
    return intersect_docs
    
    # Rerank docs based on keyword frequency
    # Not necessary with modified algorithm
    reranked = {}
    for fileName in intersect_docs:
        doc = pd.read_csv("KnowledgeDatabase/" + fileName, sep="`", names=['Line'], encoding='utf-8')
        doc = doc.applymap(lambda x: preprocess_line(x)[0])
        doc = doc.applymap(lambda x: [w for w in x if w != "?"])
        doc_keywords = doc['Line'].explode()
        matching = doc_keywords[doc_keywords.isin(keywords)]
        reranked[fileName] = matching.shape[0] / doc_keywords.shape[0]
    reranked = dict(sorted(reranked.items(), key=lambda item: item[1], reverse=True))
    return list(reranked.keys())


def score_line(doc_pos: list[tuple], ques_pos: list[tuple]) -> float:
    most_matches = 0
    matches = 0
    synonyms = {}

    for (q_keyword, q_pos) in ques_pos:
        synonyms[q_keyword] = []
        if q_pos in pos_map.keys():
            syns = wordnet.synsets(q_keyword, pos=pos_map[q_pos])
            for syn in syns:
                synonyms[q_keyword].append(syn.lemmas()[0].name())
                # for lemma in syn.lemmas():
                #     synonyms[q_keyword].append(lemma.name())

    for (d_keyword, d_pos) in doc_pos:
        if d_pos == '.' and matches > most_matches:
            most_matches = matches
            matches = 0
        for (q_keyword, q_pos) in ques_pos:
            if (d_keyword == q_keyword or d_keyword in synonyms[q_keyword]) and d_pos == q_pos:
                matches += 1
                break
    return most_matches / len(ques_pos)


def extract_answer(ranked_docs: list, pos: list[tuple]) -> str:
    best_answer = ""
    best_score = 0
    for fileName in ranked_docs:
        doc = pd.read_csv("KnowledgeDatabase/" + fileName, sep="`", names=['Line'], encoding='utf-8')
        doc_pos = doc.applymap(lambda x: preprocess_line(x)[1])
        doc_scores = doc_pos.applymap(lambda x: score_line(x, pos))
        score = doc_scores['Line'].max()
        if score > best_score:
            best_score = score
            best_answer = doc[doc_scores['Line'] == score]['Line'].values[0]
    return best_answer


def answer_question(keywords, pos) -> str:
    # TODO: Take string as input and preprocess it (stemming & POS)
    ranked_docs = rank_docs(keywords)
    return extract_answer(ranked_docs, pos)


def main():
    global keyword_dict
    # keyword_dict = preprocess_info()
    # keyword_dict.to_csv('keyword_dictionary.csv', index=False)
    keyword_dict = pd.read_csv('keyword_dictionary.csv', converters={'Docs': literal_eval})
    # print(keyword_dict)

    questions = pd.read_csv('QuestionsCorpus/AllQuestions.csv', sep=';')
    totals, q_pos, q_processed = preprocess_questions(questions)
    # print(totals)
    # print(q_pos)
    # print(q_processed)

    test_row = 0
    test_col = 'Recipe'
    keywords = q_processed.loc[test_row, test_col]
    pos = q_pos.loc[test_row, test_col]
    print("Q:", questions.loc[test_row+1, test_col])
    print("A:", answer_question(keywords, pos))


if __name__ == "__main__":
    main()