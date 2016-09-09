from nltk import word_tokenize
from collections import Counter
import random
import os
import nltk
import re
import string

REMOVE_IRRELEVANT_TEXT = 1
ADD_SENTENCE_BUUNDARY_TAG = 0
DIFFERNTIATE_CAPS = 0
REMOVE_BAD_SYMBOLS = 1

Bad_symbols = ":;'\"#$%&()*+-/<=>@[\]^_`{|}~<>\|\\"
Ending_symbols = ".!?"
Classification = "data/data_corrected/classification_task/"
Spelling = "data/data_corrected/spell_checking_task/"
Types_of_file = {"atheism", "autos", "graphics", "medicine", "motorcycles", "religion", "space"}
File_counts = 300 #0-299 0 might be invalid



def format_file_name(task_type, file_type,file_number,train_docs="train_docs"):
    if "cl" == task_type:
        return Classification + file_type + "/" + train_docs + "/" + file_type + "_file{}.txt".format(file_number)
    elif "sp" == task_type:
        if "modified" not in train_docs:
            return Spelling+ file_type + "/" + train_docs + "/" + file_type + "_file{}.txt".format(file_number)
        else:
            return Spelling+ file_type + "/" + train_docs + "/" + file_type + "_file{}_modified.txt".format(file_number)
    else:
        return None

def read_file(task_type: str, file_type: str, file_number: int, train_docs="train_docs"):
    file_name = format_file_name(task_type, file_type, file_number, train_docs)
    if os.path.exists(file_name):
        with open(file_name) as f:
            file_content = f.read()
            return file_content
    return ""

def preprocess_content(content: str):
    if REMOVE_IRRELEVANT_TEXT:
        email_pattern = '\w+@\w+\.\w+'
        content = re.sub(email_pattern, ' ', content)
    if REMOVE_BAD_SYMBOLS:
        regex = re.compile('[%s]' % re.escape(Bad_symbols))
        content = regex.sub(' ', content)
    return content
    
def tokenize(file_content: str):
    #return list of words given a str
    return word_tokenize(file_content)
       
def bow(tokens: [str]):
    #return a dict of word tokens associated with counts
    d = dict()
    for i in tokens:
        if i not in d:
            d[i] = 1
        else:
            d[i] += 1
    return d

def handle_file(task_type: str, file_type: str, file_number: int, train_docs="train_docs"):
    #return bag of word representation of a given file
    return bow(tokenize(preprocess_content(read_file(task_type, file_type, file_number, train_docs))))

def tokenize_file(task_type: str, file_type: str, file_number: int, train_docs="train_docs"):
    #return a list of pre-processed tokens given a file
    return tokenize(preprocess_content(read_file(task_type, file_type, file_number, train_docs)))

def build_unary_model(c: dict):
    #given a BoW, convert count into probability
    total = sum(c.values())
    d = dict(c)
    for i in d:
        d[i] = d[i] / total
    return d

def assign_probability_unary(c: dict)->[tuple]:
    #given a unary model, assign a lower and upper bound probability to the word. e.g. [0.23,0.34, 'word']
    lower_bound = 0
    ret = []
    for i in c:
        ret.append((lower_bound, lower_bound+c[i], i))
        lower_bound += c[i]
    return ret

def unary_random_word_generation(probability: [tuple]):
    #return a random word based on probability given probability list
    low, high = 0, len(probability) - 1
    random_int = random.random()
    
    while random_int >= probability[-1][1]:
        random_int = random.random() #normalize
        
    while (low <= high):
        mid = (low + high) // 2
        if probability[mid][0] > random_int:
            high = mid - 1
        elif probability[mid][1] <= random_int:
            low = mid + 1
        else:
            return probability[mid][2]

def unary_random_sentence_generation(task_type, file_type, train_docs="train_docs"):
    C = dict()
    for i in range(300):
        d2 = handle_file(task_type, file_type, i, train_docs)
        for i in d2:
            if i in C:
                C[i] += d2[i]
            else:
                C[i] = d2[i]
    model = build_unary_model(C)
    probability = assign_probability_unary(model)
    sentence = ""
    while 1:
        word = unary_random_word_generation(probability)
        if word in Ending_symbols:
            sentence += word + " "
            return sentence
        sentence += word + " "

def build_bigram_model(tokens: [str]):
    #turn list of tokens into bigrams dict of dict
    bigrams = list(nltk.bigrams(tokens))
    d = dict()
    for i, j in bigrams:
        if i not in d:
            d[i] = {j: 1}
        elif j not in d[i]:
            d[i][j] = 1
        else:
            d[i][j] += 1
    return d

def bigram_update_model_with_new_tokens(d:"bigram_model", tokens):
    #update current bigram model with new tokens
    new = list(nltk.bigrams(tokens))
    for i, j in new:
        if i not in d:
            d[i] = {j: 1}
        elif j not in d[i]:
            d[i][j] = 1
        else:
            d[i][j] += 1
    return d

def bigram_random_sentence_generation(task_type, file_type, train_docs="train_docs"):

    #build unary and generate start word
    C = dict() #unary dict
    d = dict() #d is bigram_model stored as dict of dict
    for i in range(300):
        #bow(tokenize(preprocess_content(read_file(task_type, file_type, file_number, train_docs))))
        _tokens = tokenize_file(task_type, file_type, i, train_docs)
        d2 = bow(_tokens)
        for i in d2:
            if i in C:
                C[i] += d2[i]
            else:
                C[i] = d2[i]
        d = bigram_update_model_with_new_tokens(d, _tokens)
    
    probability_unary_model = assign_probability_unary(build_unary_model(C))
    while 1: #make sure does not start with symbol
        current_word = unary_random_word_generation(probability_unary_model)
        if current_word not in Ending_symbols and current_word != ',':
            break
    
    ret = current_word
    while current_word not in Ending_symbols :
        unary = build_unary_model(d[current_word])
        current_word = unary_random_word_generation(assign_probability_unary(unary))
        ret += " " + current_word
    return ret

def n_ugram_random_sentence_generation(n , task_type, file_type, train_docs="train_docs"):
    for i in range(n):
        print(unary_random_sentence_generation(task_type, file_type))
        
def n_bigram_random_sentence_generation(n, task_type, file_type, train_docs="train_docs"):
    for i in range(n):
        print(bigram_random_sentence_generation(task_type,file_type))

        
n_ugram_random_sentence_generation(10,'sp','medicine')