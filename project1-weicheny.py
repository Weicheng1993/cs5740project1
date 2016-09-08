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

Bad_symbols = ",.:;'\"!#$%&()*+-/<=>@[\]^_`{|}~<>\|?!\\"
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
# print(format_file_name("sp", "religion", 4,"train_docs"))
# print(os.path.exists(format_file_name("sp", "religion", 4),))

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
    return word_tokenize(file_content)
       
def bow(tokens: [str]):
    return Counter(tokens)

def handle_file(task_type: str, file_type: str, file_number: int, train_docs="train_docs"):
    return bow(tokenize(preprocess_content(read_file(task_type, file_type, file_number, train_docs))))


def build_unary_model(c: Counter):
    total = sum(c.values())
    new_counter = Counter()
    new_counter.update(c)
    for i in c:
        new_counter[i] = c[i] / total
    return new_counter

def assign_probability_unary(c: Counter)->[tuple]:
    lower_bound = 0
    ret = []
    for i in c:
        ret.append((lower_bound, lower_bound+c[i], i))
        lower_bound += c[i]
    return ret
    
    

def unary_random_word_generation(probability: [tuple]):
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

def unary_random_n_words_generation(probability, n):
    ret = ''
    for i in range(n):
        ret += " " + unary_random_word_generation(probability)
    return ret

def unary_random_sentence_generation(task_type, file_type, sentence_length, train_docs="train_docs"):
    C = Counter()
    for i in range(300):
        C.update(handle_file(task_type, file_type, i, train_docs))
    model = build_unary_model(C)
    return unary_random_n_words_generation(assign_probability_unary(model), sentence_length).strip()
            
# sample = handle_file("sp", "religion", "10")
# unary_model = build_unary_model(sample)
# probability_unary_model = assign_probability_unary(unary_model)
unary_random_sentence_generation('sp','religion', 100)
