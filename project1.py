from nltk import word_tokenize
from collections import Counter
import random
import os
import nltk

REMOVE_IRRELEVANT_TEXT = False
ADD_SENTENCE_BUUNDARY_TAG = False
DIFFERNTIATE_CAPS = False

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
    with open(format_file_name(task_type, file_type, file_number, train_docs)) as f:
        return f.read()

def tokenize(file_content: str):
    return word_tokenize(file_content)
       
def bow(tokens: [str]):
    return Counter(tokens)

def handle_file(task_type: str, file_type: str, file_number: int, train_docs="train_docs"):
    return bow(tokenize(read_file(task_type, file_type, file_number, train_docs)))


def build_unary_model(c: Counter):
    total = sum(c.values())
    new_counter = Counter()
    new_counter.update(c)
    for i in c:
        new_counter[i] = c[i] / total
    return new_counter


sample = handle_file("sp", "graphics", "10", "train_modified_docs")
sample.most_common(4)