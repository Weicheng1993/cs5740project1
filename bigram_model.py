
def build_bigram_model(string): # string should be a list of words
    dict = {}
    string = ["\\begin"] + string + ["\stop"] #add begin/stop markers
    for i in range(len(string)):
        if i == len(string) - 1:
            break
        if string[i] not in dict:
            dict.update({string[i]:{string[i+1]:1}}) # format: {first word:{second word#1:count, second word#2: count...}}
        else:
            tmp = dict[string[i]]
            if string[i+1] not in tmp:
                tmp.update({string[i+1]:1})
            else:
                tmp[string[i+1]]+=1

    dict2 = {}
    for i, j in dict.iteritems():
        dict2.update({i:assign_probability_unary(dict)})
    return dict2 # format: {first word:[],...}

#generate one word
def bigram_random_word_generation(dict2, old_word):
    return unary_random_word_generation(dict2[old_word])


#generate a sentence until stop marker met
def bigram_random_n_word_generation(dict2, counter):
    ret = "" # unary_random_word_generation(probability: [tuple]) i.e. use randomly chosen words or '\\begin'
    old_word = ret
    while old_word != "\stop":
        new_word = bigram_random_word_generation(dict2, old_word)
        ret += " " + new_word
        old_word = new_word
    return ret

