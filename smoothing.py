import re
from nltk.tokenize import sent_tokenize
import sys
import random





"""Helper functions for tokenizing"""
def Read_Corpus(path):
    corpus_file = open(path, 'r')
    lines = corpus_file.readlines()
    text = ""
    for line in lines:
        striped_line = line.strip()
        if striped_line != '':
            text = text + " " + striped_line
    return text


def substitue_tokens(sentences):
    sub_sentences = []
    for sentence in sentences:
        sub_sen = sentence.lower()
        sub_sen = re.sub("#[a-zA-Z0-9_]+", "<HASHTAG>", sub_sen)
        sub_sen = re.sub("@[a-zA-Z0-9_]+", "<MENTION>", sub_sen)
        sub_sen = re.sub("https?://[a-zA-Z0-9_./]+", "<URL>", sub_sen)
        sub_sentences.append(sub_sen)
    return sub_sentences


def split_corpus_into_sentences(corpus):
    sentences = sent_tokenize(corpus)
    return sentences


def tokenize_sentence(sen):
    # Handles words 
    word_wise = sen.split()

    # Handeling punctuations
    tokenized_sen = []
    for word in word_wise:
        if(is_punc(word) == True):
            tokenized_sen.append(word)
        else:
            var_word = word
            while(starting_punc(var_word) == True):
                tokenized_sen.append(word[0])
                var_word = var_word[1:]
            
            end_puncs = []
            while(ending_punc(var_word) == True):
                end_puncs = [var_word[-1]] + end_puncs
                var_word = var_word[:-1]
            
            tokenized_sen.append(var_word)
            tokenized_sen += end_puncs
    
    return tokenized_sen


# Does the given word have punctation at the end
def ending_punc(word):
    if(word[-1] == ',') or (word[-1] == ':') or (word[-1] == ';') or (word[-1] == '"') or (word[-1] == ')') or (word[-1] == '}') or (word[-1] == ']') or (word[-1] == '.') or (word[-1] == '?') or (word[-1] == '!'):
        return True
    else:
        return False
    

# Does the given word have punctation at the start
def starting_punc(word):
    if(word[0] == '"') or (word[0] == '(') or (word[0] == '{') or (word[0] == '['):
        return True
    else:
        return False
    

# Is the given word a punctation
def is_punc(word):
    if(len(word) == 1 and (ending_punc(word) or starting_punc(word))):
        return True
    else:
        return False
    

def tokenize_corpus(path):
    corpus = Read_Corpus(path)
    sentences = sent_tokenize(corpus)
    url_metions = substitue_tokens(sentences)
    sentence_tokenized = []
    for sentence in url_metions:
        tokenized_sen = tokenize_sentence(sentence)
        sentence_tokenized.append(tokenized_sen)
    return sentence_tokenized






"""Helper functions for basic ngram model"""
def ngrams_from_tokens_sen(tokens, ngram_size, Counts):
    ngram = ["w" for i in range(ngram_size)]
    for i in range(len(tokens)):
        for j in range(ngram_size - 1):
            ngram[j] = ngram[j + 1]
        ngram[ngram_size - 1] = tokens[i]
        
        history = tuple((ngram[i] for i in range(ngram_size - 1)))
        word = ngram[ngram_size - 1]

        if history in Counts.keys():
            if word in Counts[history].keys():
                Counts[history][word] += 1
            else:
                Counts[history][word] = 1
        else:
            Counts[history] = {}
            Counts[history][word] = 1


def unigram_from_tokens_sen(tokens, Counts):
    for token in tokens:
        if token in Counts.keys():
            Counts[token] += 1
        else:
            Counts[token] = 1


def sum_values(Counts, ngram_size):
    SumVal = {}
    if(ngram_size == 1):
        SumVal["SUM"] = sum(Counts.values())
    else:
        for history in Counts.keys():
            SumVal[history] = sum(Counts[history].values())
    return SumVal


# extract from bigram
def unigram_keser_nay_data(Counts):
    keser_nay_unigram = {}
    
    total_unique_bigrams = 0
    for history in Counts.keys():
        total_unique_bigrams += len(Counts[history].keys())
    keser_nay_unigram["total_bigrams"] = total_unique_bigrams
    
    for history in Counts.keys():
        for word in Counts[history].keys():
            if word in keser_nay_unigram.keys():
                keser_nay_unigram[word] += 1
            else:
                keser_nay_unigram[word] = 1
    return keser_nay_unigram
    
                
def conditional_prob(history, word, Counts, SumCounts):
    if history in Counts.keys():
        if word in Counts[history].keys():
            return Counts[history][word] / SumCounts[history]
        else:
            return 0
    else:
        return 0
    

def unigram_prob(word, Counts, SumCounts):
    if word in Counts.keys():
        return Counts[word] / SumCounts["SUM"]
    else:
        return 0
    

def learn_model_data(token_sentences, N):
    assert((type(N) == int) and (N >= 2))

    Counts_all = [{} for i in range(N + 1)]
    for n in range(1, N + 1):
        if n == 1:
            for sentence in token_sentences:
                unigram_from_tokens_sen(sentence, Counts_all[n])
        else:
            for sentence in token_sentences:
                ngrams_from_tokens_sen(sentence, n, Counts_all[n])

    Sums_all = [{} for i in range(N + 1)]
    for n in range(1, N  + 1):
        Sums_all[n] = sum_values(Counts_all[n], n)

    unigram_keser_nay = unigram_keser_nay_data(Counts_all[2])
    return Counts_all, Sums_all, unigram_keser_nay





"""functions for  smoothing"""
def witten_bell_conditional(history, word, Counts_all, Sums_all, N):
    if N == 1:
        return unigram_prob(word, Counts_all[N], Sums_all[N])
    else:
        if history in Counts_all[N].keys():
            novel_history_words = len(Counts_all[N][history].keys())  #time compleixty O(1)
            accourences_history = Sums_all[N][history]
            prob_word_novel = novel_history_words / (accourences_history + novel_history_words)

            if word in Counts_all[N][history].keys():
                prob_cur = (1 - prob_word_novel) * Counts_all[N][history][word] / Sums_all[N][history]
            else:
                prob_cur = 0
                
            prob_recur = prob_word_novel * witten_bell_conditional(history[1:], word, Counts_all, Sums_all, N - 1)
            return prob_cur + prob_recur
        else:
            return 0


def keser_ney(history, word, Counts_all, Sums_all, kenser_uni, N, d):
    if N == 1:
        if word in kenser_uni.keys():
            return kenser_uni[word] / kenser_uni["total_bigrams"]
        else:
            return 0
    else:
        if history in Counts_all[N].keys():
            novel_history_words = len(Counts_all[N][history].keys())  #time compleixty O(1)
            occourences_history = Sums_all[N][history]
            rec_weight = d * (novel_history_words / occourences_history)
            
            if word in Counts_all[N][history].keys():
                cur_prob = max(0, Counts_all[N][history][word] - d) / occourences_history
            else:
                cur_prob = 0

            recur_prob = keser_ney(history[1:], word, Counts_all, Sums_all, kenser_uni, N - 1, d)
            return rec_weight * recur_prob + cur_prob
        else:
            return 0
        
 



"""functions for likehood and perplexity"""
def likehood_of_sen_tokens(tokens, ngram_size, Counts_all, Sums_all, keser_uni, d, method):
    assert(len(tokens) > 0)
    likelyhood = 1
    ngram = ["w" for i in range(ngram_size)]
    for i in range(len(tokens)):
        for j in range(ngram_size - 1):
            ngram[j] = ngram[j + 1]
        ngram[ngram_size - 1] = tokens[i]
        
        history = tuple((ngram[i] for i in range(ngram_size - 1)))
        word = ngram[ngram_size - 1]
        #print(history, word)

        if method == "witten_bell":
            likelyhood *= witten_bell_conditional(history, word, Counts_all, Sums_all, ngram_size)
        elif method == "keser_ney":
            likelyhood *= keser_ney(history, word, Counts_all, Sums_all, keser_uni, ngram_size, d)
        else:
            assert(False)
    return likelyhood


def perplexity_of_sen_tokens(tokens, ngram_size, Counts_all, Sums_all, keser_uni, d, method):
    assert(len(tokens) > 0)
    likelyhood = likehood_of_sen_tokens(tokens, ngram_size, Counts_all, Sums_all, keser_uni, d, method)
    if likelyhood != 0:
        return pow(likelyhood, -1 / len(tokens))
    else:
        return float("inf")


#infinity perpleixyt case handeling to be done
def store_perplexity(perplexities, sentences, file_path):
    file = open(file_path, "w")
    
    avg_perplexity = sum(perplexities) / len(perplexities)
    file.write(str(avg_perplexity) + "\n")

    for i in range(len(sentences)):
        sen = " ".join(sentences[i])
        perp = str(perplexities[i])
        file.write(sen + "\t" + perp + "\n")
    file.close()



if __name__ == "__main__":
    assert(len(sys.argv) == 4) #or 4
    corpus_path = sys.argv[2]
    smoothing = sys.argv[1]
    lm_index = sys.argv[3]

    # tokenization
    tokenized_corpus = tokenize_corpus(corpus_path)
    print("tokenization done")

    # test train split
    random.shuffle(tokenized_corpus)
    test_sentences = tokenized_corpus[:1000]
    train_sentences = tokenized_corpus[1000:]
    print("test train split done")

    # learn model data
    Counts_all, Sums_all, keser_uni = learn_model_data(train_sentences, 4)
    d = 0.5 #hyperparameter
    print("model data learned")
    
    # defining smoothing method (used in perplexity and likehood functions)
    if smoothing == "k":
        method = "keser_ney"
    elif smoothing == "w":
        method = "witten_bell"
    else:
        assert(False)

    # perplexity caliculation
    ## Train perplexity
    train_perplexity = []
    for sentence in train_sentences:
        perplexity = perplexity_of_sen_tokens(sentence, 4, Counts_all, Sums_all, keser_uni, d, method)
        train_perplexity.append(perplexity)
    print("train perplexity done")
    ## Test perplexity
    test_perplexity = []
    for sentence in test_sentences:
        perplexity = perplexity_of_sen_tokens(sentence, 4, Counts_all, Sums_all, keser_uni, d, method)
        test_perplexity.append(perplexity)
    print("test perplexity done")

    # storing perlplexities
    ## train
    file_path = "./2020101024_LM" + str(lm_index) + "_train-perplexity" + ".txt"
    store_perplexity(train_perplexity, train_sentences, file_path)
    ## test
    file_path = "./2020101024_LM" + str(lm_index) + "_test-perplexity" + ".txt"
    store_perplexity(test_perplexity, test_sentences, file_path)
    
    
    exit(0)