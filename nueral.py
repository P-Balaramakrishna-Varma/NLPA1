from nltk.tokenize import sent_tokenize
import re
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm





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
            while(len(var_word) != 0 and starting_punc(var_word) == True):
                tokenized_sen.append(word[0])
                var_word = var_word[1:]
            
            end_puncs = []
            while(len(var_word) != 0 and ending_punc(var_word) == True):
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





"""Helper functions for vocabulary"""
def unigram_from_token_corpus(tokens_sen):
    Counts = {}
    for tokens in tokens_sen:
        for token in tokens:
            if token in Counts.keys():
                Counts[token] += 1
            else:
                Counts[token] = 1
    return Counts


def vocab_index(unigram_counts):
    vocab_index = {}
    index = 0
    for key in unigram_counts.keys():
        if(unigram_counts[key] > 1):
            vocab_index[key] = index
            index += 1
    vocab_index["<UNK>"] = index       
    return vocab_index




""" Helper functions for data"""
def get_index(word, vocab_index):
    if word in vocab_index.keys():
        return vocab_index[word]
    else:
        return vocab_index["<UNK>"]
    

def get_data(tokenized_corpus, vocab_idx, batch_size):
    data = []                                                   
    for tokenized_sentence in tokenized_corpus:            
            tokens = [get_index(token, vocab_idx) for token in tokenized_sentence] 
            data.extend(tokens)              
    data = torch.Tensor(data)                                 
    num_batches = data.shape[0] // batch_size 
    data = data[:num_batches * batch_size]                       
    data = data.view(batch_size, num_batches)          
    return data


def get_batch(data, len, num_batches, idx):
    assert(idx + len + 1 < num_batches)
    X = data[:, idx: idx + len]                   
    y = data[:, idx + 1: idx + len + 1]             
    return X, y




"""Nueral Network architecture"""
class language_model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, no_layers=1):        
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.no_layers = no_layers

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # lstm layer single lstm
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layer=no_layers, batch_first=True)
        # output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input, hidden):
        embedding = self.embedding(input)
        output, hidden = self.lstm(embedding, hidden)          
        prediction = self.fc(output)
        return prediction, hidden
    
    def detach_hidden(self, hidden):
        hidden_h, cell = hidden
        hidden_h = hidden_h.detach()
        cell = cell.detach()
        return hidden_h, cell
    
    def init_hidden(self, batch_size, device):
        hidden = torch.zeros(self.no_layers, batch_size, self.hidden_dim).to(device)
        cell = torch.zeros(self.no_layers, batch_size, self.hidden_dim).to(device)
        return hidden, cell
    


"""Training and Testing loops"""
def train_epoch(model, data, optimizer, loss_function, batch_size, len, max_norm, device):
    # drop all batches that are not a multiple of len
    num_batches = data.shape[-1]
    data = data[:, :num_batches - (num_batches -1) % len]
    num_batches = data.shape[-1]
 
    total_loss = 0
    hidden = model.init_hidden(batch_size, device)
    
    for idx in tqdm.tqdm(range(0, num_batches - 1, len)):
        optimizer.zero_grad()
        hidden = model.detach_hidden(hidden)

        X, y = get_batch(data, len, num_batches, idx)
        X, y = X.to(device), y.to(device)
        pred, hidden = model(X, hidden)               
        pred = pred.reshape(batch_size * len, -1)   
        y = y.reshape(-1)
        loss = loss_function(pred, y)
    
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm) #avoid exploding gradient
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss


def evaluate(model, data, loss_function, batch_size, len, device):
    model.eval()

    num_batches = data.shape[-1]
    data = data[:, :num_batches - (num_batches -1) % len]
    num_batches = data.shape[-1]

    total_loss = 0
    hidden = model.init_hidden(batch_size, device)

    with torch.no_grad():
        for idx in tqdm.tqdm(range(0, num_batches - 1, len)):
            hidden = model.detach_hidden(hidden)
            X, y = get_batch(data, len, num_batches, idx)
            X, y = X.to(device), y.to(device)
            pred, hidden = model(X, hidden) 

            pred = pred.reshape(batch_size * len, -1)
            y = y.reshape(-1)
            loss = loss_function(pred, y)
            total_loss += loss.item()
    return total_loss


