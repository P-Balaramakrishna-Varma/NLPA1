from nltk.tokenize import sent_tokenize
import re
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import random
import sys




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


def append_start(tokenized_corpus):
    for sentence in tokenized_corpus:
        sentence.insert(0, "<START>")
    return tokenized_corpus




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
    data = torch.LongTensor(data)                                 
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
    def __init__(self, vocab_size, embedding_dim, hidden_dim, no_layers):        
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.no_layers = no_layers

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # lstm layer single lstm
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=no_layers, batch_first=True)
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
    model.train()
    num_batches = data.shape[-1]
    data = data[:, :num_batches - (num_batches -1) % len]
    num_batches = data.shape[-1]
 
    total_loss = 0
    hidden = model.init_hidden(batch_size, device)
    
    for idx in tqdm.tqdm(range(0, num_batches - 1 - len, len)):
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
        for idx in tqdm.tqdm(range(0, num_batches - 1 - len, len)):
            hidden = model.detach_hidden(hidden)
            X, y = get_batch(data, len, num_batches, idx)
            X, y = X.to(device), y.to(device)
            pred, hidden = model(X, hidden) 

            pred = pred.reshape(batch_size * len, -1)
            y = y.reshape(-1)
            loss = loss_function(pred, y)
            total_loss += loss.item()
    return total_loss






if __name__ == "__main__":
    if (torch.cuda.is_available() == True):
        device = torch.device("cuda:" + sys.argv[1])
    else:
        device = torch.device("cpu")

    # Tokenizing corpus    
    corpus_path = sys.argv[2]
    tokenized_corpus = tokenize_corpus(corpus_path)
    tokenize_corpus = append_start(tokenized_corpus)

    # Creating vocabulary
    unigram_counts = unigram_from_token_corpus(tokenized_corpus)
    vocab_idx = vocab_index(unigram_counts)

    # Creating data and test dev split
    random.shuffle(tokenized_corpus)
    length = len(tokenized_corpus)
    train_sentences = tokenized_corpus[:int(length*0.7)]
    test_sentences = tokenized_corpus[int(length*0.15):]
    dev_sentences = tokenized_corpus[int(length*0.15):int(length*0.7)]
    
    test_data = get_data(test_sentences, vocab_idx, 64)
    dev_data = get_data(dev_sentences, vocab_idx, 64)
    train_data = get_data(train_sentences, vocab_idx, 64)

    # Creating model
    ## Hyperparameters
    vocab_size = len(vocab_idx)
    embedding_dim = 300
    hidden_dim = 1024  
    no_layers = 1                               
    lr = 1e-2
    batch_size = 64
    len = 35
    max_norm = 10

    model = language_model(vocab_size, embedding_dim, hidden_dim, no_layers).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()

    file = open("model_" + sys.argv[1] + ".txt", "w")
    epochs = 1000
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        l1 = train_epoch(model, train_data, optimizer, loss_func, batch_size, len, max_norm, device)
        print("loss train: ", l1)
        l2 = evaluate(model, dev_data, loss_func, batch_size, len, device)
        print("loss test: ", l2)
        if(t % 50 == 0):
            torch.save(model.state_dict(), "model_" + sys.argv[1] + "__" + str(t) + ".pt")
            file.write("Epoch: " + str(t) + " loss train: " + str(l1) + " loss test: " + str(l2) + "\n")
    print("Done!")