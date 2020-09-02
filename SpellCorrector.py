# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 14:27:56 2020

@author: USER
"""
import re
from collections import Counter
import numpy as np
import pandas as pd
import math
import random
import numpy as np
import pandas as pd
import nltk
import Candidates
import OOV
import Ngram
import ErrorModel

# Read file
with open("514-8.txt", "r") as f:
    data = f.read()
#Preprocess file    
data = re.sub(r'[^A-Za-z\.\?!\']+', ' ', data) #remove special character
data = re.sub(r'[A-Z]{3,}[a-z]+', ' ',data) #remove words with more than 3 Capital letters
sentences = re.split(r'[\.\?!]+[ \n]+', data) #split data into sentences
sentences = [s.strip() for s in sentences] #Remove leading & trailing spaces
sentences = [s for s in sentences if len(s) > 0] #Remove whitespace


tokenized_sentences=[]
for sentence in sentences:
        
        # Convert to lowercase letters
        sentence = sentence.lower()
        
        # Convert into a list of words
        tokenized = nltk.word_tokenize(sentence)
        
        # append the list of wtokenized_sentencesto the list of lists
        tokenized_sentences.append(tokenized)
        

# Get Vocabulary
vocabulary = list(set(OOV.get_nplus_words(tokenized_sentences, 2)))
vocabulary = vocabulary+['<s>']+['<e>']
# Replace less frequent word by <UNK>
processed_sentences = OOV.replace_words_below_n_by_unk(tokenized_sentences, 2)
# Get the unigram and bigram
unigram_counts = Ngram.n_grams_dict(processed_sentences, 1)
bigram_counts = Ngram.n_grams_dict(processed_sentences, 2)


def get_probability(previous_n_words, word, 
                         previous_n_gram_dict, n_gram_dict, vocabulary_size, k=1.0):
    """
    Return N-gram probability given the pair of current word and previous_n_words
    """
    assert type(previous_n_words) == list
    # convert list to tuple to use it as a dictionary key
    previous_n_words = tuple(previous_n_words,)
    
    previous_n_words_count = previous_n_gram_dict[previous_n_words] if previous_n_words in previous_n_gram_dict else 0

    # k-smoothing
    denominator = previous_n_words_count + k*vocabulary_size

    # Define n plus 1 gram as the previous n-gram plus the current word as a tuple
    n_gram = previous_n_words + (word,)

    n_gram_count = n_gram_dict[n_gram] if n_gram in n_gram_dict else 0
   
    # smoothing
    numerator = n_gram_count + 1

    probability = numerator/denominator
    
    
    return probability


def get_corrections(previous_n_words_i, word, vocab, n=2, verbose = False):
    '''
    Get n candidates with individual probability
    '''
    assert type(previous_n_words_i) == list
    corpus = ' '.join(vocabulary)
    suggestions = []
    n_best = []
    ### Convert to UNK if word not in vocab
    previous_n_words = []
    for w in previous_n_words_i:
        if w not in vocabulary:
            previous_n_words.append('<unk>')
        else:
            previous_n_words.append(w)
            
    ##Suggestions include input word only if the input word in vocab
    if word in vocab:    
        suggestions = [word] + list(Candidates.edit_one_letter(word).intersection(vocabulary)) or list(Candidates.edit_two_letters(word).intersection(vocabulary)) 
    else:
        suggestions = list(Candidates.edit_one_letter(word).intersection(vocabulary)) or list(Candidates.edit_two_letters(word).intersection(vocabulary)) 
        
    words_prob = {}
    for w in suggestions: 
        # To make sure all suggestions is within edit distance of 2
        _, min_edits = Candidates.min_edit_distance(' '.join(word),w)
        if not word in vocab: ##use error model only when it is non word error
            if min_edits <= 2:
                edit = ErrorModel.editType(w,' '.join(word))
                if edit:##Some word cannot find edit
                    if edit[0] == "Insertion":
                        error_prob = ErrorModel.channelModel(edit[3][0],edit[3][1], 'add',corpus)
                    if edit[0] == 'Deletion':
                        error_prob = ErrorModel.channelModel(edit[4][0], edit[4][1], 'del',corpus)
                    if edit[0] == 'Reversal':
                        error_prob = ErrorModel.channelModel(edit[4][0], edit[4][1], 'rev',corpus)
                    if edit[0] == 'Substitution':
                        error_prob = ErrorModel.channelModel(edit[3], edit[4], 'sub',corpus)
                else:
                    error_prob = 1
            else:
                error_prob = 1
        else:
            error_prob = 1
        language_prob = get_probability(previous_n_words, w, 
                        unigram_counts, bigram_counts, len(vocabulary), k=1.0)
            
        words_prob[w] = language_prob * error_prob
        
    n_best = Counter(words_prob).most_common(n)
    
    if verbose: print("entered word = ", word, "\nsuggestions = ", suggestions)

    return n_best
