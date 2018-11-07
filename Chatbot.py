# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 04:54:04 2018

@author: swathi
"""

import numpy as np
import tensorflow as tf
import re 
import time 

lines = open('movie_lines.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
conversations = open('movie_conversations.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')

id2Line = {}
for line in lines:
    _line = line.split(" +++$+++ ")
    if len(_line)==5:
        id2Line[_line[0]] = _line[4]
        
conversations_ids = []
for conversation in conversations[:-1]:
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    conversations_ids.append(_conversation.split(','))
    
questions = []
answers = []
for conversation in conversations_ids:
    for i in range(len(conversation)-1):
        questions.append(id2Line[conversation[i]])
        answers.append(id2Line[conversation[i + 1]])
        
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm","i am",text)
    text = re.sub(r"he's","he is",text)
    text = re.sub(r"she's","she is",text)
    text = re.sub(r"that's","that is",text)
    text = re.sub(r"what's","what is",text)
    text = re.sub(r"\'ll"," will",text)
    text = re.sub(r"\'ve"," have",text)
    text = re.sub(r"\'re"," are",text)
    text = re.sub(r"\'d"," would",text)
    text = re.sub(r"[-{}\"#/@:;<>()+=~|.?,]","",text)
    return text                

questions_clean = []
for question in questions:
    questions_clean.append(clean_text(question))
    
answers_clean = []
for answer in answers:
    answers_clean.append(clean_text(answer))
    
word2Count = {}
for question in questions_clean:
    for word in question.split():
        if word not in word2Count:
            word2Count[word] = 1
        else:
            word2Count[word] += 1
for answer in answers_clean:
   for word in answer.split():
       if word not in word2Count:
           word2Count[word] = 1
       else:
           word2Count[word] += 1
        
threshold = 20
questionsWords2int = {}
word_number = 0
for word,count in word2Count.items():
    if count >= threshold:
        questionsWords2int[word] = word_number
        word_number += 1
answerssWords2int = {}
word_number = 0
for word,count in word2Count.items():
    if count >= threshold:
        answerssWords2int[word] = word_number
        word_number += 1

tokens = ['<PAD>','<EOS>','<OUT>','<SOS>']
for token in tokens:
    questionsWords2int[token] = len(questionsWords2int) + 1
    answerssWords2int[token] = len(answerssWords2int) + 1
    
answersint2word = {w_i:w for w,w_i in answerssWords2int.items()}

for i in range(len(answers_clean)):
    answers_clean[i] += ' <EOS>'

questions2int = []
for question in questions_clean:
    ints = []
    for word in question.split():
        if word not in questionsWords2int:
            ints.append(questionsWords2int['<OUT>'])
        else:
            ints.append(questionsWords2int[word])
    questions2int.append(ints)
answers2int = []
for answer in answers_clean:
    ints = []
    for word in answer.split():
        if word not in answerssWords2int:
            ints.append(answerssWords2int['<OUT>'])
        else:
            ints.append(answerssWords2int[word])
    answers2int.append(ints)
    
sortedQuestions = []
sortedAnswers = []

for length in range(1, 25):
    for i in enumerate(questions2int):
        if len(i[1]) == length:
            sortedQuestions.append(questions2int[i[0]])
            sortedAnswers.append(answers2int[i[0]])

#creating placeholders for inputs and targets
def model_inputs():
    inputs = tf.placeholder(tf.int32, [None,None], name = 'inputs')
    targets = tf.placeholder(tf.int32, [None,None], name = 'targets')
    learning_rate = tf.placeholder(tf.float32, name = 'learning_rate')
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
    return inputs,targets,learning_rate,keep_prob

def preprocess_targets(targets, word2int, batch_size):
    left_side = tf.fill([batch_size, 1],word2int['<SOS>'])
    right_side = tf.strided_slice(targets, [0,0], [batch_size,-1], [1,1])
    preprocessed_targets = tf.concat([left_side,right_side],1)
    return preprocessed_targets

def encoder_rnn_layer(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    _, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell, 
                                                       cell_bw=encoder_cell,
                                                       sequence_length=sequence_length,
                                                       inputs=rnn_inputs,
                                                       dtype=tf.float32)
    return encoder_state

def decode_training_set(encoder_state,decoder_cell,decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size,1,decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states,
                                                                                                                                    attention_option = 'bahdanau',
                                                                                                                                    num_units=decoder_cell.output_size)
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0], attention_keys, attention_values, attention_score_function, attention_construct_function, name = 'attn_dec_train')
    decoder_output,decoder_final_state,decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                            training_decoder_function,
                                                                                                            decoder_embedded_input,
                                                                                                            sequence_length,
                                                                                                            scope=decoding_scope)
    decoder_output_dropout = tf.nn.dropout(decoder_output,keep_prob)
    return output_funtion(decoder_output_dropout)

#decoding test/validation set
def decode_test_set(encoder_state,decoder_cell,decoder_embedded_matrix,sos_id,eos_id,maximum_length,num_words, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size,1,decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states,
                                                                                                                                    attention_option = 'bahdanau',
                                                                                                                                    num_units=decoder_cell.output_size)
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                              encoder_state[0], 
                                                                              attention_keys,
                                                                              attention_values, 
                                                                              attention_score_function, 
                                                                              attention_construct_function, 
                                                                              decoder_embedded_matrix,sos_id,eos_id,maximum_length,num_words,
                                                                              name = 'attn_dec_inf')
    test_predictions,decoder_final_state,decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                            test_decoder_function,
                                                                                                            scope=decoding_scope)
    return test_predictions

def decoder_rnn(decoder_embedded_input,decoder_embedded_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
        weights = tf.truncated_normal_initializer(stddev = 0.1)
        biases = tf.zeros_initializer()
        output_function = lambda x: tf.contrib.layers.fully_connected(x,
                                                                      num_words,
                                                                      None,
                                                                      scope=decoding_scope,
                                                                      weights_initializers = weights,
                                                                      biases_initializers = biases)
        trining_predictions = decode_training_set(encoder_state, decoder_cell, decoder_embedded_input,
                                                  sequence_length,decoding_scope, output_function,keep_prob,
                                                  batch_size)
        
    
    