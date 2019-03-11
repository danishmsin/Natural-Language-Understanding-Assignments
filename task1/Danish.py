# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 13:22:17 2019

@author: Danish
"""


import tensorflow as tf
import numpy as np
from nltk.corpus import reuters
import keras.preprocessing.text
import math


#Separating Training and Test Data...
training_data = [items for items in reuters.fileids() if("train") in items]
test_data = [items for items in reuters.fileids() if("test") in items]
        
        

#words_in_sent contains list of words for each sentence..
words_in_sent = reuters.sents(training_data)

#sents contain list of sentences.. 
sents = [' '.join(item) for item in words_in_sent]

#Filtering non-text data..
valid_text_data = list()
for i in range(len(sents)):
    valid_text_data.append(keras.preprocessing.text.text_to_word_sequence(sents[i], filters='0123456789!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' '))

#Unique collection of Words in the Data..                                                                    
vocab = set()
for i in range(len(valid_text_data)):
    s = set(valid_text_data[i])
    vocab.update(s)
vocab  = list(vocab)

#Creating Word pairs with window size = 2
word_pairs_2 = []
WINDOW_SIZE = 2
for sentence in valid_text_data:
    for word_index, word in enumerate(sentence):
        for nb_word in sentence[max(word_index - WINDOW_SIZE, 0) : min(word_index + WINDOW_SIZE, len(sentence)) + 1] : 
            if nb_word != word:
                word_pairs_2.append([word, nb_word])


#Dictionary Size..
vocab_size = len(vocab)

#Creating two dictionaries with indices as numbers and values as words and vice versa..
num_to_word = {}
word_to_num = {}

num_to_word = {i:v for i,v in enumerate(vocab)}
word_to_num = {v:i for i,v in enumerate(vocab)}
    
#Dividing the pairs and storing each as an list..  
pair1 = list()
pair2 = list()

for i in range(len(word_pairs_2)):
    pair1.append(word_pairs_2[i][0])
    pair2.append(word_pairs_2[i][1])
  

#Replacing both the pairs with their corresponding indices..
    
pair1_ind = list()
pair2_ind = list()

for i in range(len(pair1)):
    pair1_ind.append(word_to_num[pair1[i]])
    pair2_ind.append(word_to_num[pair2[i]])
    


pair1_ind = np.array(pair1_ind)
pair2_ind = np.array(pair2_ind)


def cosine_similarity(vec1,vec2):
 
    norm_1 = vec1/np.linalg.norm(vec1)
    norm_2 = vec2/np.linalg.norm(vec2)
    
    return  np.dot(norm_1,norm_2)
    

def ques_ans(word1,word2,word3, embeddings):
    max_index = -1*np.ones(10)
    max_value = -1000*np.ones(10)
    ind_1 = word_to_num[word1]
    ind_2 = word_to_num[word2]
    ind_3 = word_to_num[word3]
    arr = []
   
    
    embedded_vector_1 = embeddings[ind_1]
    embedded_vector_2 = embeddings[ind_2]
    embedded_vector_3 = embeddings[ind_3]
    
    embedded_vector_4 = embedded_vector_2 - embedded_vector_1 + embedded_vector_3
    
    for cnt in range(1,10):
        for i,item in enumerate(embeddings):
            if cosine_similarity(embedded_vector_4,item)> max_value[cnt] and not np.array_equal(item,embedded_vector_2) and not np.array_equal(item,embedded_vector_3) and not np.array_equal(item,embedded_vector_1):
                if i not in max_index:
                    max_value[cnt] = cosine_similarity(embedded_vector_4,item)
                    max_index[cnt] = i
        arr.append(num_to_word[max_index[cnt]])
    return arr
            


#Model Training using TensorFlow...
no_of_iter = 100
for batch_size in [64,128]:
    for neg_samples in [16,32]:
        for embedding_size in [200,300]:
            embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
            nce_weights = tf.Variable(tf.truncated_normal([vocab_size, embedding_size],stddev=1.0 / math.sqrt(embedding_size)))
            nce_biases = tf.Variable(tf.zeros([vocab_size]))
            train_pair1_ind = tf.placeholder(tf.int32, shape=[None,])
            train_pair2_ind = tf.placeholder(tf.int32, shape=[None,1])
            embed = tf.nn.embedding_lookup(embeddings, train_pair1_ind)
            sess = tf.Session()
            init = tf.global_variables_initializer()
            sess.run(init) #make sure you do this!
            loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,biases=nce_biases,labels=train_pair2_ind,inputs=embed,num_sampled=neg_samples,num_classes=vocab_size))
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)
            n_iters = int(np.ceil(len(pair1_ind)/batch_size))
            print(n_iters)
            # train for n_iter iterations
            for cnt in range(no_of_iter):
                index= 0
                for i in range(n_iters):
                    sess.run(optimizer, feed_dict={train_pair1_ind: pair1_ind[index:index+batch_size], train_pair2_ind: np.expand_dims(pair2_ind[index:index+batch_size],axis=1)})
                    index = index + batch_size
                print('loss is : ', sess.run(loss, feed_dict={train_pair1_ind: pair1_ind[0:batch_size], train_pair2_ind: np.expand_dims(pair2_ind[0:batch_size],axis=1)}))
            
                print('epoch %d done'%cnt)    
                
            RUN = np.array(sess.run(embeddings))
            text = RUN.tolist()
            
            for i,item in enumerate(text):
                item.insert(0,num_to_word[i] )
            
            EF = [' '.join(map(str,item)) for item in text]
            myfile = open(batch_size.__str__()+"_"+neg_samples.__str__()+"_"+embedding_size.__str__()+".txt","w")
            
            for item in EF:
                myfile.write("%s\n" % item)
