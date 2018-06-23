# -*- coding: utf-8 -*-


import tensorflow as tf
import numpy as np
import operator


filePath = "C:/NLP_Term/HW1-master/data/stsa.binary.train"

# 문서의 모든 단어가 들어 있는 리스트를 만듬(중복 제외)
def makeDictionary():
    dic = []
    
    f = open(filePath, 'r')
    while True:
        line = f.readline()
        if not line: break
        for word in line.split():
            if word not in dic:
                dic.append(word)
    f.close()
    return dic

# 문서에서 가장 단어의 개수가 많은 문장을 찾은 다음 그 문장의 단어 수를 반환
def findMaxLineLength():
    f = open(filePath, 'r')
    lineLen = 0
    maxLineLen = 0
    while True:
        line = f.readline()
        if not line: break
        lineLen=len(line.split())
        maxLineLen=max(lineLen,maxLineLen)
    return maxLineLen

tf.reset_default_graph()
sentimental_set = ['?','X','O']
#x_input_word = [[[0],[1],[2],[3],[3],[2]]]
#y_answer=[[1,1,1,1,2,1]]

# Config Value
num_classes = 3 # output의 클래스 수 ('?', 'X', 'O' => 3개)
input_dim = 1 # input의 Dimension
hidden_size = 100 # rnn cell size
batch_size = 1 # batch size
sequence_length = findMaxLineLength() # 가장 긴 문장 기준의 length
learning_rate=0.1


X = tf.placeholder(tf.float32, [batch_size, sequence_length, input_dim])
Y = tf.placeholder(tf.int32,[batch_size,num_classes])
early_stop = tf.placeholder(tf.int32) 

cell=tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs1, _states= tf.nn.dynamic_rnn(cell, X, initial_state=initial_state, dtype=tf.float32)
#print(outputs1)
X_for_rc = tf.reshape(outputs1, [-1, hidden_size])
outputs2=tf.contrib.layers.fully_connected(inputs=X_for_rc, num_outputs=num_classes, activation_fn=None)
#print(outputs2)
outputs3=tf.reshape(outputs2, [batch_size, sequence_length, num_classes])
#print(outputs3)
output_rnn_last=outputs3[:,early_stop-1,:]
weights = tf.ones([batch_size, sequence_length])

losses = tf.nn.softmax_cross_entropy_with_logits(logits=output_rnn_last, labels=Y)
#sequence_loss=tf.contrib.seq2seq.sequence_loss(logits=outputs3, targets=Y,weights=weights)

loss=tf.reduce_mean(losses)
train=tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)
prediction = output_rnn_last

with tf.Session() as sess:
    dic=makeDictionary()
    sess.run(tf.global_variables_initializer())
    i=0
    f = open(filePath, 'r')
    
    max_seq_length = findMaxLineLength()
    
    #한 문장씩 rnn 모델에 넣어줌
    while True:
        line=f.readline()
        if not line: break
        
        x_input_word=[[[0 for col in range(input_dim)] for row in range(max_seq_length)] for depth in range(batch_size)] #(batch size, max_seq_length(문장 길이가 짧으면 패딩),input_dim)
        y_answer=[[0 for col in range(num_classes)] for row in range(batch_size)] #(batch size, num_classes)
        word_index=0
        for word in line.split(): 
            x_input_word[0][word_index][0]=dic.index(word) # input배열에 word를 번호로 치환한 값 넣어줌
            word_index=word_index+1
            
        y_answer[0][int(line.split()[0])+1]=1 # class가 0이면 [[1, 0, 0]], class가 1이면 [[0, 1, 0]], class가 2이면 [[0, 0, 1]]
        

        #print(x_input_word)
        #print(y_answer)
        
        l, _ = sess.run([loss, train], feed_dict = {X: x_input_word, Y:y_answer, early_stop:len(line.split())})
        result = sess.run(prediction, feed_dict={X: x_input_word,  Y:y_answer, early_stop:len(line.split())})
        
        print(i, "loss:", l, "predction: ", result, "true Y: ", y_answer)
        i=i+1
        
        index, value = max(enumerate(result[0]), key=operator.itemgetter(1))
        print(sentimental_set[index],"/",sentimental_set[int(line.split()[0])+1]) # 예상class / 실제class
        #result_str = [sentimental_set[c] for c in np.squeeze(result)]
        #print("\tPrediction str: ", ''.join(result_str))
    f.close()
