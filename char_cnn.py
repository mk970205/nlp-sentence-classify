import tensorflow as tf
file_path = 'C:/NLP_Term/HW1-master/data/stsa.binary.train'

def makeCharVocab(): # 문서에서 나온 모든 Character를 리스트로 만듬(중복 제외)
    vocab=[]
    with open(file_path, mode='r') as f:
        while True:
            line = f.readline()
            if not line: break
            for i in line:
                if i not in vocab:
                    vocab.append(i)
    return vocab
 
def convertCharToID(char_list, char_vocab): # Character를 ID(숫자)로 변환
    id_list=[]
    for i in char_list:
        id_list.append(char_vocab.index(i))
    return id_list

def getMaxWordLength(): #문서에서 가장 긴 단어의 길이를 반환
    maxSize=0
    with open(file_path, mode='r') as f:
        while True:
            line = f.readline()
            if not line: break
            for i in line.split():
                if maxSize<len(i):
                    maxSize=len(i)
    return maxSize

def getMaxSequenceLength(): #문서에서 가장 긴 문장의 단어 수를 반환
    maxSize=0
    with open(file_path, mode='r') as f:
        while True:
            line = f.readline()
            if not line: break
            if maxSize < len(line.split()):
                maxSize=len(line.split())
    return maxSize

embedding_size_char = 6
max_word_length = getMaxWordLength()
max_seqence_length = getMaxSequenceLength()
batch_size=2
filter_width = 4
char_vocab = makeCharVocab()
vocab_size = len(char_vocab)

W = tf.random_uniform([vocab_size, embedding_size_char],-1.0,1.0)
char_ids = tf.placeholder(tf.int32, shape=[None, None, None]) #(batch size, max length of sentence, max length of word)
char_embeddings = tf.nn.embedding_lookup(W,char_ids)
s_charemb=tf.shape(char_embeddings)

char_embeddings = tf.reshape(char_embeddings,shape=[s_charemb[0]*s_charemb[1], max_word_length, embedding_size_char, 1]) # (batch x sentence, max length of sentence, embeddings size, Number of layers(filters))
Filter1 = tf.Variable(tf.truncated_normal(shape=[filter_width, embedding_size_char,1,1],stddev=0.1)) #(Filter_width, Embedding size, Layer Size, Number of filters)
Bias1 = tf.Variable(tf.truncated_normal(shape=[1],stddev=0.1)) #(Number of filters)
Conv1 = tf.nn.conv2d(char_embeddings, Filter1, strides=[1,1,1,1], padding='SAME') + Bias1 
Activation1 = tf.nn.relu(Conv1)
Pool1 = tf.nn.max_pool(Activation1, ksize=[1,max_word_length,1,1], strides=[1,max_word_length,1,1], padding='SAME')
Pool1 = tf.squeeze(Pool1)
output = tf.reshape(Pool1, shape = [-1, max_seqence_length, embedding_size_char]) #(batch size, max length of sentence, embeddings size)
 
with tf.Session() as sess:
    print("Start....")

    
    with open(file_path, mode='r') as f:
        count = 0
        batch=[]
        while True:
            line = f.readline()
            
            sentence=[[1 for col in range(max_word_length)] for row in range(max_seqence_length)] 
            i=0
            for word in line.split():
                input_word_fixed_length=[' ' for col in range(max_word_length)] 
                input_word = list(word)
                j=0
                for character in input_word:
                    input_word_fixed_length[j] = character
                    j=j+1
                input_word_fixed_length = convertCharToID(input_word_fixed_length, char_vocab)
                sentence[i]=input_word_fixed_length
                i=i+1
            batch.append(sentence)
            print(line)
            count=count+1
            if(count==batch_size):
                sess.run(tf.global_variables_initializer())
                print(sess.run(output,feed_dict={char_ids:batch}))
                batch=[]
                count=0