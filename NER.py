# -*- coding: utf-8 -*-
"""

@author: HEMIL
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import tensorflow as tf
#%matplotlib inline

#https://github.com/dice-group/FOX/tree/master/input/Wikiner
data_file = "aij-wikiner-en-wp3"

# Load data
print("Reading the dataset.....")
with open(data_file, "r", encoding="latin1") as data_handle:
    text = data_handle.readlines()

print(text[1])

text_flat = []
for item in text:
    text_flat.extend(item.split(" "))
print(text_flat[1])

text = []
for item in text_flat:
    text.append(item.split("|"))
print(text[1])

#data = pd.DataFrame(text[0:200000], columns=["Word", "POS", "NER"])
data = pd.DataFrame(text[0:10000], columns=["Word", "POS", "NER"]) #take a small chuck of dataset to train faster.
data.drop(["POS"], axis=1, inplace=True)
del text, text_flat


#get rid of BIO labels and instead use the 4 classes directly this is because the data is imbalanced and there are barely any B-* classes
classes_map = {
    # Incorrect classes
    "B-ERS": "PERS",
    "B-MSIC": "MISC",
    "B-OEG": "ORG",
    "B-ORF": "ORG",
    "B-PERs": "PERS",
    "B-PRG": "ORG",
    "I-PRG": "ORG",
    "IPERS": "PERS",
    "": "O",
    "o": "O",
    "O\n": "O",
    None: "O",
    "I-MISC\n": "MISC",
    "B-MIS0": "MISC",
    "B-MIS1": "MISC",
    "B-MIS2": "MISC",
    "B-MIS3": "MISC",
    "I-MIS0": "MISC",
    "I-MIS1": "MISC",
    "I-MIS2": "MISC",
    "I-MIS3": "MISC",
    "B-SPANISH": "O",
    "I-SPANISH": "O",
    "B-MIS": "MISC",
    "I-MIS": "MISC",
    "B-MIS-2": "MISC",
    "B-MIS-1": "MISC",
    "B-MIS1'": "MISC",
    "I-MIS": "MISC",
    "I-LOC\n": "LOC",
    "OO": "O",
    "I--ORG": "ORG",
    "I-ORG\n": "ORG",
    "B-MISS1": "MISC",
    "IO": "O",
    "B-ENGLISH": "O",
    "I-PER\n": "PERS",
    "B-PER": "PERS",
    "I-PER": "PERS",
    
    # Correct classes
    "B-LOC": "LOC",
    "O": "O",
    "B-ORG": "ORG",
    "I-ORG": "ORG",
    "B-PERS": "PERS",
    "I-PERS": "PERS",
    "I-LOC": "LOC",
    "I-MISC": "MISC",
    "B-MISC": "MISC",
}

data['NER'] = data['NER'].map(classes_map)
#data.head(10)

#plot to see occurance of NER
data["NER"].value_counts().drop("O", axis=0).plot(kind='bar')

#pretrained GLOVE model   https://nlp.stanford.edu/projects/glove/
filename = 'glove.6B.50d.txt' 
def loadembeddings(filename):
    vocab = []
    embd = []
    file = open(filename,'r',encoding="utf8")
    for line in file.readlines():
        row = line.strip().split(' ')
        vocab.append(row[0])
        embd.append(row[1:])
    print('Word vector embeddings Loaded.')
    file.close()
    return vocab,embd


# Pre-trained word embedding
vocab,embd = loadembeddings(filename)
print(vocab[0],embd[0])


data["Word"] = data["Word"].apply(lambda word: np.array(embd[vocab.index(word)]).astype(np.float16) if word in vocab else np.zeros((50)))
data["Word"].head(10)

#converting classes in label encoding
le = LabelEncoder()
le.fit(data["NER"].unique())

#Applying label encoding to the column
data["NER"] = data["NER"].apply(lambda x: le.transform([[x]])[0])
data["NER"].head(10)

#Now we need to turn data into sequences
def window_stack(X, Y, stride=1, time_steps=3, output_mode=0):
    """Stacks elements in a window and resizes array to be array of
       sequences."""

    # Output_mode defines if it will return sequence of Y or a
    # single Y value corresponding to a sequence of X
    # 0 => Single, 1 => Sequence

    if(len(Y) == 0):
        test = True
    else:
        test = False

    X2 = X[np.arange(0, X.shape[0]-time_steps+1, stride)[:,None] + np.arange(time_steps)]
    
    if(not test):
        Y2 = Y[np.arange(0, Y.shape[0]-time_steps+1, stride)[:,None] + np.arange(time_steps)]
        return (X2, Y2)

    return X2

SEQ_LENGTH = 10

X = data["Word"]
Y = data["NER"]

X = np.array([np.array(item) for item in X.values])
Y = np.array([np.array(item) for item in Y.values])

# Turn the data into sequences
X, Y = window_stack(X, Y, time_steps=SEQ_LENGTH, output_mode=1, stride=SEQ_LENGTH)
X[0], Y[0]

X[0].shape, Y[0].shape


X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.4)
X_Train, X_Val, Y_Train, Y_Val = train_test_split(X_Train, Y_Train, test_size=0.3)
X_Train[0], Y_Train[0]

print("Training set shapes: {} {}".format(X_Train.shape, Y_Train.shape))
print("Validation set shapes: {} {}".format(X_Val.shape, Y_Val.shape))
print("Test set shapes: {} {}".format(X_Test.shape, Y_Test.shape))

X_Train = X_Train.astype(np.float32)
X_Test = X_Test.astype(np.float32)
X_Val = X_Val.astype(np.float32)

train = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(X_Train), tf.data.Dataset.from_tensor_slices(Y_Train)))
test = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(X_Test), tf.data.Dataset.from_tensor_slices(Y_Test)))
val = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(X_Val), tf.data.Dataset.from_tensor_slices(Y_Val)))

train = train.shuffle(buffer_size=1024).batch(64).prefetch(2)
test = test.shuffle(buffer_size=1024).batch(64).prefetch(2)
val = val.shuffle(buffer_size=1024).batch(64).prefetch(2)

with tf.device("/cpu:0"), tf.name_scope("data"):
    iterator = tf.data.Iterator.from_structure(train.output_types,
                                           train.output_shapes)
    train_iterator = iterator.make_initializer(train)
    test_iterator = iterator.make_initializer(test)
    val_iterator = iterator.make_initializer(val)
    
    sequence, labels = iterator.get_next()

#LSTM layer    
with tf.name_scope("LSTM"):
    lstm_cell_forward = tf.nn.rnn_cell.LSTMCell(128, initializer=tf.glorot_uniform_initializer())
    lstm_cell_forward = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_forward, input_keep_prob=0.2, output_keep_prob=0.2, state_keep_prob=0.2)
    
    lstm_cell_backward = tf.nn.rnn_cell.LSTMCell(128, initializer=tf.glorot_uniform_initializer())
    lstm_cell_backward = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_backward, input_keep_prob=0.2, output_keep_prob=0.2, state_keep_prob=0.2)
    
    (forward_seq, backward_seq), _ = tf.nn.bidirectional_dynamic_rnn(
        cell_fw = lstm_cell_forward,
        cell_bw = lstm_cell_backward,
        inputs=sequence,
        dtype=tf.float32
    )
    output = tf.concat([forward_seq, backward_seq], axis=-1)
    
classes = len(data["NER"].unique())
with tf.name_scope("Projection"):
        W = tf.get_variable(name="W",
                            shape=[256, classes],
                            initializer=tf.glorot_uniform_initializer(),
                            dtype=tf.float32)

        b = tf.get_variable(name="b",
                            shape=[classes],
                            initializer=tf.zeros_initializer(),
                            dtype=tf.float32)
        scores = tf.tensordot(output, W, 1) + b

#Loss function
with tf.name_scope("CRF"):
    sequence_lengths = SEQ_LENGTH * tf.ones((tf.shape(sequence)[0]), dtype=tf.int32)
    log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
        scores, labels, sequence_lengths)
    viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(
        scores, transition_params, sequence_lengths)

    loss = tf.reduce_mean(-log_likelihood)

#optimization    
with tf.name_scope("Optimizer"):
    adam = tf.train.AdamOptimizer(learning_rate=0.001)
    optimizer = adam.minimize(loss)

#training phase
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(50):
        sess.run(train_iterator)
        try:
            total_loss = []
            while True:
                _, l = sess.run([optimizer, loss])
                total_loss.append(l)
        except tf.errors.OutOfRangeError as e:
            pass
            print(e)
        # Calculate epoch loss
        total_loss = np.mean(total_loss)
        print("Epoch {}: Loss {} ".format(i, total_loss))
        save_path = saver.save(sess, "./model.ckpt")


