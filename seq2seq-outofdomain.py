# -*- coding: utf-8 -*-
'''
Training from one dataset (i.e. the source domain) but testing on another dataset (i.e. the target domain).
Code is based on Keras' seq2seq example.

'''

from __future__ import print_function
from keras.models import Sequential
from keras.engine.training import slice_X
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent, Embedding, Flatten, Reshape
import numpy as np
from six.moves import range
import nltk
from keras.preprocessing.sequence import pad_sequences
from bleu_simpler import *

class CharacterTable(object):
    '''
    Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    '''
    def __init__(self, chars, maxlen):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        self.maxlen = maxlen

    def encode(self, C, maxlen=None):
        maxlen = maxlen if maxlen else self.maxlen
        X = np.zeros((maxlen, len(self.chars)))
        for i, c in enumerate(C):
            X[i, self.char_indices[c]] = 1
        return X

    def encode2D(self, C, maxlen=None):
        maxlen = maxlen if maxlen else self.maxlen
        X = np.zeros((len(chars)))
        for c in C:
        	ind = self.char_indices[c]
        	X[ind] = 1
        return X

    def decode(self, X, calc_argmax=True):
        if calc_argmax:
            X = X.argmax(axis=-1)
        return ' '.join(self.indices_char[x] for x in X)


class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'
    highlight = '\033[94m'    

# Parameters for the model and dataset
TRAINING_SIZE = 50000
INVERT = True
RNN = recurrent.LSTM
HIDDEN_SIZE = 50
EMBED_SIZE = 50
BATCH_SIZE = 128
LAYERS = 4
EPOCHS = 5000
MAXLEN = 0 # updated when reading dataset
INPUT_DATA_FILE = # source domain (training) data 
TEST_FILE = # target domain (test) data
OUTPUT_FILE= INPUT_DATA_FILE.split('.txt')[0] + '-weights.h5'

# get datasets, inputs and expected outputs
text = open(INPUT_DATA_FILE).read()
print('corpus length:', len(text), 'characters.')
chars = set(text)

inputs = []
outputs = []
token_string = ""
sentence_start_token = "BOS"
sentence_end_token = "EOS"
for line in open(INPUT_DATA_FILE, 'r'):
	part0 = line.split("===")[0]
	part0 = "%s %s %s" % (sentence_start_token, part0, sentence_end_token)
	token_string = token_string + part0 + " "
	part0 = nltk.word_tokenize(part0)
	part1 = line.split("===")[1].replace("\n", "")
	part1 = "%s %s %s" % (sentence_start_token, part1, sentence_end_token)	
	token_string = token_string + part1 + " "	
	part1 = nltk.word_tokenize(part1)	
	inputs.append(part0)
	outputs.append(part1)	
	if len(part0) > MAXLEN:
		MAXLEN = len(part0)
	elif len(part1) > MAXLEN:
		MAXLEN = len(part1)

# collect a separate validation set...

val_in = []
val_out = []
t_string = ''
for l in open(TEST_FILE, 'r'):
	part0 = l.split("===")[0]
	part0 = "%s %s %s" % (sentence_start_token, part0, sentence_end_token)
	t_string = t_string + part0 + " "
	part0 = nltk.word_tokenize(part0)
	part1 = l.split("===")[1].replace("\n", "")
	part1 = "%s %s %s" % (sentence_start_token, part1, sentence_end_token)	
	t_string = t_string + part1 + " "	
	part1 = nltk.word_tokenize(part1)	
	val_in.append(part0)
	val_out.append(part1)
	if len(part0) > MAXLEN:
		MAXLEN = len(part0)
	elif len(part1) > MAXLEN:
		MAXLEN = len(part1)	
		
		
for i, inp in enumerate(inputs):
	while len(inp) < MAXLEN:
		inp.append("PAD")
		inputs[i] = inp
for o, outp in enumerate(outputs):
	while len(outp) < MAXLEN:
		outp.append("PAD")
		outputs[o] = outp


for i1, inp1 in enumerate(val_in):
	while len(inp1) < MAXLEN:
		inp1.append("PAD")
		val_in[i1] = inp1
for o1, outp1 in enumerate(val_out):
	while len(outp1) < MAXLEN:
		outp1.append("PAD")
		val_out[o1] = outp1
		


token_string =" ".join(token_string.split())
t_string =" ".join(t_string.split())
tokens = nltk.word_tokenize(token_string) + nltk.word_tokenize(t_string)
tokens.append("PAD")
tokens.append("BOS")
tokens.append("EOS")
print('corpus length:', len(tokens), 'tokens.')
chars = set(tokens)
print(chars)
ctable = CharacterTable(chars, MAXLEN)
print('Found', len(set(tokens)), 'unique words.')
print("MAXLEN=", MAXLEN)


questions = []
expected = []

questions = inputs
expected = outputs


print('Total number of examples:', len(questions))

print('Vectorisation...')
X = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool)
y = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool)

for i, sentence in enumerate(questions):
    X[i] = ctable.encode(sentence, maxlen=MAXLEN)
for i, sentence in enumerate(expected):
    y[i] = ctable.encode(sentence, maxlen=MAXLEN)

# get validation sets...


val_q = val_in
val_e = val_out

X_val = np.zeros((len(val_q), MAXLEN, len(chars)), dtype=np.bool)
y_val = np.zeros((len(val_q), MAXLEN, len(chars)), dtype=np.bool)

for i1, sentence1 in enumerate(val_q):
    X_val[i1] = ctable.encode(sentence1, maxlen=MAXLEN)
for i1, sentence1 in enumerate(val_e):
    y_val[i1] = ctable.encode(sentence1, maxlen=MAXLEN)


# Shuffle (X, y) in unison as the later parts of X will almost all be larger digits
indices = np.arange(len(y))
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

X_train = X
y_train = y


print(X_train.shape)
print(y_train.shape)


print('Build model...')

model = Sequential()
model.add(RNN(HIDDEN_SIZE, input_shape=(MAXLEN, len(chars))))
model.add(RepeatVector(MAXLEN))
for _ in range(LAYERS):
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))
model.add(TimeDistributed(Dense(len(chars))))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
              
#plot(model, to_file='./../plots/model-chars.png', show_shapes=True)              

semPatterns = getSemPatterns(TEST_FILE)

# Train the model each generation and show predictions against the validation dataset
for iteration in range(1, EPOCHS):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=1,
              validation_data=(X_val, y_val))
    ###
    # Select 10 samples from the validation set at random so we can visualize errors
    for i in range(10):
        ind = np.random.randint(0, len(X_val))
        rowX, rowy = X_val[np.array([ind])], y_val[np.array([ind])]
        preds = model.predict_classes(rowX, verbose=0)
        q = ctable.decode(rowX[0])
        correct = ctable.decode(rowy[0])
        guess = ctable.decode(preds[0], calc_argmax=False)
        print('Q', colors.highlight, q.split(" PAD")[0], colors.close)        
        print('T', colors.highlight, correct.split(" PAD")[0], colors.close)
        print(colors.ok + '☑ ' + colors.close if correct == guess else colors.fail + '☒ ' + colors.close, guess)
        print('---')       
        guess_bleu = guess.split(" PAD")[0].split()
        ref_bleu = correct.split(" PAD")[0].split()
        ref_list = []
        print("BLEU", 3, "score:", getBleu(guess_bleu, [ref_bleu], [0.25, 0.25, 0.25]))
        print("BLEU", 4, "score:", getBleu(guess_bleu, [ref_bleu], [0.25, 0.25, 0.25, 0.25]))        
        print('---')
    json_string = model.to_json()
    model.save_weights(OUTPUT_FILE, overwrite=True)

#json_string = model.to_json()
#model.save_weights('my_model_weights.h5')
