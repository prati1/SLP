import numpy as np
from keras.callbacks import LambdaCallback, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import Adam

#load the text
with open ("alikhit.txt","r",encoding="utf-8") as f:
  corpus=f.read()

#cannot model the data directly  
#mapping of unique char to integers 
char=sorted(list(set(corpus)))

encoding={c:i for i,c in enumerate(char)}
decoding={i:c for i,c in enumerate(char)}

#summarize the dataset
print(encoding)
print(decoding)
#Total vocabulary
print("Unique characters "+str(len(char)))
#total characters in the text
print("Total characters "+str(len(corpus)))
#119162 characters with 74 distinct characters

#defie the training data for the network
#split the book text up into subsequences with fixed length of 60 chars
#allow each char a chance to be learned from 60 characters at a time

#prepare the dataset, convert the characters to integers using our lookup table
sentence_length=60
step=1
sentences=[]
nextchar=[]
#split up the dataset into training data
for i in range (0,len(corpus)-sentence_length,step):
  sentences.append(corpus[i:i+sentence_length])
  nextchar.append(corpus[i+sentence_length])
  
#total training patterns  
print("Train set length "+str(len(nextchar)))
#Finally preparation of training data is done. So, now we need to transform it so th
#it so that it is suitable for use with Keras


#convert output pattern to encoding of 1 and 0. euta matra 1 huncha.
#configure the network to predict the probability of each of 74 unique
#characters in vocabulary rather than trying to force it to predict next char
x=np.zeros((len(sentences),sentence_length,len(char)),dtype=np.bool)
y=np.zeros((len(sentences),len(char)),dtype=np.bool)

for i,sentence in enumerate(sentences):
  for t,character in enumerate(sentence):
    x[i,t,encoding[character]]=1
  y[i,encoding[nextchar[i]]]=1
  
print("Training set(X) shape"+str(x.shape))
print("Training set(Y) shape"+str(y.shape))

#single LSTM layer with 256 memory units.
#The output layer is a dense layer using the softmax activation function
#to output the probability prediction for each of the unique chars
#between 0 and 1
#e ^ (x - max(x)) / sum(e^(x - max(x)): divide each element of X by the sum of all the elements:
#log loss: measures the performance of a classification model where the prediction input is a probability value between 0 and 1.
model=Sequential()
model.add(LSTM(256,input_shape=(sentence_length,len(char)), stateful=True))
model.add(Dense(len(char)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
architecture = model.to_yaml()
with open('model.yaml', 'a') as model_file:
    model_file.write(architecture)

#checkpointing to record all of network weights to file each time
#an improvement in loss is observed at end of epoch
file_path="weights-{epoch:02d}-{loss:.3f}.hdf5"
checkpoint = ModelCheckpoint(file_path, monitor="loss", verbose=1, save_best_only=True, mode="min")
callbacks = [checkpoint]

#Finally, fit our model to the data.
#30 epoch and batch size of 128 patterns
model.fit(x,y,nb_epoch=30,batch_size=128,callbacks=callbacks)
# network loss decreased almost every epoch and I expect the network could benefit from training for many more epochs.

