import numpy as np
import tweepy
from keras.models import model_from_yaml
import time

#using twitter API to authenticate access to the twitter account for updating tweets.
auth=tweepy.OAuthHandler("ZgUVeojgqGaHAJcTVUeZwErSQ","QYWOshwDl2sle8W2YPz31XGTGV7884A2ocSenJKL7e3dUN3xpQ")
auth.set_access_token("2666722843-PMHGcuuuyqEerxmetABnP2Y2LEcrO8F8yayhezd","hPxD8gARBT09VyxsbiNxw85suj1PfLWTnJe0Pkb4v6n6l")
api=tweepy.API(auth)

#load the poem in unicode format
with open("laxmipoem.txt", "r",encoding="utf-8") as f:
	corpus = f.read()

#creating a set of distinct characters and mapping them to unique integers
chars = sorted(list(set(corpus)))
encoding = {c: i for i, c in enumerate(chars)}
decoding = {i: c for i, c in enumerate(chars)}
print("loaded encoding and decoding data sets")

sentence_length=60
#loading the network weights
with open ("model.yaml",'r') as f:
    yaml_string=f.read()

model=model_from_yaml(yaml_string)
model.load_weights("weights-30-0.890.hdf5")

#picking a random seed
seed_starting_index = np.random.randint(0, len(corpus) - sentence_length)
seed_sentence = corpus[seed_starting_index:seed_starting_index + sentence_length]
X_predict = np.zeros((1, sentence_length, len(chars)), dtype=np.bool)
for i, char in enumerate(seed_sentence):
    X_predict[0, i, encoding[char]] = 1

#use the Keras LSTM model to make predictions is to first start off with a seed sequence as input
#generate the next character then update the seed sequence to add the generated character on the end and trim off the first character
#repeat the process as long as we want to predict new characters
generated = ""
for i in range(250):
    prediction = np.argmax(model.predict(X_predict))
    generated += decoding[prediction]
    activations = np.zeros((1, 1, len(chars)), dtype=np.bool)
    activations[0, 0, prediction] = 1
    X_predict = np.concatenate((X_predict[:, 1:, :], activations), axis=1)

#api.update_status(seed_sentence + generated)
#print("Posted Tweet")
print(seed_sentence + generated)

import io
with io.open("output.txt", "a", encoding="utf-8") as f:
    f.write(seed_sentence + generated)
