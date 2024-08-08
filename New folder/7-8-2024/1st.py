import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#sample data
sentences = [
    "i love my bike!",
    "i had a worst experience in service",
    "it was ok, but nothing special",
    "the service consultant was terrible."
]
#positive = 1,negative =0, neutral = 2
labels = [1,0,2,0]

#tockenization and the padding of sequences

tokenizer = Tokenizer(num_words= 10000, oov_token='<oov>')
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

#sequences
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding='post')

#CREATE THE MODEL FOR ABOVE
model = Sequential(
    [
        Embedding(input_dim=10000, output_dim= 64),
        LSTM(64,return_sequences = False),
        Dropout(0.5),
        Dense(3, activation='softmax')
        
    ]
)

#compilation can be done for the model

model.compile(loss = 'sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#do the training for the model
model.fit(padded,np.array(labels),epochs=50,verbose=1)

#make the prediction

test_sentences = ["i enjoyed the filim"]
test_seq = tokenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(test_seq, maxlen=padded.shape[1])
prediction=model.predict(test_padded)
predicted_class = np.argmax(prediction)

print(f"predicted sentiment of the sentence :{predicted_class}")