import tensorflow as tf
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# load the data
train_file_path = "train-data.tsv"
test_file_path = "valid-data.tsv"

train_dataset = pd.read_csv(train_file_path, delimiter='\t', header=None, names=['label', 'text'])
test_dataset = pd.read_csv(test_file_path, delimiter='\t', header=None, names=['label', 'text'])
print(train_dataset.head())
print(test_dataset.head())

# transform categorical to numerical
y_train = train_dataset['label'].astype('category').cat.codes
y_test = test_dataset['label'].astype('category').cat.codes

# stopwords are common unimportant words like "is, to, at" that don't play that much of a role
# lemmatizer is tool that reducer a word to their root form

# nltk.download('stopwords')  # download stopwords
# nltk.download('wordnet')  # download vocab for lemmatizer

# had to manually download and unzip the files, for whatever reason it fails to do it

# MAKE SURE TO CHANGE THIS TO WHATEVER DIRECTORY YOU DOWNLOAD THE FILES TO
nltk.data.path.append(r'C:\Users\sandr\Desktop\Spam_vs_Ham_SMS_Identifier\nltk_data')
stopwords_eng = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()


def clean_txt(txt):
    txt = re.sub(r'([^\s\w])+', ' ', txt)
    txt = " ".join([lemmatizer.lemmatize(word) for word in txt.split()
                    if word not in stopwords_eng])
    txt = txt.lower()
    return txt


# clean the input and test text
X_train = train_dataset['text'].apply(lambda x: clean_txt(x))

# Keep top 1000 frequently occurring words
max_words = 1000

# Cut off the words after seeing 500 words in each document
max_len = 500

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)

# Transform each text to a sequence of integers
sequences = tokenizer.texts_to_sequences(X_train)
print(sequences[:5])

# pad the sequences with 0's
sequences_padded = pad_sequences(sequences, maxlen=max_len)
print(sequences_padded[:5])

# create and compile the model
model = keras.Sequential([
    Embedding(input_dim=max_words, output_dim=50, input_length=max_len),
    LSTM(64),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='relu'),
])

model.compile(optimizer='RMSprop', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

# train the model
history = model.fit(
    sequences_padded, y_train,
    batch_size=64, epochs=10,
    validation_split=0.2,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00001)],
)


# plot the training info
def plot_loss(history):
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.legend()
    plt.show()


plot_loss(history)


# easier preprocessing for input data
def preprocessing(X):
    x = X.apply(lambda x: clean_txt(x))
    x = tokenizer.texts_to_sequences(x)
    return pad_sequences(x, maxlen=max_len)


# evaluate model on test_data
loss, accuracy = model.evaluate(preprocessing(test_dataset['text']), y_test)
print(f"Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}")


# testing the model

# function to predict messages based on model
# (should return list containing prediction and label, ex. [0.008318834938108921, 'ham'])
def predict_message(pred_text):
    pred_seq = tokenizer.texts_to_sequences(pred_text)
    pred_seq_padded = pad_sequences(pred_seq, maxlen=max_len)

    prediction = model.predict(preprocessing(pd.Series([pred_text])))[0]

    label = 'spam' if prediction[0] > 0.5 else 'ham'

    return [float(prediction[0]), label]


pred_text = "how are you doing today?"

prediction = predict_message(pred_text)
print(prediction)


# Run this cell to test your function and model. Do not modify contents.
def predictions():
    test_messages = ["how are you doing today",
                     "sale today! to stop texts call 98912460324",
                     "i dont want to go. can we try it a different day? available sat",
                     "our new mobile video service is live. just install on your phone to start watching.",
                     "you have won £1000 cash! call to claim your prize.",
                     "i'll bring it tomorrow. don't forget the milk.",
                     "wow, is your arm alright. that happened to me one time too"
                     ]

    test_answers = ["ham", "spam", "ham", "spam", "spam", "ham", "ham"]
    my_answers = []
    passed = True

    for msg, ans in zip(test_messages, test_answers):
        prediction = predict_message(msg)
        my_answers.append(prediction[1])
        if prediction[1] != ans:
            passed = False

    print(test_answers)
    print(my_answers)
    if passed:
        print("You passed the challenge. Great job!")
    else:
        print("You haven't passed yet. Keep trying.")


predictions()
