import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

# Read data from CSV
data = pd.read_csv("writing_samples.csv")

# Extract prompts and labels
prompts = data["Prompt"].tolist()
labels = data["isNotBiased"].tolist()

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(prompts, labels, test_size=0.2, random_state=42)

print(X_train)
print(y_train)

print("* * * * *")

print(X_val)
print(y_val)

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index
train_sequences = tokenizer.texts_to_sequences(X_train)
val_sequences = tokenizer.texts_to_sequences(X_val)
padded_train_sequences = pad_sequences(train_sequences, maxlen=20)
padded_val_sequences = pad_sequences(val_sequences, maxlen=20)

# Define your model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(word_index) + 1, 16, input_length=20),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(.4),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
# model.fit(padded_train_sequences, np.array(y_train), epochs=10, validation_data=(padded_val_sequences, np.array(y_val)))

model.fit(padded_train_sequences, np.array(y_train), epochs=100, validation_data=(padded_val_sequences, np.array(y_val)))

# Evaluate the model
# Evaluate your model on a test set if available

# print(model.evaluate("we would love to learn abour climate"))
# print(model.predict("we would love to learn abour climate"))

# wrong
# new_prompt = "we want to build a bunch of chips"

# new_prompt = "mclaren is one underdog. it will win soon"

# coorect: .56
# new_prompt = "naruto is a show with 500+ episodes"

# new_prompt = "Google is a big company"

# new_prompt = "Dwight is so wholesome"

# new_prompt = "sahil shah is a great comedian!"

new_prompt = "the weather is 70 degrees"


# Tokenize and pad the new prompt
new_sequence = tokenizer.texts_to_sequences([new_prompt])
padded_new_sequence = pad_sequences(new_sequence, maxlen=20)

# Make prediction
prediction = model.predict(padded_new_sequence)

# Print prediction
print(new_prompt)
print(prediction)
if prediction <= .5:
    print("Prediction: Biased")
else:
    print("Prediction: Not Biased")


# Make predictions
# Use your trained model to make predictions on new data