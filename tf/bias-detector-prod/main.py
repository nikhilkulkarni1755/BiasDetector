import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import sys

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the model
model = tf.keras.models.load_model("bias_detector_v1.keras")

# print("Enter value:")
new_prompt = sys.argv[1]

# Tokenize and pad the new prompt
new_sequence = tokenizer.texts_to_sequences([new_prompt])
padded_new_sequence = pad_sequences(new_sequence, maxlen=20)

# Make prediction
prediction = model.predict(padded_new_sequence)

# Print prediction
# print(new_prompt)
# print(prediction)
if prediction <= .5:
    print("Prediction: Biased")
else:
    print("Prediction: Not Biased")
