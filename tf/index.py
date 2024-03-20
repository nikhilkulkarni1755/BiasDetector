import tensorflow as tf
import pandas as pd
import numpy as np
# from sklearn.model_selection import train_test_split

df = pd.read_csv("writing_samples.csv")
X = df[df.columns[0]].values
y = df[df.columns[1]].values

# works!
# print(df.head())



# following text tokenization from: 
# https://youtu.be/VtRLrQ3Ev-U?si=uZ4lrOUO57oODvbi&t=5314

# def df_to_dataset(dataframe, shuffle=True, batch_size=32):
#   dataframe = dataframe.copy()
#   labels = dataframe.pop('df')
#   ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
#   if shuffle:
#     ds = ds.shuffle(buffer_size=len(dataframe))
#   ds = ds.batch(batch_size)
#   return ds

train, validation, test = np.split(df.sample(frac=1), [int(.8*len(df)), int(.9*len(df))])

print(train)
print("* * * * *")
print(validation)
print("* * * * *")
print(test)

# train_data = df_to_dataset(train)
# validation_data = df_to_dataset(validation)
# test_data = df_to_dataset(test)

# print(list(train_data)[0])

# X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size = .4, random_state = 0)
# X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size = .5, random_state = 0)
# print('* * * * *')
# print(X_train)
# print(y_train)
# print('* * * * *')

# copied from https://www.tensorflow.org/tutorials/structured_data/feature_columns
# A utility method to create a tf.data dataset from a Pandas Dataframe


encoder = tf.keras.layers.TextVectorization(
    max_tokens=10000,
    output_mode='int',
    output_sequence_length=250
)

train_text = train.map(lambda x, y:x)
encoder.adapt(train_text)

model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(
        input_dim = len(encoder.get_vocabulary()),
        output_dim=32,
        mask_zero=True
    ),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(32, activation='relu'),
    # tf.keras.layers.Dropout(.4),
    tf.keras.layers.Dense(32, activation='relu'),
    # tf.keras.layers.Dropout(.4),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(32, activation='relu'),
#     # tf.keras.layers.Dropout(.4),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=.001),
    loss=tf.keras.losses.BinaryCrossEntropy(),
    metrics=['accuracy']
)

# vocab = np.array(encoder.get_vocabulary())

# print(vocab[:20])




# works!
# print(X)
# print(y)



#isBiased (0 is true, 1 is false)

# test with these sentences

# clearly biased
# "Just met a group of 23 year olds and it was honestly kind of insane"
# "My current working theory is that Charles is dying, which means William would be King, and William did not want Kate to be his Queen (because: commoner? separated? rocky marriage) so HE asked for a divorce and planned to marry Rose Hanbury"
# "I think I’m done with the internet for today"

# clearly not biased
# "Apple just quietly unveiled MM1, a new LLM that competes with GPT-4 and Gemini."
# "In December 1963, Rodney Fox was spearfishing off the southern coast of Australia when he was abruptly pulled under the water."

# some opinion with facts