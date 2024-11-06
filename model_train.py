from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense , LSTM , Embedding , GlobalAveragePooling1D
import pandas as pd
import numpy as np
# 28130   323
def model():
    model = Sequential()
    model.add(Embedding(output_dim = 128 , input_dim = 50000))
    model.add(LSTM(64 , return_sequences = True))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(128 , activation = 'relu'))
    model.add(Dense(1 , activation = 'sigmoid'))

    model.compile(loss = 'binary_crossentropy' , optimizer = "Adam" , metrics = ['accuracy'])

    xtrain = pd.read_csv("Data//tokenized//xtrain.csv")
    xtest = pd.read_csv("Data//tokenized//xtest.csv")
    ytrain = pd.read_csv("Data//tokenized//ytrain.csv")
    ytest = pd.read_csv("Data//tokenized//ytest.csv")

    model.fit(np.array(xtrain) , np.array(ytrain['label']) , epochs = 10 , validation_data = (np.array(xtest) , np.array(ytest['label'])))


    # print(model.evaluate(np.array(xtest) , np.array(ytest)))

    model.save("LSTM.h5")

    return "Your Task is completed."


if __name__ == "__main__":
    print(model())