import pickle

import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.neural_network import MLPClassifier

import dataset

img_length = 750


def get_ai(regenerate=False):
    if regenerate:
        return start_training()
    try:
        with open("data.ai", "rb") as f:
            print("Found an existing IA")
            return pickle.load(f)
    except Exception as ex:
        return start_training()


def save_ai(ai):
    try:
        with open("data.ai", "wb") as f:
            pickle.dump(ai, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)


def start_training():
    print("Regenerating IA")
    train_x, train_y = dataset.import_dataset(dataset.Dataset.TRAIN, img_length)
    train_x = dataset.format_dataset(train_x, img_length)
    train_y = np_utils.to_categorical(train_y, 2)

    test_x, test_y = dataset.import_dataset(dataset.Dataset.TEST, img_length)
    test_x = dataset.format_dataset(test_x, img_length)
    test_y = np_utils.to_categorical(test_y, 2)

    #choose your model
    return cnn_model(train_x, train_y, test_x, test_y)


def cnn_model(train_x, train_y, test_x, test_y):
    # building a linear stack of layers with the sequential model
    model = Sequential()
    # hidden layer
    model.add(Dense(100, input_shape=(img_length * img_length,), activation='relu'))
    # output layer
    model.add(Dense(2, activation='softmax'))

    # looking at the model summary
    model.summary()
    # compiling the sequential model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    # training the model for 10 epochs
    model.fit(train_x, train_y, batch_size=128, epochs=10, validation_data=(test_x, test_y))
    score(model, test_x, test_y)

    # Calculer la précision du modèle sur l'ensemble de test
    test_acc = model.evaluate(test_x, test_y)[1]
    print('Précision sur l\'ensemble de test :', test_acc)

    return model

def mlp_model(train_x, train_y, test_x, test_y):
    # building a linear stack of layers with the sequential model
    model = MLPClassifier(hidden_layer_sizes=(300,), verbose=2)

    model.fit(train_x, train_y)

    score(model, test_x, test_y)

    # Calculer la précision du modèle sur l'ensemble de test
    test_acc = model.score(test_x, test_y)
    print('Précision sur l\'ensemble de test :', test_acc)

    return model

def score(model, test_x, test_y):
    test_y_pred = np.argmax(model.predict(test_x), axis=-1)
    test_y_inverse = np.argmax(test_y, axis=1)
    print('L\'ensemble de test :\n', test_y_inverse)
    print('Prédictions sur l\'ensemble de test :\n', test_y_pred)
    # print("\n")
    print('20 premieres données de test :\n', test_y_inverse[:20])
    print('20 premieres données de la prediction de test :\n', test_y_pred[:20])

    # # Calculer la précision du modèle sur l'ensemble de test
    # test_acc = model.evaluate(test_x, test_y)[1]
    # print('Précision sur l\'ensemble de test :', test_acc)
