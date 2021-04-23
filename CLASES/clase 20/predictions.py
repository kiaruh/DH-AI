#!/usr/bin/env python3
import numpy as np

import os
import pickle

from sklearn.preprocessing import LabelEncoder
# from keras.models import model_from_json
from tensorflow.keras.models import model_from_json
from keras import backend as K

from config import *
import aws_utils as au

""" Fetch trained models, encoders and tokenizers. Make predictions. """
def fetch_pickle(bucket_name, folder, file_name):
    # TODO Part 2 
    # au.download_pickle_from_s3(bucket_name, folder, file_name)
    # Load normalizer from local
    print(f'Loading {file_name} from local')    
    with open(os.path.join(folder, file_name), 'rb') as f:
        fetched_object = pickle.load(f)

    return fetched_object    

def get_model_and_encoders():
    print('Loading models')
    normalizer = fetch_pickle(BUCKET_NAME, FOLDER, 'normalizer.pkl')
    encoder = fetch_pickle(BUCKET_NAME, FOLDER, 'encoder.pkl')
    model = fetch_pickle(BUCKET_NAME, FOLDER, 'model.pkl')

    return normalizer, encoder, model

def predict(sample: list) -> dict:
    """
        'sample': List of four floats
    """
    # Fetch the model
    normalizer, encoder, model = get_model_and_encoders()

    # TODO Part 3
    # model = get_deep_model()

    # Convert to np.array([[,,,]]) 1x4
    test_data = normalizer.transform(np.array(sample).reshape(1, -1))

    # Arrays of arrays with the class probabilities distribution
    # e.g [[0.25, 0.50, 0.25]]
    y_probabilities = model.predict(test_data)
    print(f'y_probabilities: {y_probabilities}')

    # Get the best class. Array with only one element
    # e.g. [[0.25, 0.50, 0.25]]: [1]
    y_class = y_probabilities.argmax(axis=-1)
    print(f'y_class: {y_class}')
    y_class_decoded = encoder.inverse_transform(y_class)
    print(f'y_class_decoded: {y_class_decoded}')

    predicted_class = {
        'sample': sample,
        'class': y_class_decoded[0],
        'confidence': round(y_probabilities.flatten()[y_class[0]] * 100, 2)
    }

    # TODO Part 3
    # K.clear_session()

    return predicted_class

def get_deep_model():
    print('Loading deep model from disk')

    # Model definition
    file_name = 'model.json'
    au.download_json_from_s3(BUCKET_NAME, FOLDER, file_name)    
    with open(os.path.join(FOLDER, file_name), 'r') as json_file:
        model = model_from_json(json_file.read())
    print('Model definition loaded from disk')

    # Model weights
    file_name = 'model.h5'
    au.download_h5py_from_s3(BUCKET_NAME, FOLDER, file_name)    
    model.load_weights(os.path.join(FOLDER, file_name))
    print("Model weights loaded from disk")

    return model