import pickle

def predict_with_PHOC(loaded_model, phocs):
    loaded_model = pickle.load(open('knn_classifier', 'rb'))
    result = loaded_model.predict(phocs)
    return result
