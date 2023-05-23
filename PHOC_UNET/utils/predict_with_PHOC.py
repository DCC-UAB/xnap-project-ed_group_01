import pickle

def predict_with_PHOC(phocs):
    loaded_model = pickle.load(open('PHOC_UNET/utils/knn_classifier', 'rb'))
    result = loaded_model.predict(phocs)
    return result
