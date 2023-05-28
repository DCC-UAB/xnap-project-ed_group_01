import pickle

def load_model():
    return pickle.load(open('xavi_ruc/utils/knn_classifier', 'rb'))
def predict_with_PHOC(phocs, model):
    result = model.predict(phocs)
    return result
