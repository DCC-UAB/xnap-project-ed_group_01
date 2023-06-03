import pickle

def load_model():
    return pickle.load(open('/home/alumne/xnap-project-ed_group_01/28_05/PIPELINE/PHOCNET/utils/knn_classifier', 'rb'))
def predict_with_PHOC(phocs, model):
    result = model.predict(phocs)
    return result
