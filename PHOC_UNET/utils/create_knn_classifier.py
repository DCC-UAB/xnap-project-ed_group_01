from utils import phoc
from sklearn.neighbors import KNeighborsClassifier

import pickle


annotations_file = "Datasets/lexicon.txt"

with open(annotations_file, "r") as file:
    list_of_words = file.readlines()

list_of_words = [l[:-1] for l in list_of_words][:40]
phoc_representations = phoc(list_of_words)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(phoc_representations, list_of_words)

knnPickle = open('PHOC/knn_classifier', 'wb') 
pickle.dump(knn, knnPickle)  
knnPickle.close()




