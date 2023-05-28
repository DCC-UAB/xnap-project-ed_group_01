from build_phoc import phoc
from sklearn.neighbors import KNeighborsClassifier

import pickle
import os


"""annotations_file = "Datasets/lexicon.txt"

with open(annotations_file, "r") as file:
    list_of_words = file.readlines()
"""

img_dir = "C:/Users/xavid/Desktop/Dataset_easy/train_easy/"
list_of_words = os.listdir(img_dir)
list_of_words = [word.split("_")[-1].split(".")[0] for word in list_of_words]
#list_of_words = [l[:-1] for l in list_of_words]
phoc_representations = phoc(list_of_words)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(phoc_representations, list_of_words)

knnPickle = open('xavi_ruc/utils/knn_classifier', 'wb') 
pickle.dump(knn, knnPickle)  
knnPickle.close()




