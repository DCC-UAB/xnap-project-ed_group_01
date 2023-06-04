
## Generació de dataset

Per generar un dataset primer s’ha de modificar el **params.py**:

- type_generate: pots indicar si el nou dataset ha de contenir paraules random (random_words) o paraules creades a partir de lletres random (random_chars). 
- multiclass: pots indicar si vols que els labels siguin multiclasse (True) o d’una única classe (False). Els labels multiclasse són útils per treballar amb un YOLO que detecti caràcters i els labels d’una sola classe són útils per treballar amb un YOLO que detecti i predeixi caràcters alhora.
- folder: carpeta on guardar les imatges.  Per defecte l’has de tenir creada amb el següent format, tot i que al mateix params.py pots canviar l’estructura:
 
  - folder_name
    - labels
      - train
      - test
	- images 
      - train
	  - test

-	n_train_images: número d’imatges de train a crear
-	n_test_images: número d’imatges de test a crear

A continuació ja pots executar:

``` python generate_images/generate_images.py```


## Detection

#### OCR
Per fer predicció dels bounding boxes dels caràcters amb OCR primer s’ha de modificar el **params.py**:
-	ocr_predictions: carpeta on guardar els fitxers .txt de cada imatges amb els seus bboxes.
Ara ja es pot executar l’arxiu Detection/OCR/ocr.py.
Per avaluar aquestes prediccions, executar:

``` python Detection/OCR/evaluate.py``` 

#### YOLO
Per entrenar el YOLO per fer només detecció de caràcters primer s’ha de canviar el **params.py**.
-	data_yaml_detection: carpeta pròpia on s’executa el fitxer i es guardarà el yaml amb les dades d’entrenament
-	path_yolo_detection: carpeta d’on yolo agafa les imatges i labels. El format requerit és el mateix que especifiquem a l’hora de generar el dataset
Després s’ha d’executar:

``` python Detection/YOLO/yolo8.py```

Es genera un carpeta de sortida runs a l’arrel de la carpeta on guarda els diferents entrenament. S’hi guarden diferents resultats, mètriques i el model en qüestió. 

## Recognition

#### CNN
Per entrenar els models basats en CNN primer has de modificar el **params.py**.
-	saved_model_cnn: carpeta on vols que se’t guardi el model amb els pesos al final de cada epoch
A posteriori s’ha d’executar:

``` python Recognition/CNN/main.py``` 

En aquest fitxer main.py pots definir quin model crear (“vggnet”, “resnet” o “googlenet”) i canviar hiperparàmetres. A l’inici fa un registre a wandb on es pujaran els diferents losses i accuracies.

## Pipeline Detection + Recognition

#### OCR + CNN
Per fer la validació de la pipeline de OCR + CNN amb el CNN ja entrenat  primer s’han de fer canvis en el **params.py**:
-	dataset: dataset a on avaluar
-	cnn_model: nom del model CNN guardat
-	model_cnn_entrenat: carpeta on es guarden els models CNN entrenats
-	ocr_cnn_store_files: carpeta on es guarden les prediccions
Després s’ha d’executar el fitxer 

``` python PIPELINE/OCR_CNN/pipeline.py ```

Genera les matrius de confusió pertinents i printeja l’accuracy i l’edit distancce. 

#### YOLO + CNN
Per fer la validació de la pipeline de YOLO + CNN amb el YOLO i CNN ja entrenats primer s’han de fer canvis en el **params.py**:
-	dataset: dataset a on avaluar
-	cnn_model: nom del model CNN guardat
-	model_cnn_entrenat: carpeta on es guarden els models CNN entrenats
-	yolo_model: nom del model YOLO guardat
-	model_yolo_entrenat:  carpeta on es guarda el model YOLO entrenat
-	yolo_cnn_store_files: carpeta on es guarden les prediccions
Després s’ha d’executar: 

``` python PIPELINE/**YOLO_CNN/pipeline.py```

Genera les matrius de confusió pertinents i printeja l’accuracy i l’edit distancce. 

#### YOLO
Per entrenar el YOLO per fer detecció i reconeixement de caràcters primer s’ha de modificar el **params.py**:
-	data_yaml_detection: carpeta pròpia on s’executa el fitxer i es guardarà el yaml amb les dades d’entrenament
-	path_yolo_detection: carpeta d’on yolo agafa les imatges i labels. El format requerit és el mateix que especifiquem a l’hora de generar el dataset
Després s’ha d’executar:

``` python PIPELINE/YOLO/yolo8.py ``` 

Genera un carpeta de sortida runs a l’arrel de la carpeta on guarda els diferents entrenament. S’hi guarden diferents resultats, mètriques i el model en qüestió. 
Per avaluar aquest YOLO, s’ha de modificar el fitxer **params.py**:
-	yolo_pipeline_model: nom del model YOLO guardat
-	yolo_entrenat_recog_detect fitxer: carpeta on es guarda el model YOLO entrenat
-	yolo_pipeline_store_files: carpeta on es guarden les prediccions
Després s’ha d’executar: 

``` python PIPELINE/YOLO/pipeline.py```

#### PHOCNET
Per tal d’entrenar i validar alhora la PHOCNET primer has de modificar el fitxer **params.py**.
-	saved_model_phocnet: carpeta on guardar el model entrenat a cada iteració
-	store_knn_classifier: fitxer que conté un pickle del knn classifier
-	lexicon_file = fitxer que guarda les 88000 paraules en anglès
-	bigrams_file: fitxer que guarda els bigrams i trigrams més utilitzats en anglès
Primer has d’executar:

``` python PIPELINE\PHOCNET\utils\knn_classifierpy``` 

Es crea l’objecte knn de sklearn per tal de poder predir les paraules a partir d’una representació PHOC.
A continuació ja es pot executar:

``` python PIPELINE\PHOCNET\main.py``` 

En aquest fitxer es poden especificar els hiperparàmetres. Està programat perquè les imatges d’entrada tinguin un canal.
