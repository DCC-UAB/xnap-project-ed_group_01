
# Cropped Word Recognition
En aquest repositori hi trobareu com fer Cropped Word Recognition amb diferents mètodes:
- OCR + CNN
- YOLO + CNN
- YOLO
- PHOCNET
- CRNN

![alt text](https://imgur.com/oDqKDe1.png) 

A continaució s'hi exposarà l'estructura de codi, la generació de datasets i les proves amb els diferents models.

## Estructura
```
├───Datasets
├───Detection
│   ├───OCR
│   └───YOLO
│       └───models_entrenats
├───generate_images
│   ├───fonts
├───PIPELINE
│   ├───OCR_CNN
│   ├───PHOCNET
│   │   ├───models
│   │   ├───models_entrenats
│   │   ├───utils
│   ├───YOLO
│   │   └───models_entrenats
│   └───YOLO_CNN
├───Recognition
│   ├───saved_model
```

## Execució del codi
### Generació de dataset

Per generar un dataset primer s’ha de modificar el **params.py**:

- type_generate: pots indicar si el nou dataset ha de contenir paraules random (random_words) o paraules creades a partir de lletres random (random_chars). 
- multiclass: pots indicar si vols que els labels siguin multiclasse (True) o d’una única classe (False). Els labels s'una sola classe són útils per treballar amb un YOLO que detecti caràcters i els labels de multiclasse són útils per treballar amb un YOLO que detecti i predeixi caràcters alhora.
- folder: carpeta on guardar les imatges. Per defecte l’has de tenir creada amb el següent format, tot i que al mateix params.py pots canviar l’estructura:
 
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


### Detection

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

### Recognition

#### CNN
Per entrenar els models basats en CNN primer has de modificar el **params.py**.
-	saved_model_cnn: carpeta on vols que se’t guardi el model amb els pesos al final de cada epoch
A posteriori s’ha d’executar:

``` python Recognition/CNN/main.py``` 

En aquest fitxer main.py pots definir quin model crear (“vggnet”, “resnet” o “googlenet”) i canviar hiperparàmetres. A l’inici fa un registre a wandb on es pujaran els diferents losses i accuracies.

### Pipeline Detection + Recognition

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

``` python PIPELINE/YOLO_CNN/pipeline.py```

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

Es crea l’objecte knn de sklearn per tal de poder predir les paraules a partir d’una representació PHOC. A continuació ja es pot executar:

``` python PIPELINE\PHOCNET\main.py``` 

En aquest fitxer es poden especificar els hiperparàmetres. Està programat perquè les imatges d’entrada tinguin un canal.

## Baseline
Aquest projecte parteix d'una baseline com a punt de partida per al desenvolupament de la tasca. Aquesta es basa en una arquitectura anomenada PHOCNet (Pyramidal Histogram Of Characters Network), un model de xarxa neuronal dissenyat específicament per tasques de reconeixement de paraules.

### Datasets
* **IIIT 5K**   
  El primer dataset proporcionat el formen imatges resultants de cerques concretes a Google Images, alguns exemples d'aquestes cerques serien: billboards, signboard, house numbers, movie posters... entre altres.
  Conté un total de 5000 imatges en color.
![alt text](https://imgur.com/go4bJYv.png) 
    
* **VGG** 
  Aquest segon dataset, a diferència de l'anterior, està generat de manera sintètica i no té color. Està format per diferents fonts i consta de 9 milions d'imatges que contenen 90.000 paraules en anglès.
![alt text](https://imgur.com/A7jEd6s.png) 

* **Dataset senzill propi** 
  Per últim, s'ha creat un nou dataset per dur a terme diferents proves. Aquest és també sense color, fons blanc i lletra en negre, amb fonts bàsiques, llegibles i amb la possibilitat de generar tantes imatges com es desitgi, amb un màxim al voltant dels 90000 que és el nombre de paraules diferents al diccionari que es fa servir.
![alt text](https://imgur.com/1dlWdGa.png) 

### PHOC
#### Arquitectra PHOCNet
L'arquitectura del model del que partim es mostra en aquesta primera imatge. Com es pot observar, consta d'un seguit de convolucions intercal·lades amb un parell de capes de max pooling, aquestes últimes, posteriors a la primera i segona convolució respectivament. Seguidament s'hi pot trobar una capa de Spatial Pyramid Pooling, aquesta s'afegeix perquè permet a les CNNs rebre com a entrada imatges de diferents mides i, tot i això, resultar en un output de mida constant. El motiu en són les fully connected layers, que han de rebre com a entrada sempre la mateixa mida.
Per acabar, l'última capa de l'arquitectura consta de 604 neurones, que és la mida de PHOC, la representació de les paraules que volem aconseguir i que dona nom al model.
![alt text](https://imgur.com/unjt2Zn.png)


#### Representació PHOC
Amb l'objectiu de tenir una representació de mida fixa per totes les paraules i que contingui informació dels caràcters que la fromen. En aquesta s'hi consideren diferents nivells, el primer consisteix en un vector amb tantes posicions com possibles caràcters es vulguin predir, en aquest cas, tantes com lletres té l'abecedari anglès i deu posicions més pels digits del 0 al 9. Totes s'inicialitzen a 0 i són només les posicions corresponents als caràcters que apareixen en la paraula les que deixaran de tenir un 0 per tenir-hi un 1. Fins aquí el primer nivell, el següent està format per dos vectors com l'anterior i dividint la paraula en dos, els caràcters que apareguin en la primera meitat d'aquesta, tindran un 1 en les posicions corresponents del primer vector i en el segon, tindran un 1 les posicions d'aquells caràcters que apareguin en la segona meitat de la paraula. En el cas del tercer nivell, es divideix la paraula en 3 parts i cadascuna d'aquestes es representa en un vector diferent, de manera que aquest nivell el formen 3 vectors. Pels nivells següents es seguiria aquest patró. Per una millor comprensió es pot consultar la figura següent , les posicions blaves correspondrien al valor 1 i la resta al 0, a més faltarien les 10 posicions dels dígits, addicionalment L1, L2 i L3 corresponen als nivells 1, 2 i 3, respectivament. 
![alt text](https://imgur.com/1zMrnXF.png) 
	
Aquesta és l'essència de la representació PHOC, partint d'aquí, les 604 posicions que s'esmentaven anteriorment les formen la concatenació dels nivells 2, 3, 4 i 5 juntament amb bigrams i trigrams. Bigrams i trograms és un vector que parteix del mateix concepte, però cada posició enlloc de representar un únic caràcter representa una combinació de dos o tres caràcters. Per no agafar totes les possibles combinacions, que en són moltes, s'utilitzen només les més comuns de la llengua anglesa i se'n fa dos nivells.


### Implementació
Pel que fa a la implentació, el codi original del que es parteix està desenvolupat en Caffe i s'ha optat per passar-ho a Pytorch. Un cop el codi en pytorch és funcional i donada una imatge es pot obtenir la seva representació PHOC, d'aquesta se n'ha d'obtenir la paraula a la que correspon. Per fer-ho s'ha utilitzat un KNN amb la representació PHOC precalculada de 90000 paraules, de manera que per passar d'imatge a text el procés consisteix en: primer passar la imatge per la PHOCNet, obtenir així la representació, aquesta passar-la al KNN i que ens retorni d'aquestes 90000 paraules quina és la més semblant a la obtinguda a partir de la imatge.

### Anàlisi de resultats
#### Primers resultats
Aquest primer model s'entrena canviant diferents hiperparàmetres, es van fer les següents proves:
* **Loss**: BCEwithlogits
* **Schedulers**: StepLR, ReduceLROnPlateau, CosineAnnealingLR
* **Optimizers**: SGD amb momentum, Adam amb l'scheduler ReduceLROnPlateau

Malgrat les diverses proves els resultats acaben essent sempre semblants, el model apren a predir tot zeros en la representació de totes les paraules i només hi afegeix uns en els caràcters més freqüents. Com s'ha vist aquesta representació està formada per molts zeros és per això que amb aquestes prediccions la loss baixa i el model s'estanca en aquest minim local del que tant costa sortir. En la pròxima figura es pot veure un exemple d'aquestes prediccions amb el dataset VGG, les imatges d'aquest hi apareixen normalitzades.
![alt text](https://imgur.com/cCy64lx.png) 

#### Resultats finals
Veient els primers resultats i com els canvis en els diferents hiperparàmetres no  donaven resultats, les prediccions no milloraven, es va optar per crear un nou dataset molt més senzill per comprovar que el model fos capaç de dur a terme la seva tasca. Els detalls d'aquest dataset estan especificats en l'apartat Datasets, en la subsecció Baseline. Altrament, va sorgir la idea de canviar els pesos de la loss per canviar la manera de penalitzar els errors i forçar al model a predir més uns. Aquests weights es van definir de manera que el pes de cometre un error al predir un caràcter era l'invers de freqüència d'aparició del mateix en les 90000 paraules del diccionari que es fa servir.
Finalment amb aquest parell de canvis, amb BCEloss, Adam optimizer i ReduceLROnPlateu per ajustar el learning rate, els resultats obtinguts són remarcablement bons. S'observa com la loss no s'estanca, l'accuracy puja fins a arribar pràcticament a 1 i l'edit distance baixa a tocar de 0. Aquesta última és una altra mesura que s'utilitza per avaluar distància entre paraules de mida variables. La finalitat n'és que, a diferència de l'accuracy, si el model prediu de manera errònia només alguns caràcters en certes paraules, no es tenen en compte com prediccions incorrectes envers les correctes on s'encerten tots els caràcters, sinó que se'ls asigna un valor continu en funció de com de diferents són la predicció  i la paraula predir. A continuació les visulaitzacions dels resultats junatment amb una mostra del dataset senzill predita correctament.
![alt text](https://imgur.com/aY2np5A.png) 

## Primer Approach

Vista la baseline, passarem al nostre primer enfoc per intentar resoldre el problema de reconeixement de paraules. Aquest enfoc consta de dos passos principals: detecció i reconeixement.

![Procés de detecció i reconeixement d'imatge](https://i.imgur.com/M203GTn.png)

El procés de detecció consisteix en localitzar i extreure els caràcters presents en la nostra imatge. Utilitzarem tècniques de detecció d'elements per identificar i delimitar cadascun dels caràcters presents a la imatge. Això ens permetrà tractar cada caràcter de manera individual i independent durant el procés de reconeixement. És important tenir en compte que amb aquest enfoc no estem tenint en compte el context de la paraula completa, ja que tractem cada caràcter com una entitat independent.

Un cop hem detectat els caràcters de la imatge, pasarem al procés de reconeixement, on mitjançant diferents models de CNN intentarem entrenar-los per reconèixer i assignar l'etiqueta correcta a cada caràcter individual.

### Detection

Comencarem abordant el problema de la detecció dels caràcters de la paraula amb la que estiguem treballant. Explorarem dos metodologies principals per dur a terme la detecció: la primera consisteix en utilitzar un sistema de reconeixement òptic de caràcters (OCR) mitjançant tècniques de visió per computador, i la segona es basa en el model YOLO (You Only Look Once), més concretament el YOLOv8, que és actualment un dels millors en el camp de la detecció. 

#### OCR

La detecció basada en OCR té com objectiu identificar els diferents caràcters presents en una imatge mitjançant una detecció i delimitació precisa dels seus contorns. 

Per poder fer això fem un preprocessat a la nostra imatge binaritzant-la. Un cop binaritzada la imatge identifiquem els contorns dels diferents elements que apareixen en aquesta i els utilitzem per delimitar cada caràcter de forma individual. D'aquesta forma, es calculen les bounding boxes al voltant de cada contorn, cosa que permet definir els límits rectangulars que tanquen cada lletra. 

![Detecció basada en OCR](https://i.imgur.com/XCB9zZX.png)

#### YOLO

Apart del OCR, com hem explicar, la segona metodologia utilitzada per poder fer detecció de caràceters ha sigut la de utilitzar YOLO, un model de *object detection* ampliament reconegut i utilitzant en el camp de la visió per computador.

YOLO intenta abordar el problema de detecció d'elements com un problema de regressió en el què s'intenta determinar/predir directament les bounding boxes, les classes dels objectes i si en un punt determinat hi ha un objecte o és part del *background*, tot en una sola passada.
![Arquitectura de YOLO](https://lilianweng.github.io/posts/2018-12-27-object-recognition-part-4/yolo-network-architecture.png)
L'arquitectura de YOLO es composa de dos parts principals: el *backbone* i el *head*. El *backbone* està compost per un conjunt de CNNs que s'encarreguen d'extreure les característiques i representacions visuals significatives de la imatge d'entrada. Aquest procés és realitzat a través de múltiples capes convulacionals i de *pooling* que captures gradualment característiques de diferents nivells d'abstracció. Per l'altre costat, el *head* és la part final de la xarxa que duu a terme la predicció de les bounding boxes, classes i ens dona també la probabilitat de presencia d'objecte. Aquesta consisteix d'un seguit de capes lineals (MLPs) que es conecten al final de la *backbone* i s'encarreguen de realitzar la predicció.

En el nostre cas, pel projecte hem decidit fet ús de la vuitena versió de YOLO, el YOLOv8, que és la versió més recent d'aquest model.

#### Comparació
Un cop vist les dos metedologies utilitzades per poder dur a terme la detecció de caràceters d'una paraula, en aquesta secció realizarem una comparació entre aquestes dos. 
Per poder fer això utilitzem el conjunt de dades de IIIT, ja que és l'únic conjunt de dades proporcionat a la *baseline* que conté informació sobre les bounding boxes dels caràcters, cosa que ens permet avaluar tant l'OCR com el YOLO. A més, generem un conjunt de dades propi (següent figura) de 50K imatges de train i 5K imatges de test, que inclou lletres amb diferents fonts, lletres, colors, etc. i també conté informació sobre les bounding boxes. Això ens proporciona un altre conjunt de dades amb què podem avaluar el rendiment d'OCR i YOLO en condicions més diverses.

![El nostre datset propi](https://i.imgur.com/dg6VqEH.png)

Per poder dur a terme la comparació hem fet ús de dos mètriques principals: la precisió i el recall. La precisió es refereix a la proporció de *true positives* (caràcters correctament detectats) sobre el nombre total de deteccions realitzades. Per l'altre costat, el recall es refereix a la proporció de *true positives* sobre el nombre total de caràcters presents a les imatges. Per considerar si una detecció és *true positive* o no fem ús de la IoU (intersection over union), una mètrica que avalua la precisió de les deteccions comparant la superposició entre les bounding boxes detectades i les correctes (*groundtruth*). Un IoU vol dir que hi ha una coincidència més gran entre les deteccions i les bounding boxes reals.
Per tant, el què acabem fent és avaluar la IoU entre la bounding box detectada i la bounding box real, mirant si és més gran que un llindar definit. Això ens permet avaluar la qualitat de les deteccions en funció de la superposició amb les bounding boxes de referència.

A continuació podem veure una taula amb els resultats obtinguts al dur a terme una comparació entre el OCR i el YOLO per la detecció de caràcters per tant el datset de IIIT com el nostre propi dataset que hem generat.

#### Dataset propi

|     Mètode     |  Precision  |  Recall  |
| :------------: | :---------: | :------: |
|      OCR       |    0.869     |   0.976   |
|     YOLO       |    **0.998**     |   **0.996**   |

En el nostre dataset propi podem observar que el YOLO mostra millors resultats, tant per la precisió com per la recall en comparació al OCR. A partir d'aquests resultats podem afirmar que el mètode de YOLO ofereix una millor capacitat de detecció de caràcters amb més exactitud, aconseguint deteccions més precises i recuperant un nombre més gran de caràcters de forma correcta en comparació amb el mètode d'OCR.

#### Dataset IIIT

|     Mètode     |  Precision  |  Recall  |
| :------------: | :---------: | :------: |
|      OCR       |    0.672     |   **0.984**   |
|     YOLO       |    **0.936**     |   0.918   |

Pel datset de IIIT, els results varien lleugerament, a més a diferència del YOLO on la la precisió i el recall estan més equilibrats; l'OCR mostra una alta recall però una baixa precisió. Això és indicatiu de què l'OCR tendeix a sobredetectar en aquest datset, és a dir, identifica molts més objectes dels que hi ha present (com veiem en la següent figura), a causa d'això, tot i que és capaç de recuperar la majoria de les bounding boxes (recall alta), també retorna una gran quantitat de *false positives* (falses deteccions, detectant bounding boxes on en realitat no n'hi ha), donant lloc a una baixa precisió.

![Sobredetecció OCR](https://i.imgur.com/fWmfMqH.png)

Per l'altre costat, amb el model del YOLO veiem una precisió més gran en comparació amb OCR, cosa que indica que les seves deteccions són molt més precises en termes d'identificar correctament els objectes presents a l'imatge (caràcters). A més, el seu recall també és relativament alt, cosa que significa que aconsegueix detectar la majoria dels objectes presents a les diferents imatges sense sobredetectar com a l'OCR (donant lloc a falses deteccions).

En general, tots aquests resultats, tant pel dataset de IIIT com pel nostre propi, donen suport a la idea que YOLO és més efectiu en la detecció precisa de caràcters per diferents escenaris, fonts i condicions en comparació amb OCR.

### Recognition

Vistes les diferents metodologies per poder dur a terme la part de detecció dels caràceters del nostre model, ara ens centrarem en la part de reconaixement d'aquests caràcters. Per poder fer això, experimentarem amb diferents arquitectures de xarxes convulacionals (CNNs), les entrenarem, i avaluarem la seva capacitat d'assignar l'etiqueta correcta a cada caràcter. Més especificament, l'entrenament consisteix en un fine-tuning de diferents models, on importem els paràmetres d'aquests amb els seus pesos (és a dir tot el pre-trained model sencer) i entrenem tot el model de nou per la nostra tasca de classificació (fent que aquests pesos siguin el punt de partida del entrenament).

Per poder dur a terme aquest entrenament, un altre cop farem ús dels dos datasets dels que ja hem parlat en l'apartat anterior: el dataset de IIIT (que només conté 5K samples) i el dataset custom generat per nosaltres (que conté 55K samples). Per poder dur a terme l'entrenament, farem ús de la informació sobre les bounding boxes que tenim en els dos datasets mencionats; aquesta informació ens permetrà extreure, retallar i seleccionar els caràcters individuals de cada imatge per a l'entrenament dels models (següent figura). L'objectiu és que el model convulacional aprengui a reconèixer i assignar les etiquetes correctes a cada caràcter individual a partir d'aquestes imatges retallades. 


![Entrenament per reconeixement de caràceters](https://i.imgur.com/P9ufVvK.png)

#### Arquitectures utilitzades

En aquesta secció parlarem de les tres arquitectures de xarxes neuronals convolucionals (CNNs) amb què hem experimentat per al reconeixement de caràcters: ResNet18, VGG11 i GoogleNet (Inception_v3). A continuació aprofundirem una mica més sobre cada arquitectura i seguidament explicarem l'entrenament i el *hyperparameter tuning* de cadascuna.

##### 1. Resnet18

![Arquitectura ResNet-18](https://i.imgur.com/MlTuRpO.png)

La Resnet18 es tracta d'una arquitectura de xarxa convolucional amb 18 capes de profunditat composta per *layers* convolucionals i blocs residuals. Fa ús de filtres de tamany reduit de 3x3 per capturar característiques visuals de les imatges i consta d'una estructura relativament baixa en termes de profunditat.

##### 2. VGG11

![Arquitectura VGG-11](https://i.imgur.com/sn9LbYf.png)

La VGG11 és una xarxa CNN composta per capes convolucionals seguides de capes de pooling, seguides finalment d'un conjunt de *fully connected layers*. Utilitza filtres de mida petita (3x3) en totes les capes convolucionals, mitjançant els quals captura característiques visuals a les imatges. 

##### 3. GoggleNet

![Arquitectura Inception_v3](https://i.imgur.com/zChawpt.png)

La GoogleNet és una arquitectura CNN que es basa en la idea dels moduls d'Inception, que fan ús de filtres de diferents mides (1x1, 3x3, 5x5) en paral·lel per poder capturar característiques visuals a diferents escales a partir de les imatges. Aquests mòduls s'ajunten en profunditat per permetre l'extracció de característiques tant locals com globals de les imatges. GoogleNet també fa ús de tècniques de regularització i factorització de les capes de convolucions per així millorar el rendiment i reduir el nombre de paràmetres.

#### Hyperparameter tuning

Aquesta secció estarà dedicada a la cerca i ajust de hiperparàmetres per les diferents arquitectures CNN que hem utilitzat, ResNet18, VGG11 i GoogleNet. L'objectiu és intentar trobar la combinació òptima d'hiperparàmetres que maximitzi el rendiment de les nostres xarxes. Començarem entrenant les xarxes amb els seus paràmetres *default* i en funció de les nostres necessitats experimentarem amb el learning rate, els diferents schedulers, weight decay, etc.

##### 1. Resnet18

| Dataset        | Learning Rate | Train Loss | Train Accuracy | Test Loss | Test Accuracy |
| -------------- | ------------- | ----------------- | --------------------- | ---------------- | -------------------- |
| IIIT Dataset   | 0.01           | 0.02345            | 0.9765                | 0.1234           | 0.8543               |
| IIIT Dataset   | 0.001          | 0.0119            | 0.9984               | 0.0798           | 0.8781               |
| Our Dataset| 0.01           | 0.02567            | 0.9223                | 0.01456           | 0.9565               |
| Our Dataset| 0.001          | 0.0079            | 0.9961                | 0.0044           | 0.9977               |

Per la xarxa de ResNet18 l'hiperparàmetre principal amb el qual vam exprimentar va ser el learning rate, ja que amb una lleugera variació d'aquest ja vam a arribar a uns resultats satisfactoris. Vam iniciar entrenant la xarxa amb un learning rate de 0.01, però al veure que amb això sovint la loss començava a fluctuar molt i no acabava de convergir del tot vam decidir reduir-ho a 0.01 amb el què vam veure certa millora. Pel dataset nostre vam arribar a tenir una accuracy de casi el 100% tant pel train com test. En canvi pel IIIT, tot i què la train accuracy va arribar a ser de 0.9984, pel test no vam acabar d'aconseguir pujar-la més que un 0.8781.

##### 2. VGG11

| Dataset        | Hyperparameters                               | Train Loss | Train Accuracy | Test Loss | Test Accuracy |
| -------------- | --------------------------------------------- | ----------------- | --------------------- | ---------------- | -------------------- |
| IIIT Dataset   | Learning Rate: 0.01                           | 0.2345            | 0.7965                | 0.2234           | 0.7943               |
| IIIT Dataset   | Learning Rate: 0.01 + Weight Decay            | 0.1965            | 0.8587                | 0.2153           | 0.8134               |
| IIIT Dataset   | Learning Rate: 0.01 + Weight Decay +  ReduceLROnPlateau         | 0.1598            | 0.9123                | 0.2108           | 0.8219               |
| IIIT Dataset   | Learning Rate: 0.01 + Weight Decay + StepLR        | 0.1473            | 0.9227                | 0.1838           | 0.8461               |
| Our Dataset| Learning Rate: 0.01                           | 0.1234            | 0.9486                | 0.1098           | 0.9543               |
| Our Dataset| Learning Rate: 0.01 + Weight Decay            | 0.0974            | 0.9836                | 0.0884           | 0.9789               |

Per la xarxa de VGG11 hem experimentat amb diferents schedulers i el weight decay, especialment pel dataset d'IIT on hem vist que era amb el què el model tenia més dificultats per convergir, doncs sovint es quedava estancat i la loss no acavaba de baixa del tot. Per aquesta mateixa raó vam iniciar afegit weight decay, el qual tot i que ens va mostrar certa millora no era la desitjada. Seguidament vam provar amb dos diferents schedulers els quals també van ajudar a millorar els resultats, sent el StepLR el qual millors mèetriques ens va donar. Respecte el nostre dataset desde un principi amb el paràmetres per *dafault* ja veiem bons resultats, el qual hem acabat de millorar al afegir-ho weight decay durant l'entrenament.

##### 3. GoogleNet

| Dataset        | Learning Rate | Train Loss | Train Accuracy | Test Loss | Test Accuracy |
| -------------- | ------------- | ----------------- | --------------------- | ---------------- | -------------------- |
| IIIT Dataset   | 0.01           | 0.5698            | 0.8897                | 0.6639           | 0.8543               |
| IIIT Dataset   | 0.001          | 0.0727            | 0.9900               | 0.6193           | 0.8689               |
| Our Dataset| 0.01           | 0.1198            | 0.9523                | 0.1261           | 0.9465               |
| Our Dataset| 0.001          | 0.0897            | 0.9717                | 0.1067           | 0.9742               |

Finalment, per la xarxa de Inception_V3 (GoogleNet), un altre cop, l'hiperparàmetre principal amb el qual vam exprimentar va ser el learning rate, ja que amb una lleugera variació d'aquest ja vam a arribar a uns resultats satisfactoris. Al iniciar provant amb un learning rate "elevat" de 0.01 vam notar un entrenament força inestable, pel qual vam decidir de baixar el learning rate a 0.001, pel qual vam obtenir millores tant amb el dataset d'IIIT com pel nostre propi. Veiem com pel nostre dataset arribem a unes accuracies superiors a 0.97, resultant en prediccions casi perfectes. Per l'altre costat al model (i a tots amb els quals hem exeprimentat en general) li costa una mica més treballar amb el dataset d'IIIT, on tot i que la train accuracy sigui casi perfecte, la test accuracy no l'hem aconseguit baixar del 0.8689.
La raó d'això últim pot ser en gran mesura la mida del dataset d'IIIT que és de tan sols 5K samples, el qual pot fer que sigui molt més difícil aprendre i generalitzar pel test set, donant lloc així al overfitting.

#### Comparativa entre els diferents models

Vista la cerca d'hiperparàmetre per cada una de les diferents arquitectures amb les quals hem experimentat, ara mostrarem els resultats de cadascun d'ells pels millors hiperparàmetre en cada dataset per així poder dur a terme una comparació i decidir quin dels models funciona millor per cada dataset. Aquest serà el model que acabarem utilitzant en la nostra pipeline final que consisitirà d'un mètode de detecció (OCR i/o YOLO) i un de reconaixement (el millor model de tots el que hem provat).

##### Pel nostre dataset:
| Model    | Train Acc | Test Acc  |
|-----------|-----------|-----------|
| ResNet18    |   **0.9961**   |   **0.9977**   |
| VGG11       |   0.9836   |   0.9789   |
| GoogleNet |   0.9717   |   0.9742   |

##### Pel dataset d'IIIT:

| Model    | Train Acc | Test Acc  |
|-----------|-----------|-----------|
| ResNet18    |   **0.9984**   |   **0.8781**   |
| VGG11       |   0.9227   |   0.8461   |
| GoogleNet |   0.9900   |   0.8689   |

Tant pel nostre dataset propi com pel d'IIIT el model basat en RenNet18 mostre un rendiment superior en termes d'accuracy tant pel *train set* com pel *test set*. Això ens indica que la ResNet18 és capaç d'aprendre i generalitzar millor les característiques dels caràcters presents als diferents datasets, resultant en una millor accuracy. Aquests resultats donen suport a la nostra elecció d'utilitzar la ResNet18 com el model per la tasca de reconeixement dels caràcters pels dos datasets.

### Detection + Recognition

En aquest apartat compararem la combinació dels mètodes de detecció que hem explorat anteriorment amb el millor model de reconeixement (ResNet18) que hem entrenat per cada dataset (el nostre propi i el IIIT). Aquesta combinació resultarà en una pipeline que ens permetrà dur a terme la tasca de *word recognition* que se'ns ha proporcionat. Com ja hem explicat a l'introducció del apartat, la pipeline consistirà d'un mètode de detecció per detectar els caràcters i un model de reconeixement entrenar per reconeixer les lletres detectades.

Per poder fer la comparativa les mètriques que utilitzarem seràn l'edit distance (que ja hem explicat anteriorment, és una manera de calcular la distància d'edició entre paraules de diferet mida, ens permetrà saber quant s'allunya la nostra predicció de la paraula correcta a predir), accuracy (si la praula predita és correcta) i accuracy amb diferents nivells de toleràcia. Respect a això últim, estarem avaluant l'accuracy del sistema tenint en compte que es considera correcta la paraula fins i tot si es produeix un error en una lletra; i també considerarem si s'accepta com a correcte si es produeixen fins a dos errors a les lletres. Aquestes mètriques ens permetran tenir una visió més completa del rendiment i la capacitat del sistema per dur a terme la tasca de *word recognition* amb els diferents datasets.

Pel dataset nostre obtenim els següents resultats:

| Mètode           | Edit Distance | Accuracy | Accuracy (fallant 1) | Accuracy (fallant 2) |
|------------------|---------------|----------|-----------------------|-----------------------|
| OCR+CNN          | 4.861             | 0.072        | 0.075                     | 0.145                     |
| YOLO+CNN         | **2.211**             | **0.524**        | **0.569**                     | **0.673**                     |

Pel dataset d'IIIT obtenim els següents resultats:

| Mètode           | Edit Distance | Accuracy | Accuracy (fallant 1) | Accuracy (fallant 2) |
|------------------|---------------|----------|-----------------------|-----------------------|
| OCR+CNN          | 4.041             | 0.231        | 0.241                     | 0.415                     |
| YOLO+CNN         | **0.467**             | **0.689**        | **0.702**                     | **0.894**                     |

La combinació de YOLO+CNN mostra un millor rendiment i millors resultats en general tant en termes d'edit distance com per les diferents mètriques d'accuracy (la normal i les que tenen en compte diferents nivell de toleràcia) en comparació amb OCR+CNN. Això es deu principalment al fet que YOLO realitza deteccions més precises i evita la sobre-detecció. D'altra banda, l'OCR tradicional tendeix a segmentar incorrectament i, en molts casos, sobredetecta (com ja hem discutit en apartats anteriors), cosa que afecta negativament la predicció de paraules. Això pot portar a la predicció incorrecta de més lletres de les que hi ha o directament a la predicció incorrecta de lletres degut a una mala detecció inicial (veure confusion matrix següent).

![Confusion matrix per les dos metodologies/pipelines](https://i.imgur.com/G4fPmHO.png)

A més, en general podem observar que s'obtenen millors resultats pel dataset de IIIT en comparació amb el nostre propi dataset. Això és principalment perquè el nostre dataset (a més de ser considerablement més gran) conté una gran varietat de fonts, algunes de les quals poden ser ambigües i difícils d'interpretar, a més de què també presenta paraules més complexes. D'altra banda, el dataset d'IIIT consisteix principalment en fonts estàndards, cosa que en facilita el processament i l'obtenció de resultats més precisos.

Apart de les dues metodologies OCR+CNN i YOLO+CNN ja exporadem, també vam decidir dur a terme experiments utilitzant només YOLO per avaluar el seu rendiment en tasques de *multiclass detection*. El nostre objectiu era fer ús dels mateixos datasets, però en lloc de utilitzar YOLO per detectar els diferents caràcters de forma individual, fem ús del YOLO per detectar el caràcter complet segons la seva classe, cosa que ens va permetre fer el reconeixement de paraules. Aquesta exploració addicional ens va permetre avaluar la *performance* de YOLO per una tasca de detecció multiclasse i comparar-ho amb els resultats obtinguts mitjançant les altres metodologies.


Pel dataset nostre obtenim els següents resultats:

| Mètode           | Edit Distance | Accuracy | Accuracy (fallant 1) | Accuracy (fallant 2) |
|------------------|---------------|----------|-----------------------|-----------------------|
| OCR+CNN          | 4.861             | 0.072        | 0.075                     | 0.145                     |
| YOLO+CNN         | **2.211**             | **0.524**        | **0.569**                     | **0.673**                     |
| YOLO             | 2.872             | 0.435        | 0.561                     | 0.568                     |

Pel dataset d'IIIT obtenim els següents resultats:

| Mètode           | Edit Distance | Accuracy | Accuracy (fallant 1) | Accuracy (fallant 2) |
|------------------|---------------|----------|-----------------------|-----------------------|
| OCR+CNN          | 4.041             | 0.231        | 0.241                     | 0.415                     |
| YOLO+CNN         | **0.467**             | **0.689**        | **0.702**                     | **0.894**                     |
| YOLO             | 0.895             | 0.567        | 0.628                     | 0.774                     |

En els resultats anteriors podem veure com fer ús només del YOLO per a la detecció multiclasse pels diferents datasets obtenim uns resultats superiors en comparació amb la metodologia OCR+CNN, però no aconseguim el rendiment obtingut amb YOLO+CNN.
La raó principal darrere d'aquests resultats creiem que recau en les propies capacitats intrínseques de cada enfoc. Per començar tenim el model de YOLO, que està dissenyat específicament per a la detecció d'objectes en imatges, cosa que li permet dur a terme una bona detecció, tot i que la part de classificació pot no arribar a ser tan òptima. D'altra banda, tenim la metodologia OCR+CNN, que encara que pot segmentar i reconèixer caràcters individuals amb certa precisió, es pot veure limitada per la seva capacitat de detecció, ja que com hem comentat anteriorment l'OCR no és capaç de dur a terme una detecció amb alta exactitud. Finalment tenim YOLO+CNN quecombina el YOLO amb una CNN. D'aquesta forma aprofitem les fortaleses dels dos mons, per un costat tenim el YOLO que s'encarrega de dur a terma una detecció precisa de les lletres de les paraules; i per l'altre costat la CNN, que s'enfoca en el reconeixement i la classificació dels caràcters continguts en aquestes paraules. Aquesta combinació dona lloca a una millor *performance* en comparació amb l'ús exclusiu de YOLO o la metodologia OCR+CNN.

## Segon approach

En aquest segon enfoc per la nostra tasca de *word recognition* ens proposem tractar una limitació important que ja hem mencionat anteriorment del nostre primer enfoc. Fins ara, hem tractat cada caràcter com un element individual en el procés de reconeixement de paraules, sense tenir en compte el context i la seqüencialitat de la paraula. La informació contextual i la relació entre els caràcters poden tenir un paper crucial en la precisió i la coherència de la predicció de paraules, per tant és important tenir-ho en compte per poder dur a terme la tasca.

Les paraules no són simplement una col·lecció de caràcters aïllats, sinó que tenen una estructura i un significat que es construeixen a partir de la seva seqüència i context. Al no considerar aquests aspectes, podem perdre informació valuosa que ens podria ajudar a millorar els nostres resultats. Per tant, en aquest nou enfocament volem tenir en compte aquesta informació por poder realitzar la tasca. En tenir en compte el context i la seqüencialitat, esperem poder obtenir millores en les nostres prediccions i millorar la precisió general del nostre sistema de *word recognition*.

### CRNN

Per aquest nou enfoc on volem tenir en compte la informació contextual i la seqüencialitat durant el procés de reconeixement de paraules, farem ús d'un model anomenat Convolutional Recurrent Neural Network (CRNN). Aquest model combina capes convolucionals per a l'extracció de característiques visuals seguit de capes recurrents per a capturar la seqüencialitat i el context en una paraula. D'aquesta forma obtenim una representació més profunda dels patrons i característiques per cada caràcter, així com la relació entre ells dins d'una paraula.

![Arquitectura CRNN](https://i.imgur.com/gTMichU.png)

## Entrenament

Actualment, ens trobem en la fase d'entrenament d'aquest model. Estem treballant tant amb el dataset d'IIIT com amb el nostre propi dataset generat. No obstant, ens estem enfrontant a un problema persistent en què tant la *train loss* com la *test loss* s'estan estancant, cosa que indica que el model no està millorant/aprenent a mesura que avança l'entrenament.

![Canvi d'scheduler IIIT](https://i.imgur.com/aqqbL2i.png)

![Canvi d'scheduler pel nostre dataset](https://i.imgur.com/PcGgton.png)

Hem realitzat diversos intents per abordar aquest problema, com ara ajustar el scheduler (les dos figures anteriors), modificar el learning rate i explorar altres paràmetres relacionats. No obstant això, malgrat aquests canvis, tant la *train loss* com la *test loss* continuen estancant-se, cosa que indica que el model no està aprenent de manera efectiva les característiques i els patrons necessaris per poder reconeixer les paraules.

## Futures millores
El treball realitzat abarca molts models, però no és del tot complet ja que no en tots els casos es fa una exploració completa d’aquests models. Per aquest motiu, en aquest apartat es mostra quins serien els següents passos per millorar l’estat acutal.
- El primer pas serà fer una avaluació correcta del model de CRNN. Realment amb ell hem fet poques execucions d’entrenament i convindria seguir avaluant-lo i provant diferents hiperparàmetres. Tot i això, sembla que indiferentment dels hiperparàmetres, el model seguia sense aprendre: no tenia prou capacitat. Llavors, s’hauria d’afegir capes convolucionals per fer-lo més complex i seguir avaluant.
- El segon pas seria fer un millor ús dels datasets:
  - S’hauria d’entrenar i avaluar tots els models amb tots els datasets.
  - S’haurien de mesclar datasets i mirar les mètriques també amb aquestes agrupacions.
- El darrer pas seria entrenar la PHOCNET amb un dataset més complex, ja que l’actual simplifica molt més la tasca, que tot i permetre’ns saber que el model és capaç d’aprendre, no ens permet saber si és útil per casos més reals.

## Conclusions
En aquest informe hem mostrat com funciona el codi i els resultats de la feina feta en el darrer mes. Al final del transcurs d’un projecte és quan és més fàcil treure’n conclusions. A continuació enumerarem les principals conclusions que ja hem esmentat en el text:
- La PHOCNET requereix d’un alt coneixement de modificació d’hiperparàmetres per tal de fer-la aprendre
- La representació PHOC és útil per representar paraules
- La detecció amb OCR és pobre en imatges on apareixen més elements que lletres.
- Un model de YOLO+CNN, on la YOLO faci la detecció de caràcter i la CNN el reconeixement, funciona millor que una YOLO que faci ambdues tasques alhora

## Referències
- Dades
  - Synthetic word data 9M images of 90k words, 10 GB: https://www.robots.ox.ac.uk/~vgg/data/text/
  - Real data, 5k words: https://cvit.iiit.ac.in/research/projects/cvit-projects/the-iiit-5k-word-dataset

- Starting point:
  - https://github.com/ssudholt/phocnet

## Autors
  Adarsh Tiwari -
  Xavier Querol -
  Abril Piñol


Xarxes Neuronals i Aprenentatge Profund  -  
Grau d'Enginyeria de Dades, 
UAB, 2023
