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

#### Comparación
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

En aquesta secció parlarem de les tres arquitectures de xarxes neuronals convolucionals (CNNs) amb què hem experimentat per al reconeixement de caràcters: ResNet18, VGG11 i Inception_v3. A continuació aprofundirem una mica més sobre cada arquitectura i seguidament explicarem l'entrenament i el *hyperparameter tuning* de cadascuna.

##### 1. Resnet18

![Arquitectura ResNet-18](https://i.imgur.com/MlTuRpO.png)

La Resnet18 es tracta d'una arquitectura de xarxa convolucional amb 18 capes de profunditat composta per *layers* convolucionals i blocs residuals. Fa ús de filtres de tamany reduit de 3x3 per capturar característiques visuals de les imatges i consta d'una estructura relativament baixa en termes de profunditat.

##### 2. VGG11

![Arquitectura VGG-11](https://i.imgur.com/sn9LbYf.png)

La VGG11 és una xarxa CNN composta per capes convolucionals seguides de capes de pooling, seguides finalment d'un conjunt de *fully connected layers*. Utilitza filtres de mida petita (3x3) en totes les capes convolucionals, mitjançant els quals captura característiques visuals a les imatges. 

##### 3. GoggleNet (Inception_v3)

![Arquitectura Inception_v3](https://i.imgur.com/zChawpt.png)

La GoogleNet (??? és lo mateix que Inception_v3???) és una arquitectura CNN que es basa en la idea dels moduls d'Inception, que fan ús de filtres de diferents mides (1x1, 3x3, 5x5) en paral·lel per poder capturar característiques visuals a diferents escales a partir de les imatges. Aquests mòduls s'ajunten en profunditat per permetre l'extracció de característiques tant locals com globals de les imatges. Inception_V3 també fa ús de tècniques de regularització i factorització de les capes de convolucions per així millorar el rendiment i reduir el nombre de paràmetres.

#### Hyperparameter tuning

Aquesta secció estarà dedicada a la cerca i ajust de hiperparàmetres per les diferents arquitectures CNN que hem utilitzat, ResNet18, VGG11 e Inception V3. L'objectiu és intentar trobar la combinació òptima d'hiperparàmetres que maximitzi el rendiment de les nostres xarxes. Començarem entrenant les xarxes amb els seus paràmetres *default* i en funció de les nostres necessitats experimentarem amb el learning rate, els diferents schedulers, weight decay, etc.

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

##### 3. GoogleNet (Inception_v3)

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
| Inception V3 |   0.9717   |   0.9742   |

##### Pel dataset d'IIIT:

| Model    | Train Acc | Test Acc  |
|-----------|-----------|-----------|
| ResNet18    |   **0.9984**   |   **0.8781**   |
| VGG11       |   0.9227   |   0.8461   |
| Inception V3 |   0.9900   |   0.8689   |

Tant pel nostre dataset propi com pel d'IIIT el model basat en RenNet18 mostre un rendiment superior en termes d'accuracy tant pel *train set* com pel *test set*. Això ens indica que la ResNet18 és capaç d'aprendre i generalitzar millor les característiques dels caràcters presents als diferents datasets, resultant en una millor accuracy. Aquests resultats donen suport a la nostra elecció d'utilitzar la ResNet18 com el model per la tasca de reconeixement dels caràcters pels dos datasets.

### Detection + Recognition