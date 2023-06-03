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
Per poder fer això utilitzem el conjunt de dades de IIIT, ja que és l'únic conjunt de dades proporcionat a la *baseline* que conté informació sobre les bounding boxes dels caràcters, cosa que ens permet avaluar tant l'OCR com el YOLO. A més, generem un conjunt de dades propi (següent figura) que inclou lletres amb diferents fonts, lletres, colors, etc. i també conté informació sobre les bounding boxes. Això ens proporciona un altre conjunt de dades amb què podem avaluar el rendiment d'OCR i YOLO en condicions més diverses.
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



![Sobredetecció OCR](https://i.imgur.com/fWmfMqH.png)

Aquests resultats donen suport a la idea que YOLO és més efectiu en la detecció precisa de caràcters en diferents escenaris i condicions en comparació amb OCR.

### Recognition

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam fermentum fringilla massa, non rhon
