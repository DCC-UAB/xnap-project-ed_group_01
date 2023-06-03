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

Vistes les diferents metodologies per poder dur a terme la part de detecció dels caràceters del nostre model, ara ens centrarem en la part de reconaixement d'aquests caràcters. Per poder fer això, experimentarem amb diferents arquitectures de xarxes convulacionals (CNNs), les entrenarem, i avaluarem la seva capacitat d'assignar l'etiqueta correcta a cada caràcter. Per poder dur a terme aquest entrenament, un altre cop farem ús dels dos datasets dels que ja hem parlat en l'apartat anterior: el dataset de IIIT (que només conté 5K samples) i el dataset custom generat per nosaltres (que conté 55K samples). Per poder dur a terme l'entrenament, farem ús de la informació sobre les bounding boxes que tenim en els dos datasets mencionats; aquesta informació ens permetrà extreure, retallar i seleccionar els caràcters individuals de cada imatge per a l'entrenament dels models (següent figura). L'objectiu és que el model convulacional aprengui a reconèixer i assignar les etiquetes correctes a cada caràcter individual a partir d'aquestes imatges retallades. 


![Entrenament per reconeixement de caràceters](https://i.imgur.com/P9ufVvK.png)

#### Arquitectures utilitzades

En aquesta secció parlarem de les tres arquitectures de xarxes neuronals convolucionals (CNNs) amb què hem experimentat per al reconeixement de caràcters: ResNet18, VGG11 i Inception_v3. A continuació aprofundirem una mica més sobre cada arquitectura i seguidament explicarem l'entrenament i el *hyperparameter tuning* de cadascuna.

##### 1. Resnet18

![Arquitectura ResNet-18](https://www.researchgate.net/publication/336642248/figure/fig1/AS:839151377203201@1577080687133/Original-ResNet-18-Architecture.png)

La Resnet18 es tracta d'una arquitectura de xarxa convolucional amb 18 capes de profunditat composta per *layers* convolucionals i blocs residuals. Fa ús de filtres de tamany reduit de 3x3 per capturar característiques visuals de les imatges i consta d'una estructura relativament baixa en termes de profunditat.

##### 2. VGG11

![Arquitectura VGG-11](https://www.researchgate.net/publication/336550999/figure/fig1/AS:814110748987402@1571110536721/Figure-S1-Block-diagram-of-the-VGG11-architecture-Adapted-from-https-bitly-2ksX5Eq.png)

La VGG11 és una xarxa CNN composta per capes convolucionals seguides de capes de pooling, seguides finalment d'un conjunt de *fully connected layers*. Utilitza filtres de mida petita (3x3) en totes les capes convolucionals, mitjançant els quals captura característiques visuals a les imatges. 

##### 3. GoggleNet (Inception_v3)

![Arquitectura Inception_v3](https://www.researchgate.net/publication/349717475/figure/fig5/AS:996933934014464@1614698980419/The-architecture-of-Inception-V3-model.ppm)

La GoogleNet (??? és lo mateix que Inception_v3???) és una arquitectura CNN que es basa en la idea dels moduls d'Inception, que fan ús de filtres de diferents mides (1x1, 3x3, 5x5) en paral·lel per poder capturar característiques visuals a diferents escales a partir de les imatges. Aquests mòduls s'ajunten en profunditat per permetre l'extracció de característiques tant locals com globals de les imatges. Inception_V3 també fa ús de tècniques de regularització i factorització de les capes de convolucions per així millorar el rendiment i reduir el nombre de paràmetres.
