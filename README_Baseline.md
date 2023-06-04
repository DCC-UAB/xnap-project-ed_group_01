
# Introducció a la tasca
	Cropped word recognition és la tasca que es tractarà. Parteix de les cropped words, que són les regions concretes d’imatges que contenen paraules, nombres o una combinació d’ambdòs, i s'obtenen a partir dels seus bounding boxes. Donat aquest retall de la imatge, la finalitat de la tasca és reconeixer el text que hi apareix. 


# Baseline
	Aquest projecte parteix d'una baseline com a punt de partida per al desenvolupament de la tasca. Aquesta es basa en una arquitectura anomenada PHOCNet (Pyramidal Histogram Of Characters Network) [referència paper][link a l'apartat PHOC], un model de xarxa neuronal dissenyat específicament per tasques de reconeixement de paraules.

## Datasets
* IIIT 5K [link a la citació dataset]
	El primer dataset proporcionat el formen imatges resultants de cerques concretes a Google Images, alguns exemples d'aquestes cerques serien: billboards, signboard, house numbers, movie posters... entre altres.
	Conté un total de 5000 imatges en color.
    ![alt text](https://imgur.com/u1qPQYL.png) 
    
* VGG [link a la citació dataset]
	Aquest segon dataset, a diferència de l'anterior, està generat de manera sintètica i no té color. Està format per diferents fonts i consta de 9 milions d'imatges que contenen 90.000 paraules en anglès.
    ![alt text](https://imgur.com/dtUGt6v.png) 

* Dataset senzill propi 
	Per últim, s'ha creat un nou dataset per dur a terme diferents proves. Aquest és també sense color, fons blanc i lletra en negre, amb fonts bàsiques, llegibles i amb la possibilitat de generar tantes imatges com es desitgi, amb un màxim al voltant dels 90000 que és el nombre de paraules diferents al diccionari que es fa servir.
    ![alt text](https://imgur.com/0d6YWgr.png) 

## PHOC
### Arquitectra PHOCNet
	L'arquitectura del model del que partim es mostra en aquesta primera imatge. Com es pot observar, consta d'un seguit de convolucions intercal·lades amb un parell de capes de max pooling, aquestes últimes, posteriors a la primera i segona convolució respectivament. Seguidament s'hi pot trobar una capa de Spatial Pyramid Pooling, aquesta s'afegeix perquè permet a les CNNs rebre com a entrada imatges de diferents mides i, tot i això, resultar en un output de mida constant. El motiu en són les fully connected layers, que han de rebre com a entrada sempre la mateixa mida.
	Per acabar, l'última capa de l'arquitectura consta de 604 neurones, que és la mida de PHOC, la representació de les paraules que volem aconseguir i que dona nom al model.
![alt text](https://imgur.com/unjt2Zn.png)


### Representació PHOC
	Amb l'objectiu de tenir una representació de mida fixa per totes les paraules i que contingui informació dels caràcters que la fromen. En aquesta s'hi consideren diferents nivells, el primer consisteix en un vector amb tantes posicions com possibles caràcters es vulguin predir, en aquest cas, tantes com lletres té l'abecedari anglès i deu posicions més pels digits del 0 al 9. Totes s'inicialitzen a 0 i són només les posicions corresponents als caràcters que apareixen en la paraula les que deixaran de tenir un 0 per tenir-hi un 1. Fins aquí el primer nivell, el següent està format per dos vectors com l'anterior i dividint la paraula en dos, els caràcters que apareguin en la primera meitat d'aquesta, tindran un 1 en les posicions corresponents del primer vector i en el segon, tindran un 1 les posicions d'aquells caràcters que apareguin en la segona meitat de la paraula. En el cas del tercer nivell, es divideix la paraula en 3 parts i cadascuna d'aquestes es representa en un vector diferent, de manera que aquest nivell el formen 3 vectors. Pels nivells següents es seguiria aquest patró. Per una millor comprensió es pot consultar la figura següent , les posicions blaves correspondrien al valor 1 i la resta al 0, a més faltarien les 10 posicions dels dígits, addicionalment L1, L2 i L3 corresponen als nivells 1, 2 i 3, respectivament. 
![alt text](https://imgur.com/C0UxRMN.png) 
	
	Aquesta és l'essència de la representació PHOC, partint d'aquí, les 604 posicions que s'esmentaven anteriorment les formen la concatenació dels nivells 2, 3, 4 i 5 juntament amb bigrams. Bigrams és un vector que parteix del mateix concepte, però cada posició enlloc de representar un únic caràcter representa una combinació de dos caràcters. Per no agafar totes les possibles combinacions, que en són moltes, s'utilitzen només les més comuns de la llengua anglesa i se'n fa dos nivells.


## Implementació
	Pel que fa a la implentació, el codi original del que es parteix està desenvolupat en Caffe i s'ha optat per passar-ho a Pytorch. Un cop el codi en pytorch és funcional i donada una imatge es pot obtenir la seva representació PHOC, d'aquesta se n'ha d'obtenir la paraula a la que correspon. Per fer-ho s'ha utilitzat un KNN amb la representació PHOC precalculada de 90000 paraules, de manera que per passar d'imatge a text el procés consisteix en: primer passar la imatge per la PHOCNet, obtenir així la representació, aquesta passar-la al KNN i que ens retorni d'aquestes 90000 paraules quina és la més semblant a la obtinguda a partir de la imatge.

## Anàlisi de resultats
### Primers resultats
	Aquest primer model s'entrena canviant diferents hiperparàmetres, es van fer les següents proves:
    * Loss: BCEwithlogits
    * Schedulers: StepLR, ReduceLROnPlateau, CosineAnnealingLR
    * Optimizers: SGD amb momentum, Adam amb l'scheduler ReduceLROnPlateau
    Malgrat les diverses proves els resultats acaben essent sempre semblants, el model apren a predir tot zeros en la representació de totes les paraules i només hi afegeix uns en els caràcters més freqüents. Com s'ha vist aquesta representació està formada per molts zeros és per això que amb aquestes prediccions la loss baixa i el model s'estanca en aquest minim local del que tant costa sortir. En la pròxima figura es pot veure un exemple d'aquestes prediccions amb el dataset VGG, les imatges d'aquest hi apareixen normalitzades.
![alt text](https://imgur.com/cCy64lx.png) 

### Resultats finals
	Veient els primers resultats i com els canvis en els diferents hiperparàmetres no  donaven resultats, les prediccions no milloraven, es va optar per crear un nou dataset molt més senzill per comprovar que el model fos capaç de dur a terme la seva tasca. Els detalls d'aquest dataset estan especificats en l'apartat Datasets[link], en la subsecció Baseline[link]. Altrament, va sorgir la idea de canviar els pesos de la loss per canviar la manera de penalitzar els errors i forçar al model a predir més uns. Aquests weights es van definir de manera que el pes de cometre un error al predir un caràcter era l'invers de freqüència d'aparició del mateix en les 90000 paraules del diccionari que es fa servir.
	Finalment amb aquest parell de canvis, amb BCEloss, Adam optimizer i ReduceLROnPlateu per ajustar el learning rate, els resultats obtinguts són remarcablement bons. S'observa com la loss no s'estanca, l'accuracy puja fins a arribar pràcticament a 1 i l'edit distance baixa a tocar de 0. Aquesta última és una altra mesura que s'utilitza per avaluar distància entre paraules de mida variables. La finalitat n'és que, a diferència de l'accuracy, si el model prediu de manera errònia només alguns caràcters en certes paraules, no es tenen en compte com prediccions incorrectes envers les correctes on s'encerten tots els caràcters, sinó que se'ls asigna un valor continu en funció de com de diferents són la predicció  i la paraula predir. A continuació les visulaitzacions dels resultats junatment amb una mostra del dataset senzill predita correctament.
![alt text](https://imgur.com/cr3EJY8.png) 


