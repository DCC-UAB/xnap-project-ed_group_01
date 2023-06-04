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