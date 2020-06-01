# SI4 - Spring 2020 - Project 2 Parallelism
## Author:	Jean-Marie DORMOY

### 0. Fonctionnement du main

On lit le fichier et on le stocke dans une matrice ```input_matx```. On met à jour cette 
matrice pour la transformer en W. On alloue les ressources néccesaires (cf. §3) puis on 
propage ```log2(W->height)``` à tous les processus. Le but, par exemple pour un input de 
taille ```[8x8]```, est de faire 3 tours de boucle, où on calculera successivement:
- ```W² = Do_Multiply(W, W)```
- ```W⁴ = Do_Multiply(W², W²)```
- ```W^8 = Do_Multiply(W⁴, W⁴)```\
\
Cet algorithme nous donne une complexité en O(log2(N)) au lieu de O(N) pour obtenir W^n.\
\
Lorsqu'on arrive à la fin de chaque tour de boucle, on libère les ressources qui ont été allouées
lors de la multiplication en anneau effectuée durant ce tour de boucle (la liste de Local_Result
allouée par chaque processus). Cette partie pourrait être
optimisée en conservant l'allocation de ces ressources et en modifiant les valeurs contenues
dans celles-ci à chaque multiplication en anneau (on libère toutes les ressources, et ce uniquement à la fin du programme).

```c
void Do_Multiply(
		int rank, int numprocs, Matrix *a, Matrix *b,
		Matrix ***sub_matrices_a, Matrix ***sub_matrices_b,
		int *len_submat_a, int *len_submat_b,
		Matrix **local_a_submatrix, Matrix **local_b_submatrix,
		Local_Result **local_res_list, Local_Result **local_res,
		int *round);
```
Cette fonction permet de regrouper toutes les étapes d'une multiplication min/+ de 2 matrices en
anneau. Elle prend en paramètres toutes les data structures nécessaires au bon déroulement de chacune
des étapes, en particulier pour le stockage des données dans chacun des processus. Brièvement, elle 
appelle dans l'ordre:

- Initialization_Local_Result (calloc, cf. §3)
- Scatter_A_Lines
- Scatter_B_Cols
- Local_Computation_Each_Proc
- Gather_Local_Results

Les appels à chacune de ces fonctions (qui utilisent toutes MPI_Send/Recv, hormis la première) se font avec un tag MPI spécifique pour chacune d'entre elles, afin de discriminer les envois et réceptions
de message MPI entre différents appels de fonction (à l'origine de ces envois de message).

```c
Fill_Matrix_With_Results(w, local_res_list, numprocs);
```
- Après l'exécution de la méthode Do_Multiply (qui a fait un gather), la variable ```local_res_list``` dans le 
main du processus 0 contient toutes les sous-matrices correspondant aux résultats partiels de tous les 
processeurs. Cette méthode met "proprement" à jour la matrice W avec ces résultats partiels. W mise à jour, on peut continuer la boucle pour calculer W^n.
- il n'est pas possible, a priori, d'utiliser openMP pour paralléliser cette méthode: en effet, les 
données doivent être écrites dans un ordre bien précis dans la variable w, ce qui ne se prête pas au 
parallélisme. C'est dommage, car cette méthode contient 4 boucles imbriquées (car on fait circuler les 
blocs de A et pas ceux de B), ce qui ajoute 2s de temps d'exécution supplémentaires pour -np 5 et mat_3 
en input. Une amélioration possible pour gagner en temps d'exécution est la circulation des blocs de B 
au lieu de ceux de A: ainsi, l'écriture de ```Fill_Matrix_With_Results``` se ferait avec moins de boucles imbriquées (car on devrait fusionner des lignes, pas des colonnes).

### 1. Lecture de Fichier
```c
int first_pass(char *s);
```
Parse la 1ère ligne du fichier d'input pour déterminer le nombre d'éléments contenus dans
cette ligne. On en déduit ensuite les dimensions de la matrice carrée passée en input.

```c
Matrix *build_matrix(FILE *fp);
```
Renvoie l'adresse d'une Matrice allouée dynamiquement et remplie avec le contenu du fichier
correspondant au FILE* fp. À l'intérieur, on utilise fgets (std=c99) pour lire le fichier ligne par ligne
depuis le début, et on donne la ligne lue à une fonction qui va se charger de la parser afin de remplir les cases correspondantes dans la matrice, tout en mettant à jour une variable d'offset (pour remplir correctement la matrice).

### 2. Transformation A -> W
```c
void Transform_A_Into_W(Matrix *a);
```
Pour Obtenir la matrice W, on modifie la matrice a passée en paramètre "in-place" avec les règles suivantes: 
- wij = 0 if i = j
- wij = weight of  (i,j) if there is an edge between i and j
- wij = +inf otherwise
### 3. Data Structures et Allocations
```c
typedef struct Matrix {
    unsigned int *data;
    int width;
    int height;
    int size;
} Matrix;

typedef struct Local_Result {
	int index;
	Matrix *mat;
} Local_Result;
```
Ces 2 C-Struct possèdent chacune des méthodes pour construire, libérer une Matrix/Local_Result alloués
dynamiquement ou une liste/matrice de ces objets. Il y a également des fonctions permettant de faire une
copie de Matrix ou de Local_Result (copie profonde en utilisant calloc) et des fonctions d'affichages
pour ces deux types d'objets qui ont été utilisés pour le débuggage.
```c
void Initialization_Local_Result_List(Local_Result ***local_res_list, int numprocs);
```
Permet d'allouer dynamiquement un tableau 2D de largeur et longueur égales (valant
numprocs) et contenant numprocs * numprocs Local_Result. Ce tableau 2D permettra
de stocker les sous-matrices résultat calculées par tous les processeurs lors de la
phase de gather.
```c
void Initialization_Local_Result(Local_Result **local_res, int numprocs);
```
Alloue dynamiquement un tableau 1D de taille numprocs. Cette Data Structure permet
de stocker toutes les sous-matrices résultat propres à chaque processus.

### 4. Phase de Scatter

La première étape consiste à déterminer les dimensions de chaque sous-matrice de A et de B, ce qui se
fait avec la fonction Compute_Distribution qui renvoie un tableau de dimensions. Cette fonction est utilisée dans les 2 fonctions suivantes:
```c
Matrix** Explode_A_Into_Lines(Matrix *A, const int numprocs, int *len);
```
```c
Matrix** Explode_B_Into_Columns(Matrix *B, int numprocs, int *len);
```
Elles prennent en argument la matrice A ou B à éclater en sous-matrices, le nombre de processus et un
pointeur de int: len. Ces fonctions renvoies un tableau de numprocs sous-matrices et mettent à jour 
l'argument len passé par référence à sa nouvelle valeur numprocs.

Jusque là, le calcul des dimensions et la construction des sous-matrices s'est fait dans le processus 0.

Enfin, voici les fonctions réalisant le scatter:
```c
void Scatter_A_Lines(
		int rank, int numprocs,
		Matrix ***sub_matrices_a, int *len_submat_a,
		Matrix *a, Matrix **local_a_submatrix, int round);
```
```c
void Scatter_B_Cols(
		int rank, int numprocs, Matrix ***sub_matrices_b,
		int *len_submat_b, Matrix *b, Matrix **local_b_submatrix,
		int round);
```
Elles utilisent les 2 méthodes précédentes générant la liste de sous-matrices de A ou de B. Ensuite, le
processus p0 envoie l'une derrière l'autre les sous-matrices d'index numprocs -1, numprocs - 2, ..., 1.
Le processus p_numprocs-1 reçoit au final la sous-matrice d'index numproc -1, ..., le processus p1 reçoit celle d'index 1.
Pour les processus différents de p0, tant que la sous-matrice reçue n'est pas arrivée à son destinataire
final, elle est retransmise au processus suivant (grâce à Transmit_SubMatrix).
À la fin du procédé, chaque processus possède un bloc de A et un bloc de B. On précise qu'après la phase
de scatter, un bloc de B ne bougera pas de son processus, tandis qu'un bloc de A sera amené à circuler
lors de la phase de calcul.

### 5. Phase de Calcul (inclut la circulation)

Une fois le scatter terminé, on peut réaliser la phase de calcul qui consiste à alterner:\
```Calcul local -> Circulation -> Calcul local -> Circulation -> ... -> Calcul local -> Circulation```

On utilise une structure Local_Result qui contient: 
- un int correspondant à l'index/position de la sous-matrice résultat dans le processus considéré, ce qui nous sera utile pour la phase de gather pour parcourir dans l'ordre les sous-matrices résultat
- l'adresse de la sous-matrice résultat

À la fin de la phase de calcul, chaque processus contient une liste de numprocs Local_Result qu'il faudra
ensuite gather.
```c
void Local_Computation_Each_Proc(
		int numprocs, int rank, Matrix **local_a_submatrix,
		Matrix *local_b_submatrix, Local_Result *local_res,
		int round);
```
Cette fonction consiste, pour chacun des numprocs Local_Result de  chaque processus, à calculer la 
sous-matrice résultant de la multiplication openMP du sous-bloc de A et du sous-bloc de B actuellement 
présents dans le processus, et à stocker cette sous-matrice résultat dans le Local_Result courant.
Pour tous les processus, chaque étape de calcul de sous-matrice résultat est alterné avec un appel à
la circulation pour faire tourner les sous-blocs de A dans l'anneau.
```c
void Make_Local_A_Submatrices_Circulate(
		int rank, int numprocs, Matrix **local_a_submatrix, 
		int round);
```
Le code de cette méthode est assez bref: tous les processus de rang pair envoient leur sous-bloc de A
au processus suivant. Ensuite, chaque processus impair sauvegarde son sous-bloc de A courant, le remplace
par le sous-bloc de A reçu (celui qui a été envoyé par le processus pair situé juste avant lui dans
l'anneau) et transmet l'ancien sous-bloc sauvegardé au processus pair qui suit dans l'aneau.

### 6. Phase de Gather

Un bon exemple illustrant le fonctionnement de mon gather est plus parlant:

On suppose que chaque process possède sa liste de numprocs Local_Result et qu'on se situe juste après
la fin de la phase de calcul. A, B, C et D représentent la liste Local_Result de P0, P1, P2 et P3, 
respectivement (on suppose que numprocs=4).

Début de Gather:
P0			P1		P2		P3
A		<---B	<---C   <---D	Boucle 1

P0			P1		P2		P3
A,B		<---C	<---D		X	Boucle 2

P0			P1		P2		P3
A,B,C	<---D		X		X	Boucle 3 (on boucle de 1 à numprocs-1 = 3)

P0			P1		P2		P3
A,B,C,D		X		X		X

/!\ Ici X signifie juste qu'on ne va plus utiliser la liste de Local_Result car on est à la fin d'une
multiplication en anneau.

/!\ X pourrait faire croire qu'on libère la mémoire dans le Gather: ce n'est pas le cas, la mémoire
occupée par la liste de Local_Result de chaque processus est libérée à la fin d'une multiplication
en anneau (et réallouée dynamiquement au début de la prochaine multiplication par anneau: c'est un
point évoqué dans le §0 qui peut être amélioré).

### Multiplication min/+ parallèle openMP

Exemple avec le calcul de l'élément en position (i,j) dans la sous-matrice résultat ```res```
considérée. On a:

Colonne B:	```2 4 5 0 1 4 3 3```\
Ligne A :	```1 2 3 4 5 6 7 9```\
On calcule:
- ```1+2 = 3```	stocké dans temp[0]\
		   ```2+4 = 6```		        temp[1]\
		   etc..

- On obtient: le tableau temp:\
```3 6 8 4 6 10 10 12```\
On fait: ```temp[0] = min(temp[0], temp[1]) puis temp[2] = min(temp[2], temp[3]) ,etc...```:\
```3 6 4 4 6 10 10 12```\
Répétition du procédé: ```temp[0] = min(temp[0], temp[2]) ,etc...```:\
```3 6 4 4 6 10 10 12```\
Idem: ```temp[0] = min(temp[0], temp[4]) ,etc...```:\
```3 6 4 4 6 10 10 12```

Le procédé terminé, on a seulement à récupérer le résultat de ```min(a1 + b1, ..., a8 + b8)``` en lisant
la valeur de ```temp[0]```. Ce procédé consiste en ```log(8) = 3``` boucles, parallélisables en openMP. On
préfère faire 3 boucles plutôt que 8!

