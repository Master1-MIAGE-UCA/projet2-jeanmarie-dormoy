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
lors de la multiplication en anneau effectuée durant ce tour de boucle. Cette partie pourrait être
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
correspondant au FILE* fp. 

### 2. Transformation A -> W
```c
void Transform_A_Into_W(Matrix *a);
```
Pour Obtenir la matrice W, on modifie la matrice a passée en paramètre "in-place" avec les règles suivantes: 
- wij = 0 if i = j
- wij = weight of  (i,j) if there is an edge between i and j
- wij = +inf otherwise
### 3. Allocation des Data Structures
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

### Multiplication min/+ parallèle openMP

Exemple avec le calcul de l'élément en position (i,j) dans la sous-matrice résultat ```res```
considérée. On a:

Colonne B:	```2
			4
			5
			0				
			1		
			4		
			3
			3```\
Ligne A : ```1 2 3 4 5 6 7 9```\
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
préfère faire 3 boucles plutôt que 8!\

