# SI4 - Spring 2020 - Project 2 Parallelism
## Author:	Jean-Marie DORMOY


### Partie multiplication min/+ parallèle openMP

Exemple avec le calcul de l'élément en position (i,j) dans la sous-matrice résultat ```res```
considérée. On a:

Colonne B:	```2
			4
			5
			0				
			1		
			4		
			3
			3```
Ligne A : ```1 2 3 4 5 6 7 9```
On calcule:
- ```1+2 = 3```	stocké dans temp[0]
		   ```2+4 = 6```		        temp[1]
		   etc..

- On obtient: le tableau temp:
```3 6 8 4 6 10 10 12```
On fait: ```temp[0] = min(temp[0], temp[1]) puis temp[2] = min(temp[2], temp[3]) ,etc...```:
```3 6 4 4 6 10 10 12```
Répétition du procédé: ```temp[0] = min(temp[0], temp[2]) ,etc...```:
```3 6 4 4 6 10 10 12```
Idem: ```temp[0] = min(temp[0], temp[4]) ,etc...```:
```3 6 4 4 6 10 10 12```

Le procédé terminé, on a seulement à récupérer le résultat de ```min(a1 + b1, ..., a8 + b8)``` en lisant
la valeur de ```temp[0]```. Ce procédé consiste en ```log(8) = 3``` boucles, parallélisables en openMP. On
préfère faire 3 boucles plutôt que 8!

