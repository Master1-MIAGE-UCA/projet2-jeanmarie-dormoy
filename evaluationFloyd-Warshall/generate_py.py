import random
import os

SAMPLEDIR="data"
SAMPLENAME="mat_"
RESULTNAME="result_"
MAX=7
MIN=-2 #put less of 0 to increase probability to have inf values
START=4
END=6

def rint():
    v = random.randint(MIN, MAX)
    return v if v>0 else float('+inf')

def floydWarshall(graph, size):
    dist = list(map(lambda i : list(map(lambda j : j, i)), graph) )
    
    for k in range(size):
        for i in range(size): 
            for j in range(size): 
                dist[i][j] = min(dist[i][j], dist[i][k]+ dist[k][j]) 
    
    return dist

def generate(n, size):
    sample = open(SAMPLEDIR+"/"+ SAMPLENAME + str(n), "w")
    result = open(SAMPLEDIR+"/"+ RESULTNAME + str(n), "w")

    A = [[rint() if j!=i else 0 for i in range(size)] for j in range(size)]

	#Note qu'il est extrêmement dure d'écrire
	#sample.write(str(size))
	#sample.write("\n")
	#Pour donner en première ligne la taille de l'échantillon

    for raw in A:
        sample.write(' '.join([str(e) if e != float('+inf') else '0' for e in raw]))
        sample.write(' ')
        sample.write('\n')
    
    res = floydWarshall(A, size)
    
    for raw in res:
        result.write(' '.join([str(e) if e != float('+inf') else 'i' for e in raw]))
        result.write(' ')
        result.write('\n')

def main():
    for k in range(START, END):
        #size = random.randint(2**(i+1), 3*(i+1))
        generate(k, 2**(k))
        #print(str(k) + " " + str(2**(k//2)))

if __name__ == "__main__":
    main()
