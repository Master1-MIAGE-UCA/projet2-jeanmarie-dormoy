CC=mpicc
EXEC=mpirun
#CFLAGS=-Wall -fopenmp -lm -g
CFLAGS=-std=c99 -lm -fopenmp -Wall
PFLAGS=-DPRINT
SRC=src/Dormoy.c
EXE=bin/Dormoy
PEXE=bin/Dormoy_print
run: src/Dormoy
	./run.sh $(np)

all: classic print

re: clean classic

reprint: clean print

print: $(SRC)
	$(CC) $(CFLAGS) -o $(PEXE) $^ $(PFLAGS)

classic: $(SRC)
	$(CC) $(CFLAGS) -o $(EXE) $^

mycode_mpi: mycode_mpi.c
	$(CC) -o $@ $^

#re: clean mycode_mpi

del_logs:
	rm -rf logs/*

clean:
	rm -rf *~ $(EXE) $(PEXE)
