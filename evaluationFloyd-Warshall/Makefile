CC=mpicc
EXEC=mpirun
#CFLAGS=-Wall -fopenmp -lm -g
CFLAGS=-std=c99 -lm -fopenmp -Wall
SRC=src/Dormoy.c
EXE=bin/Dormoy
run: src/Dormoy
	./run.sh $(np)

re: clean $(EXE)

$(EXE): $(SRC)
	$(CC) $(CFLAGS) -o $@ $^

mycode_mpi: mycode_mpi.c
	$(CC) -o $@ $^

#re: clean mycode_mpi

del_logs:
	rm -rf logs/*

clean:
	rm -rf *~ $(EXE)
