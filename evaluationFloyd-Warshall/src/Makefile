CC=mpicc
EXEC=mpirun
CFLAGS=-Wall -fopenmp -lm -g

run: Dormoy
	./run.sh $(np)

re: clean Dormoy
Dormoy: Dormoy.c
	$(CC) $(CFLAGS) -o $@ $^

mycode_mpi: mycode_mpi.c
	$(CC) -o $@ $^

#re: clean mycode_mpi


clean:
	rm -rf *~ mycode_mpi Dormoy
