CC=gcc
CFLAGS=-Wall -fopenmp -lm -g

re: clean reverse_index

reverse_index: reverse_index.c
	$(CC) $(CFLAGS) -o $@ $^

Dormoy: Dormoy.c
	$(CC) $(CFLAGS) -o $@ $^

mycode_mpi: mycode_mpi.c
	$(CC) -o $@ $^

#re: clean mycode_mpi


clean:
	rm -rf *~ mycode_mpi Dormoy reverse_index
