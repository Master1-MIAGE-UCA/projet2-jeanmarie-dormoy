#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/time.h>

#define LINE_SIZE 500

typedef struct Matrix {
    unsigned int *data;
    int width;
    int height;
    int size;
} Matrix;
#define GET(mat, i, j) mat->data[j * mat->width + i]
#define SET(mat, i, j, val) GET(mat, i, j) = val

Matrix *new_Matrix(int width, int height) {
	Matrix *m = malloc(sizeof(Matrix));
	if (!m) {
		fprintf(stderr, "new_Matrix: Matrix malloc error\n");
		exit(4);
	}
	m->width = width;
	m->height = height;
	m->size = width * height;
	m->data = calloc(m->size, sizeof(unsigned int));

	if (!m->data) {
		fprintf(stderr, "build_matrix: res->data malloc error\n");
		exit(5);
	}
	return m;
}

int first_pass(char *s) {
	char *token;
	int len = 0;
	token = strtok(s, " ");
	while (token) {
		len +=1;
		token = strtok(NULL, " ");
	}
	return len;
}

void print_matrix(Matrix *m) {
	int len = m->width * m->height;
	for (int i = 0; i < len; ++i)
		printf("%d ", m->data[i]);
	puts("");
}

void pretty_matrix(Matrix *m) {
	int len = m->width * m->height;
	for (int i = 0; i < len; ++i) {
		if (i % m->width == 0)
			printf("\n%5d ", m->data[i]);
		else
			printf("%5d ", m->data[i]);
	}
	puts("");
}

void fill_matrix_with_line(
		unsigned int *array, int *index, char *line) {
	char *token;
	token = strtok(line, " ");
	while (token) {
		array[*index] = atoi(token);
		*index += 1;
		token = strtok(NULL, " ");
	}
}
Matrix *build_matrix(
		FILE *fp) {
	Matrix *res = malloc(sizeof(Matrix));
	int n = 0, index = 0;
	char *line = NULL, *save;
    size_t len = 0;
   	ssize_t read;
	if (!res) {
		fprintf(stderr, "build_matrix: res malloc error\n");
		exit(3);
	}
	line = calloc(LINE_SIZE, sizeof(char));
	if (!line) {
		fprintf(stderr, "build_matrix: line calloc error\n");
		exit(4);
	}
	save = calloc(LINE_SIZE, sizeof(char));
	if (!save) {
		fprintf(stderr, "build_matrix: save calloc error\n");
		exit(5);
	}
	if ((read = getline(&line, &len, fp)) != -1) {
		strcpy(save, line);
		n = first_pass(save);
	}
	if (!n) {
		if (res) free(res);
		if (line) free(line);
		if (save) free(save);
		return NULL;
	}
	res->height = res->width = n;
	res->data = calloc(n * n, sizeof(unsigned int));
	if (!res->data) {
		fprintf(stderr, "build_matrix: res->data malloc error\n");
		exit(5);
	}
	while (read != -1) {
		fill_matrix_with_line(res->data, &index, line);
		//printf("Retrieved line of length %zu:\n", read);
		read = getline(&line, &len, fp);
	}
	
	if (line) free(line);
	if (save) free(save);
	return res;
}

void sequentialMultiply(Matrix *a, Matrix *b, Matrix *res) {
	struct timeval t0, t1;
	gettimeofday(&t0, 0);
	for(int i=0; i < a->height; i++){
		for(int j=0; j < a->width; j++){
			for(int k=0; k< a->width; k++){
				GET(res, i, j) += 
					GET(a, i, k) * GET(b, k, j);
			}
		}
	}
	gettimeofday(&t1, 0);
	double elapsed = (t1.tv_sec-t0.tv_sec) * 1.0f + (t1.tv_usec - t0.tv_usec) / 1000000.0f;

	printf("sequentialMultiply time:  %f\n", elapsed);
}
int main(int argc, char *argv[]) {

    int rank, numprocs;// line_size = 0;
	FILE *fp;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	Matrix *mat, *res;
	//MPI_Status status;
	if (argc != 2) {
		fprintf(stderr,
				"usage:\n\tmpiexec -np X ./Dormoy input.txt");
		exit(1);
	}
	if (argv[1])
		fp = fopen(argv[1], "r");
    if (!fp) {
		fprintf(stderr, "error opening file%s\n",
				argv[1]);
        exit(2);
	}
	switch(rank) {
		case 0:
			puts("process 0");
			printf("argc= %d\n", argc);
			printf("=>%s\n", argv[0]);
			printf("=>%s\n", argv[1]);
			mat = build_matrix(fp);
			if (!mat) {
				fprintf(stderr, "var mat: build_matrix ret NULL\n");
				exit(6);
			}
			print_matrix(mat);
			puts("");
			pretty_matrix(mat);
			puts("-------------------");
			res = new_Matrix(4, 4);
			sequentialMultiply(mat, mat, res);
			pretty_matrix(res);
			break;
		case 1:
			puts("process 1");
			break;
	}
	MPI_Finalize();
	return 0;
}
