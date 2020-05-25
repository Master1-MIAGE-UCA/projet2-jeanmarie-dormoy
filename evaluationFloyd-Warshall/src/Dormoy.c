#include <omp.h>
#include <mpi.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/time.h>
#include <stdarg.h>
#define LINE_SIZE 500

typedef struct Matrix {
    unsigned int *data;
    int width;
    int height;
    int size;
} Matrix;
#define GET(mat, i, j) mat->data[i * mat->width + j]
#define SET(mat, i, j, val) GET(mat, i, j) = val

Matrix *new_Matrix(int height, int width);
void Destroy_Matrix(Matrix *m);
void Destroy_All_Matrices(int num, ...);
void Destroy_Matrix_Array(Matrix **arr, int len);
void print_raw_matrix(Matrix *m);
void print_matrix(Matrix *m);
void print_raw_array(unsigned int *arr, int len);

/* File Reading */
int first_pass(char *s);
void fill_matrix_with_line(
		unsigned int *array, int *index, char *line);
Matrix *build_matrix(FILE *fp);

/* Matrix filling & copying */
void randomlyFillMatrix(Matrix *m);
Matrix *matrixcpy(Matrix *src);
Matrix *matrixcpy_bis(Matrix *dest, Matrix *src);
Matrix *matrixcpy_reverseIndex(Matrix *src);

/* Tests */
int Test_Equals(Matrix *a, Matrix *b);

/* Scatter */
void fill_submatrix(Matrix *m, Matrix *A, int *offset);
void print_matrix_list(Matrix **m, int len);
void print_matrix_list_bis(Matrix **m, int len);

void fill_submatrix_bis(Matrix *m, Matrix *B, int *offset);
void print_raw_matrix_list(Matrix **m, int len);


void Finalizer(int rank, int numprocs, Matrix **sub_matrices_a,
		int len_submat_a, Matrix *a);

Matrix** Explode_A_Into_Lines(Matrix *A, int numprocs, int *len){ 
	int lines_A, remain, lines_remaining, divider, offset = 0;
	lines_A	= A->height;
	divider = lines_A / numprocs;
	remain = lines_A % numprocs;
	Matrix **matrix_list = NULL;
	printf("remain=%d\n", remain);

	matrix_list = calloc(numprocs, sizeof(Matrix*));
	*len = numprocs;
	if (remain == 0) {
		//we have divider sub matrices
		if (!matrix_list) {
			fprintf(stderr,
					"Explode_A_Into_Lines: matrix_list calloc error");
			exit(9);
		}
		for (int i = 0; i < numprocs; i++) {
			matrix_list[i] = new_Matrix(divider, A->width);
			fill_submatrix(matrix_list[i], A, &offset);
		}
	} else {
		lines_remaining = A->height - divider * (numprocs - 1);
		for (int i = 0; i <	numprocs - 1; i++) {
			matrix_list[i] = new_Matrix(divider, A->width);	
			fill_submatrix(matrix_list[i], A, &offset);
		}		
		matrix_list[numprocs - 1] = new_Matrix(
				lines_remaining, A->width);
		fill_submatrix(matrix_list[numprocs - 1], A, &offset);
	}
	return matrix_list;
}

Matrix** Explode_B_Into_Columns(Matrix *B, int numprocs, int *len){ 
	int cols_B, remain, cols_remaining, divider, offset = 0;
	cols_B	= B->width;
	divider = cols_B / numprocs;
	remain = cols_B % numprocs;
	Matrix **matrix_list = NULL;
	printf("remain=%d\n", remain);

	matrix_list = calloc(numprocs, sizeof(Matrix*));
	*len = numprocs;
	if (remain == 0) {
		//we have divider sub matrices
		if (!matrix_list) {
			fprintf(stderr,
					"Explode_A_Into_Lines: matrix_list calloc error");
			exit(9);
		}
		for (int i = 0; i < numprocs; i++) {
			//printf("b->height=%d divider=%d\n", B->height, divider);
			matrix_list[i] = new_Matrix(B->height, divider);
			//printf("1 => i=%d offset=%d\n", i, offset);
			fill_submatrix_bis(matrix_list[i], B, &offset);//TODO
			print_raw_array(
					matrix_list[i]->data, matrix_list[i]->size);
		}
	} else {
		cols_remaining = B->width - divider * (numprocs - 1);
		for (int i = 0; i <	numprocs - 1; i++) {
			matrix_list[i] = new_Matrix(B->height, divider);	
			fill_submatrix_bis(matrix_list[i], B, &offset);
		}		
		matrix_list[numprocs - 1] = new_Matrix(
				B->height, cols_remaining);
		fill_submatrix_bis(matrix_list[numprocs - 1], B, &offset);
	}
	return matrix_list;
}

void Scatter_A_Lines(
		FILE *fp, int rank, int numprocs, MPI_Status status,
		Matrix ***sub_matrices_a, int *len_submat_a, Matrix **a) {
	unsigned int *_exit;//, *buffer = NULL, ;
	//int size_msg;
	Matrix	*mat;
	_exit = calloc(1, sizeof(unsigned int));
	switch(rank) {
		case 0:	
			puts("process 0");
			//printf("argv[0]=%s\n", argv[0]);
			mat = build_matrix(fp);
			if (!mat) {
				fprintf(stderr, "var mat: build_matrix ret NULL\n");
				exit(6);
			}
			*a = new_Matrix(11, 4);
			randomlyFillMatrix(*a);
			puts("a:");
			print_matrix(*a);
			*sub_matrices_a = Explode_A_Into_Lines(
					*a, numprocs, len_submat_a);
			printf("len=%d\n", *len_submat_a);
			print_matrix_list(*sub_matrices_a, *len_submat_a);
			
			/*
			for (int i = 0; i < numprocs - 1; i++)
				MPI_Send(
						(*sub_matrices_a)[i]->data,
						(*sub_matrices_a)[i]->size,
						MPI_INT, (rank + 1) % numprocs,
						0, MPI_COMM_WORLD);*/
			break;
		default:
			printf("process %d\n", rank);
			/*
			for (int i = 0; i < numprocs - rank; i++) {
				MPI_Probe(rank - 1, 0, MPI_COMM_WORLD, &status);
				MPI_Get_count(&status, MPI_INT, &size_msg);
				buffer = calloc(size_msg, sizeof(unsigned int));
				if (!buffer) {
					fprintf(stderr, "buffer: calloc error\n");
					exit(10);
				}
				MPI_Recv(buffer, size_msg, MPI_INT, rank - 1, 0,
		   				MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				printf("received at rank: %d\n", rank);
				print_raw_array(buffer, size_msg);
				if (rank != numprocs - 1)
					MPI_Send(buffer, size_msg, MPI_INT,
							(rank + 1) % numprocs, 0,
							MPI_COMM_WORLD);
				else {*/
				if (rank == numprocs - 1) {
					_exit[0] = 255;
					MPI_Send(_exit, 1, MPI_INT, 0, 0, 
							MPI_COMM_WORLD);
				}
				//if (buffer) free(buffer);
				
			//}
			break;
	}
}

void Scatter_B_Cols(
		int rank, int numprocs, MPI_Status status,
		Matrix ***sub_matrices_b, int *len_submat_b, Matrix **b) {
	//unsigned int *_exit;//, *buffer = NULL, ;
	//int size_msg;
	//_exit = calloc(1, sizeof(unsigned int));
	//Matrix *temp;
	switch(rank) {
		case 0:	
			puts("process 0");
			*b = new_Matrix(3, 8);
			randomlyFillMatrix(*b);
			puts("b:");
			print_matrix(*b);
			*sub_matrices_b = Explode_B_Into_Columns(
					*b, numprocs, len_submat_b);
			//temp = matrixcpy_reverseIndex(*b);
			//print_matrix(temp);
			printf("len=%d\n", *len_submat_b);
			print_raw_matrix_list(*sub_matrices_b, *len_submat_b);
			puts("---------");
			print_matrix_list_bis(*sub_matrices_b, *len_submat_b);
			/*
			for (int i = 0; i < numprocs - 1; i++)
				MPI_Send(
						(*sub_matrices_a)[i]->data,
						(*sub_matrices_a)[i]->size,
						MPI_INT, (rank + 1) % numprocs,
						0, MPI_COMM_WORLD);*/
			break;
		default:
			printf("process %d\n", rank);

			break;
	}
}
			
void Finalizer(
		int rank, int numprocs,
		Matrix **sub_matrices_a, int len_submat_a, Matrix *a){
	unsigned int *buffer = calloc(1, sizeof(unsigned int));
	switch(rank) {
		case 0:
			MPI_Recv(buffer, 1, MPI_INT, rank - 1, 0,
					MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			if (*buffer == 255) {
				//printf("===>buffer=%d\n", *buffer);
				//printf("=>submat_a=%p\n", (void*) sub_matrices_a);
				//printf("=>len_submat_a=%d\n", len_submat_a);
				Destroy_All_Matrices(1, a);
				Destroy_Matrix_Array(sub_matrices_a, len_submat_a);
			}	
			break;
		default:
			break;
	}	
}
int main(int argc, char *argv[]) {
    int rank, numprocs, len_submat_a, len_submat_b;
	Matrix **sub_matrices_a, *a, **sub_matrices_b, *b;
	FILE *fp;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Status status;
	if (argc != 2) {
		fprintf(stderr,
				"usage:\n\tmpirun -np X ./Dormoy input.txt");
		exit(1);
	}
	if (argv[1])
		fp = fopen(argv[1], "r");
    if (!fp) {
		fprintf(stderr, "error opening file%s\n",
				argv[1]);
        exit(2);
	}
	//initialization
	Scatter_A_Lines(fp, rank, numprocs, status,
			&sub_matrices_a, &len_submat_a, &a);
	Scatter_B_Cols(rank, numprocs, status,
			&sub_matrices_b, &len_submat_b, &b);
	//finalizaion
	
	//MPI_Barrier(MPI_COMM_WORLD);
	Finalizer(
			rank, numprocs, sub_matrices_a,
			len_submat_a, a);
	MPI_Finalize();
	return 0;
}

/* .---------------. 
 * |  CONSTRUCTOR  |
 * .---------------.
 */
Matrix *new_Matrix(int height, int width) {
	Matrix *m = malloc(sizeof(Matrix));
	if (!m) {
		fprintf(stderr, "new_Matrix: Matrix malloc error\n");
		exit(4);
	}
	puts("after first malloc");
	m->width = width;
	m->height = height;
	m->size = width * height;
	puts("before 2nd malloc");
	m->data = calloc(m->size, sizeof(unsigned int));

	if (!m->data) {
		fprintf(stderr, "build_matrix: res->data malloc error\n");
		exit(5);
	}
	return m;
}

/* .---------------. 
 * |   DESTRUCTOR  |
 * .---------------.
 */
void Destroy_Matrix(Matrix *m) {
	//printf("======>debuuug: %p\n", (void*) m);
	//printf("======>debuuug: %p\n", (void*) m->data);
	if (m) {
		//puts("1");
		if (m->data)
			free(m->data);
		//puts("2");
		free(m);
		//puts("3");
	}
}
void Destroy_Matrix_Array(Matrix **arr, int len) {
	//print_matrix_list(arr, len);
	//printf("len=%d\n", len);
	for (int i = 0; i < len; i++) {
		//printf("i=%d\n", i);
		Destroy_Matrix(arr[i]);
	}
}
void Destroy_All_Matrices(int num, ...) {
	va_list valist; 
	Matrix *temp; 
    va_start(valist, num); 
    for (int i = 0; i < num; i++) { 
		temp = va_arg(valist, Matrix*);
		Destroy_Matrix(temp);
	}
    va_end(valist); 
}

/* .-----------. 
 * |  DISPLAY  |
 * .-----------.
 */

void print_raw_array(unsigned int *arr, int len) {
	for (int i = 0; i < len; i++)
		printf("%d  ", arr[i]);
	puts("\\");
}
void print_matrix_list(Matrix **m, int len) {
	for (int i = 0; i < len; i++) {
		print_matrix(m[i]);
	}
}
static void print_matrix_lineno(Matrix *m, int line) {
	for (int j = line * m->width; j < (line+1) * m->width ;j++) {
		printf("%4d ", m->data[j]);
	}
	printf("\t");
}
void print_matrix_list_bis(Matrix **m, int len) {
	for (int line = 0; line < (*m)->height; line++) {
		for (int mat_index = 0; mat_index < len; mat_index++)
			print_matrix_lineno(m[mat_index], line);
		puts("");
	}
	
}
void print_raw_matrix_list(Matrix **m, int len) {
	for (int i = 0; i < len; i++) {
		print_raw_array(m[i]->data, m[i]->size);
	}
}
void print_raw_matrix(Matrix *m) {
	int len = m->width * m->height;
	for (int i = 0; i < len; ++i)
		printf("%d ", m->data[i]);
	puts("");
}
void print_matrix(Matrix *m) {
	if (!m) {
		puts("print_matrix: null Matrix*");
		return;
	}
	int len = m->width * m->height;
	for (int i = 0; i < len; ++i) {
		if (i % m->width == 0)
			if (i == 0)
				printf("%5u ", m->data[i]);
			else
				printf("\n%5u ", m->data[i]);
		else
			printf("%5u ", m->data[i]);
	}
	puts("\n");
}

/* .----------------------------. 
 * |  MATRIX FILLING & COPYING  |
 * .----------------------------.
 */
void randomlyFillMatrix(Matrix *m){
	//Random seed
	srandom(time(0)+clock()+random());
	#pragma omp parallel for
	for(int i=0; i < m->size; i++){
			m->data[i] = rand() % 200 + 1;
	}
}
Matrix *matrixcpy(Matrix *src) {
	if (!src || !src->data)
		return NULL;
	Matrix *dest = new_Matrix(src->height, src->width);
	#pragma omp parallel for
	for(int i=0; i < src->size; i++)
		dest->data[i] = src->data[i];
	return dest;
}
Matrix *matrixcpy_bis(Matrix *dest, Matrix *src) {
	if (!src || !src->data)
		return NULL;
	#pragma omp parallel for
	for(int i=0; i < src->size; i++)
		dest->data[i] = src->data[i];
	return dest;
}
Matrix *matrixcpy_reverseIndex(Matrix *src) {
	if (!src || !src->data)
		return NULL;
	Matrix *res = new_Matrix(src->width, src->height);
	#pragma omp parallel for
	for(int i=0; i < src->height; i++)
		for(int j=0; j < src->width; j++)
			res->data[j*src->height + i] = 
				GET(src, i, j);
	return res;
}

/* .--------------------. 
 * |    FILE READING    |
 * .--------------------.
 */
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
Matrix *build_matrix(FILE *fp) {
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
		read = getline(&line, &len, fp);
	}
	if (line) free(line);
	if (save) free(save);
	return res;
}
/* .--------------------------. 
 * |    SEQUENTIAL PRODUCT    |
 * .--------------------------.
 */
Matrix *sequentialMultiply(Matrix *a, Matrix *b) {
	if (a->width != b->height) {
		fprintf(
				stderr, 
				"matrix product: a->width=%d b->height=%d\n",
				a->width, b->height);
		exit(7);
	}
	Matrix *res = new_Matrix(a->height, b->width);
	struct timeval t0, t1;
	gettimeofday(&t0, 0);
	for(int i=0; i < a->height; i++){
		for(int j=0; j < b->width; j++){
			SET(res, i, j, 0);
			for(int k=0; k< a->width; k++){
				GET(res, i, j) += 
					GET(a, i, k) * GET(b, k, j);
			}
		}
	}
	gettimeofday(&t1, 0);
	double elapsed = (t1.tv_sec-t0.tv_sec) * 1.0f + (t1.tv_usec - t0.tv_usec) / 1000000.0f;

	printf("sequentialMultiply time:  %f\n", elapsed);
	return res;
}
Matrix *sequentialMultiplyBySelf(Matrix *m) {
	Matrix *a, *b, *res;
	a = matrixcpy(m); b = matrixcpy(m);
	res = sequentialMultiply(a, b);
	Destroy_All_Matrices(2, a, b);
	return res;
}

/* .------------------------. 
 * |    PARALLEL PRODUCT    |
 * .------------------------.
 */
Matrix *parallelMultiply(Matrix *a, Matrix *b) {
	if (a->width != b->height) {
		fprintf(
				stderr, 
				"matrix product: a->width=%d b->height=%d\n",
				a->width, b->height);
		exit(8);
	}
	Matrix *res = new_Matrix(a->height, b->width);
	Matrix *convB = matrixcpy_reverseIndex(b);
	int i, j, k, iOff, jOff, sum;
	struct timeval t0, t1;
	gettimeofday(&t0, 0);

	#pragma omp parallel for private(i, j, k, iOff, jOff, sum) \
		shared(res)
	for(i=0; i < a->height; i++){
		iOff = i * a->width;
		for(j=0; j < b->width; j++){
			jOff = j * b->height;
			sum = 0;
			#pragma omp parallel for reduction(+: sum)
			for(k=0; k< a->width; k++){
				sum += a->data[iOff + k] * convB->data[jOff + k];
			}
			SET(res, i, j, sum);
		}
	}
	gettimeofday(&t1, 0);
	double elapsed =
		(t1.tv_sec-t0.tv_sec) * 1.0f + 
		(t1.tv_usec - t0.tv_usec) / 1000000.0f;
	printf("parallelMultiply time:  %f\n", elapsed);
	return res;
}

Matrix *parallelMultiplyBySelf(Matrix *m) {
	Matrix *a, *b, *res;
	a = matrixcpy(m); b= matrixcpy(m);
	res = parallelMultiply(a, b);
	Destroy_All_Matrices(2, a, b);
	return res;
}
/* .-------------. 
 * |   SCATTER   |
 * .-------------.
 */
void fill_submatrix(Matrix *m, Matrix *A, int *offset) {
	for (int i = 0; i < m->size; i++) {
		m->data[i] = A->data[*offset +i];
	}
	*offset += m->size;
}
void fill_submatrix_bis(Matrix *m, Matrix *B, int *offset_col) {
	int counter = 0;
	for (int line = 0; line < m->height; line++)
		for (int col = *offset_col;
				col < *offset_col + m->width;
				col++) {
			//printf("line=%d, col=%d\n", line, col);
			m->data[counter] = GET(B, line, col);
			counter += 1;
		}
	*offset_col += m->width;
}

/* .-----------. 
 * |   TESTS   |
 * .-----------.
 */
int Test_Equals(Matrix *a, Matrix *b) {
	if (a->height != b->height 
			|| a->width != b->width
			|| a->size != b->size)
		return 0;
	//TODO: parallelize this at the end
	//#pragma omp parallel for
	for (int i = 0; i < a->size; i++)
		if (a->data[i] != b->data[i]) {
			printf("i=%d\n", i);
			printf("a[i]=%d b[i]=%d\n",
					a->data[i], b->data[i]);
			return 0;
		}
	return 1;
}

/*
equals = Test_Equals(seq_res, parallel_res);
printf("seq_res == parallel_res: %s\n",
		equals ? "yes" : "no");
		*/
