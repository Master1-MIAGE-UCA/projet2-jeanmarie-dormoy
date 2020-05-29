#include <omp.h>
#include <mpi.h>
#include <limits.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/time.h>
#include <stdarg.h>

#define LINE_SIZE 500
#define GET(mat, i, j) mat->data[i * mat->width + j]
#define SET(mat, i, j, val) GET(mat, i, j) = val

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

/* Constructors */
Matrix *new_Matrix(int height, int width);
Matrix *new_Matrix_Crescendo(int height, int width, int base);
Matrix *build_matrix(FILE *fp);
Local_Result *new_Local_Result_Default();
Local_Result *new_Local_Result(int index, Matrix *m);
/* Destructors  */
void Destroy_Matrix(Matrix *m);
void Destroy_All_Matrices(int num, ...);
void Destroy_Matrix_Array(Matrix **arr, int len);
void Destroy_Local_Result_Array(Local_Result *local_res, int len);
void Destroy_Local_Result_List(
		Local_Result **local_res_list, int len);
void Destroy_Local_Result_Matrix(
		Local_Result **local_res_list, int len);

/* Display Matrix or unsigned int array */
void print_raw_array(unsigned int *arr, int len);
void print_raw_matrix(Matrix *m);
void print_matrix(Matrix *m);
void print_local_result(Local_Result *local_res, int rank);
void print_local_result_list(
		Local_Result *local_res, int len, int rank);
void print_local_result_matrix(Local_Result **local_res, 
		int len, int rank);
/* Scatter Debug			*/
void print_matrix_list(Matrix **m, int len);
void print_matrix_list_bis(Matrix **m, int len);
void print_raw_matrix_list(Matrix **m, int len);

/* Matrix filling & copying				*/
void randomlyFillMatrix(Matrix *m);
unsigned int* unsigned_int_cpy(unsigned int *dest, unsigned int *src,
		int len);
Matrix *matrixcpy(Matrix *src);
Matrix *matrixcpy_bis(Matrix *dest, Matrix *src);
Matrix *matrixcpy_reverseIndex(Matrix *src);
void Fill_Local_Result_With(Local_Result *allocated,
		Local_Result *src);
Local_Result *Local_Result_cpy(Local_Result *src);

/* IPC Communication					*/
void Propagate_Number(int rank, int numprocs,
		unsigned int *nb, int round);
void My_MPI_Send(Matrix *m, int recipient, int round);
void My_MPI_Recv(Matrix **local_a_submatrix, int sender, int round);
void Transmit_SubMatrix(int iter, int numprocs, 
		int rank, Matrix **local_submatrix, int round);
void MPI_Recv_Local_Result(Local_Result **local_res, int sender,
		int round);
void MPI_Send_Local_Result(Local_Result *local_res,
		int recipient, int round);

/* File Reading							*/
int first_pass(char *s);
void fill_matrix_with_line(unsigned int *array, int *index,
		char *line);
void File_Reading(int argc, char *argv[], FILE **fp,
		Matrix **input_mat);

/* Initialization						*/
void Transform_A_Into_W(Matrix *a);
void Initialization_Local_Result(Local_Result **local_res,
		int numprocs);

/* Scatter								*/
void fill_submatrix(Matrix *m, Matrix *A, int *offset);
void fill_submatrix_bis(Matrix *m, Matrix *B, int *offset);
Matrix** Explode_A_Into_Lines(Matrix *A, int numprocs, int *len);
Matrix** Explode_B_Into_Columns(Matrix *B, int numprocs, int *len);
void Scatter_A_Lines(
		int rank, int numprocs,
		Matrix ***sub_matrices_a, int *len_submat_a,
		Matrix *a, Matrix **local_a_submatrix, int *round);
void Scatter_B_Cols(
		int rank, int numprocs, Matrix ***sub_matrices_b,
		int *len_submat_b, Matrix *b, Matrix **local_b_submatrix,
		int round);

/* Specific Product with min/+			*/
Matrix *parallelMultiplyBySelf(Matrix *m);
Matrix *parallelMultiply(Matrix *a, Matrix *b);

/* Circulation of Local A SubMatrices	*/
void Make_Local_A_Submatrices_Circulate(int rank, int numprocs,
		Matrix **local_a_submatrix, int round);

/* Local Computations in each Processus */
void Do_Local_Computation(int rank, int numprocs,
		Matrix *local_a_submatrix, Matrix *local_b_submatrix,
		Local_Result *local_res, int iter);
void Local_Computation_Each_Proc(
		int numprocs, int rank, Matrix **local_a_submatrix,
		Matrix *local_b_submatrix, Local_Result *local_res,
		int *round);

/* Tests								*/
int Test_Equals(Matrix *a, Matrix *b);

void Initialization_Local_Result_List(
		Local_Result ***local_res_list, int numprocs) {
	Local_Result *temp;
	*local_res_list = calloc(numprocs, sizeof(Local_Result*));
	if (!*local_res_list) {
		fprintf(stderr, "local_res_list: calloc error\n");
		exit(14);
	}
	for (int i = 0; i < numprocs; i++) {
		temp = calloc(numprocs, sizeof(Local_Result));
		if (!temp) {
			fprintf(stderr, 
					"local_res_list[%d]: calloc error\n", i);
			exit(15);
		}
		temp->mat = NULL;
		(*local_res_list)[i] = temp;
	}
}
void Gather_Local_Results(
		int rank, int numprocs, Local_Result **local_res_list,
		Local_Result *local_res, int *round) {
	Local_Result *temp;
	switch(rank) {
		case 0:
			for (int j = 0; j < numprocs; j++)
				Fill_Local_Result_With(local_res_list[0] + j,
						local_res + j);
			for (int i = 1; i < numprocs; i++, (*round)++) {
				for (int j = 0; j < numprocs; j++) {
					//printf("rank 0 before:\n");
					//print_local_result_list(
					//		local_res_list[i], numprocs, rank);
					MPI_Recv_Local_Result(&temp, 1, *round);
					//print_local_result(temp, rank);
					Fill_Local_Result_With(
							local_res_list[i] + temp->index, temp);
					//printf("rank 0 after:\n");
					//print_local_result_list(
					//		local_res_list[i], numprocs, rank);
					free(temp);
				}
			}
			break;
		default:
			for (int i = 0; i < numprocs - 1; i++, (*round)++) {
				if (rank <= numprocs - 1 -i) {	
					for (int j= 0; j < numprocs; j++) {
						printf("rank %d sends:\n", rank);
						print_local_result(local_res +j, rank);
						MPI_Send_Local_Result(local_res + j,
								rank - 1, *round);
					}
				}
				if (rank < numprocs - 1 -i) {
					for (int j =0; j < numprocs; j++) {
						MPI_Recv_Local_Result(&temp, rank +1, *round);
						Fill_Local_Result_With(local_res + j,
							temp);	
						free(temp);
					}	
				}
			}
			break;
	}
}
unsigned int* Get_Lineno(Matrix *m, int line) {
	unsigned int* res = calloc(m->width, sizeof(unsigned int));
	if (!res) {
		fprintf(stderr, "Get_Lineno: calloc error\n");
		exit(16);
	}
	#pragma omp parallel for
	for (int col = 0; col < m->width; col++)
		res[col] = GET(m, line, col);
	return res;
}
void Fill_Matrix_With_Results(Matrix *to_fill, 
		Local_Result **local_res_list, int numprocs) {
	Local_Result temp;
	unsigned int *tmp_data;
	int count = 0, nb_lines;

	for (int index = 0; index < numprocs; index++) {
		//printf("\nindex=%d ", index);
		nb_lines = local_res_list[0][index].mat->height;
		//printf(" nb_lines=%d ", nb_lines);
		for (int line = 0; line < nb_lines; line++) {
			//printf("line=%d ", line);
			for (int col = 0; col < numprocs; col++) {
				//printf("col= %d ", col);
				temp = local_res_list[col][index];
				//printf("temp=\n");
				print_matrix(temp.mat);
				tmp_data = Get_Lineno(temp.mat, line);
				//#pragma omp parallel for
				for (int i = 0; i < temp.mat->width; i++) {
					to_fill->data[count] = tmp_data[i];
					count++;
				}
				free(tmp_data);
			}
		}
	}
}
void Do_Multiply(
		int rank, int numprocs, Matrix *a, Matrix *b,
		Matrix ***sub_matrices_a, Matrix ***sub_matrices_b,
		int *len_submat_a, int *len_submat_b,
		Matrix **local_a_submatrix, Matrix **local_b_submatrix,
		Local_Result **local_res_list, Local_Result **local_res,
		int *round)
{
	Initialization_Local_Result(local_res, numprocs);
	//print_local_result_matrix(local_res_list, numprocs, rank);

	Scatter_A_Lines(rank, numprocs, sub_matrices_a, len_submat_a,
			a, local_a_submatrix, round);
	(*round)++;
	//printf("rank %d has A submatrix:\n", rank);
	//print_matrix(local_a_submatrix);

	Scatter_B_Cols(rank, numprocs, sub_matrices_b, len_submat_b,
			b, local_b_submatrix, *round);
	(*round)++;
	//printf("rank %d has B submatrix:\n", rank);
	//print_matrix(local_b_submatrix);
	puts("coucou 1");	
	Local_Computation_Each_Proc(numprocs, rank,
		local_a_submatrix, *local_b_submatrix, *local_res, round);
	//print_local_result_matrix(local_res_list, numprocs, rank);
	puts("coucou 2");
	(*round)++;
	Gather_Local_Results(rank, numprocs, local_res_list,
		   *local_res, round);
}
int main(int argc, char *argv[]) {
    int rank, numprocs;
	FILE *fp = NULL;
	int len_submat_a = 0, len_submat_b = 0;
	Local_Result *local_res = NULL,
				 **local_res_list = NULL;
	Matrix **sub_matrices_a = NULL, 
		   *a = NULL;
	Matrix *local_a_submatrix = NULL,
		   *local_b_submatrix = NULL, 
		   *b = NULL,
		   *w = NULL,
		   **sub_matrices_b = NULL;
	Matrix *input_matx = NULL, *res = NULL;
    unsigned int times;
	int round = 0;
	MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	/* Processus 0 Initialization */	
	if (rank == 0) {	
		File_Reading(argc, argv, &fp, &input_matx);
		times = input_matx->height;
		Transform_A_Into_W(input_matx);
		w = matrixcpy(input_matx);
		Initialization_Local_Result_List(
				&local_res_list, numprocs);
	}
	Propagate_Number(rank, numprocs, &times, round);
	round++;
	printf("times=%d\n", times);
	while (times) {
		Do_Multiply(rank, numprocs, w, input_matx,
				&sub_matrices_a, &sub_matrices_b,
				&len_submat_a, &len_submat_b, &local_a_submatrix,
				&local_b_submatrix, local_res_list,
				&local_res, &round);
		puts("------------------------------------");
		//print_local_result_matrix(local_res_list, numprocs, rank);
		Fill_Matrix_With_Results(w, local_res_list,
				numprocs);			
		print_matrix(w);
		printf("rank= %d\n", rank);
		times--;
	}

	/* Finalizer */
	if (rank == 0) {
		Destroy_All_Matrices(4, a, b, w, input_matx, res);
		Destroy_Local_Result_Matrix(local_res_list, numprocs);
		if (fp) fclose(fp);
	}
	Destroy_All_Matrices(2, 
			local_a_submatrix, local_b_submatrix);
	Destroy_Matrix_Array(sub_matrices_a, len_submat_a);
	Destroy_Matrix_Array(sub_matrices_b, len_submat_b);
	Destroy_Local_Result_Array(local_res, numprocs);
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
Matrix *new_Matrix_Crescendo(int height, int width, int base) {
	Matrix *res = new_Matrix(height, width);
	for (int i = 0; i < res->size; i++, base++)
		res->data[i] = base;
	return res;
}
Local_Result *new_Local_Result_Default() {
	Local_Result *res = calloc(1, sizeof(Local_Result));
	if (!res) {
		fprintf(stderr,
				"new_Local_Result: Local_Result calloc error\n");
		exit(15);
	}
	return res;
}
Local_Result *new_Local_Result(int index, Matrix *m) {
	Local_Result *res = calloc(1, sizeof(Local_Result));
	if (!res) {
		fprintf(stderr,
				"new_Local_Result: Local_Result calloc error\n");
		exit(15);
	}
	res->index = index;
	res->mat = matrixcpy(m);
	return res;
}
/* .---------------. 
 * |   DESTRUCTOR  |
 * .---------------.
 */
void Destroy_Matrix(Matrix *m) {
	if (m) {
		if (m->data)
			free(m->data);
		free(m);
	}
}
void Destroy_Matrix_Array(Matrix **arr, int len) {
	if (arr)
		for (int i = 0; i < len; i++) {
			Destroy_Matrix(arr[i]);
		}
}
void Destroy_Local_Result_Array(
		Local_Result *local_res, int len) {
	if (local_res) {
		for (int i = 0; i < len; i++) {
			Destroy_Matrix(local_res[i].mat);
		}
		free(local_res);
	}
}
void Destroy_Local_Result_List(
		Local_Result **local_res_list, int len) {
	if (local_res_list) {
		for (int i = 0; i < len; i++) {
			Destroy_Matrix(local_res_list[i]->mat);
			free(local_res_list[i]);
		}
		free(local_res_list);
	}
}
void Destroy_Local_Result_Matrix(
		Local_Result **local_res_list, int len) {
	if (local_res_list) {
		for (int i = 0; i < len; i++) {
			Destroy_Local_Result_Array(local_res_list[i], len);
		}
		free(local_res_list);
	}
}
void Destroy_All_Matrices(int num, ...) {
	va_list valist; 
	Matrix *temp; 
    va_start(valist, num);
    for (int i = 0; i < num; i++) { 
		temp = va_arg(valist, Matrix*);
		//printf("i=%d\n", i);
		//printf("matrix @= %p\n", (void*) temp);
		if (temp)
			Destroy_Matrix(temp);
	}
    va_end(valist); 
}

/* .-----------. 
 * |  DISPLAY  |
 * .-----------.
 */

void print_local_result(Local_Result *local_res, int rank) {
	if (local_res) {
		printf("rank= %d\n", rank);
		printf("local_res.index = %d\n", local_res->index);
		puts("local_res.mat = ");
		print_matrix(local_res->mat);
	}
}
void print_local_result_list(Local_Result *local_res,
		int len, int rank) {
	if (local_res)
		for (int i = 0; i < len; i++) {
			print_local_result(local_res + i, rank);
		}
}
void print_local_result_matrix(Local_Result **local_res, 
		int len, int rank) {
	if (local_res)
		for (int i = 0; i < len; i++)
			print_local_result_list(local_res[i], len, rank);
}
void print_raw_array(unsigned int *arr, int len) {
	//printf("print_raw_array arr=%p len=%d\n", arr, len);
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
		puts("print_matrix: null Matrix*\n");
		return;
	}
	//printf("h=%d w=%d\n", m->height, m->width);
	int len = m->width * m->height;
	for (int i = 0; i < len; ++i) {
		if (m->data[i] == UINT_MAX)
			if (i % m->width == 0)
				if (i == 0)
					printf("    i ");
				else
					printf("\n    i ");
			else
				printf("    i ");
		else
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
Local_Result *Local_Result_cpy(Local_Result *src) {
	if (!src || !src->mat || !src->mat->data)
		return NULL;
	Local_Result *dest = new_Local_Result(src->index, src->mat);
	return dest;
}
void Fill_Local_Result_With(Local_Result *allocated,
		Local_Result *src) {
	if (!src || !src->mat || !src->mat->data)
		return;
	if (allocated == NULL)
		puts("allocated is NULL");
	else {
		allocated->index = src->index;
		if (allocated->mat) Destroy_Matrix(allocated->mat);
		allocated->mat = matrixcpy(src->mat);
	}

}
void randomlyFillMatrix(Matrix *m){
	//Random seed
	srandom(time(0)+clock()+random());
	#pragma omp parallel for
	for(int i=0; i < m->size; i++){
			m->data[i] = rand() % 200 + 1;
	}
}
unsigned int* unsigned_int_cpy(
		unsigned int *dest, unsigned int *src, int len) {
	#pragma omp parallel for
	for (int i = 0; i < len; i++)
		dest[i] = src[i];
	return dest;	
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
	if (!dest) puts("NULL dest!");
	if (!src || !src->data || !dest || !dest->data)
		return NULL;
	dest->size = src->size;
	dest->width = src->width;
	dest->height = src->height;
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
		//printf("|%s| ", token);
		if (strcmp(token, "\n") != 0)
			len +=1;
		token = strtok(NULL, " ");
	}
	//puts("");
	return len;
}
void fill_matrix_with_line(
		unsigned int *array, int *index, char *line) {
	char *token;
	token = strtok(line, " ");
	while (token) {
		//printf("|%s| ", token);
		if (strcmp(token, "\n") != 0) {
			array[*index] = atoi(token);
			*index += 1;
		}
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
	res->size = n * n;
	res->data = calloc(res->size, sizeof(unsigned int));
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
void File_Reading(int argc, char *argv[], 
		FILE **fp, Matrix **input_mat) {
	if (argc != 2) {
		fprintf(stderr,
				"usage:\n\tmpirun -np X ./Dormoy input.txt");
		exit(1);
	}
	if (argv[1])
		*fp = fopen(argv[1], "r");
    if (!*fp) {
		fprintf(stderr, "error opening file%s\n",
				argv[1]);
        exit(2);
	}
	*input_mat = build_matrix(*fp);
	if (!*input_mat) {
		fprintf(stderr, "var mat: build_matrix ret NULL\n");
		exit(6);
	}
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
void Transform_A_Into_W(Matrix *a) {
	/* wij = 0 if i = j
	 * wij = weight of  (i,j) if there is an edge between i and j
	 * wij = +inf otherwise
	 * */
	for (int line = 0; line < a->height; line++)
		for (int col = 0; col < a->width; col++)
			if (line == col)
				SET(a, line, col, 0);
			else if (GET(a, line, col) == 0)
				SET(a, line, col, UINT_MAX);
}
unsigned int Check_Infinity(unsigned int a, unsigned int b) {
	unsigned int t = a + b;
	if (t < a || t < b)
		return UINT_MAX;
	return a + b;
}
/*
 * Compute W^n by replacing the multiplication operation with an 
 * addition and the addition with a min. 
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
	int i, j, k, iOff, jOff, current_min, a_oprd, b_oprd;
	unsigned int _min;
	struct timeval t0, t1;
	gettimeofday(&t0, 0);
	#pragma omp parallel for private(i, j, k, iOff, jOff, \
			current_min, _min, a_oprd, b_oprd) shared(res)
	for(i=0; i < a->height; i++){
		iOff = i * a->width;
		for(j=0; j < b->width; j++){
			jOff = j * b->height;
			current_min = UINT_MAX;
			_min = UINT_MAX;
			#pragma omp parallel for reduction(min: _min)
			for(k=0; k< a->width; k++){
				a_oprd = a->data[iOff + k] ;
				b_oprd = convB->data[jOff + k];
				current_min = Check_Infinity(a_oprd, b_oprd);
				if (current_min < _min)
					_min = current_min;
			}
			SET(res, i, j, _min);
		}
	}
	gettimeofday(&t1, 0);
	double elapsed =
		(t1.tv_sec-t0.tv_sec) * 1.0f + 
		(t1.tv_usec - t0.tv_usec) / 1000000.0f;
	printf("parallelMultiply time:  %f\n", elapsed);
	free(convB);
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
			m->data[counter] = GET(B, line, col);
			counter += 1;
		}
	*offset_col += m->width;
}
Matrix** Explode_A_Into_Lines(Matrix *A, int numprocs, int *len){ 
	int lines_A, remain, lines_remaining, divider, offset = 0;
	lines_A	= A->height;
	divider = lines_A / numprocs;
	remain = lines_A % numprocs;
	Matrix **matrix_list = NULL;

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
					"Explode_B_Into_Columns: matrix_list \
					calloc error");
			exit(9);
		}
		for (int i = 0; i < numprocs; i++) {
			//printf("b->height=%d divider=%d\n", B->height, divider);
			matrix_list[i] = new_Matrix(B->height, divider);
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
//TODO remove status parameter
void Scatter_A_Lines(
		int rank, int numprocs,
		Matrix ***sub_matrices_a, int *len_submat_a,
		Matrix *a, Matrix **local_a_submatrix, int *round) {
	Matrix *ptr_m;
	//puts("begin scatter A Lines");
	//printf("rank %d local_a_submatrix: %p\n", rank,
	//		(void*) local_a_submatrix);
	*local_a_submatrix = NULL;//ERROR here	
	switch(rank) {
		case 0:	
			//printf("a: %p\n", (void*) a);
			puts("a:");
			print_matrix(a);
			*sub_matrices_a = Explode_A_Into_Lines(
					a, numprocs, len_submat_a);
			for (int i = numprocs - 1; i > 0; i--) {
				ptr_m = (*sub_matrices_a)[i];
				My_MPI_Send(ptr_m, (rank + 1) % numprocs, *round);
			}
			*local_a_submatrix = matrixcpy((*sub_matrices_a)[0]);
			break;
		default:
			for (int i = 0; i < numprocs - rank; i++) {
				Transmit_SubMatrix(i, numprocs,
						rank, local_a_submatrix, *round);
			}
			break;
	}
	//printf("rank=%d end Scatter_A_Lines\n", rank);
}
void Scatter_B_Cols(
		int rank, int numprocs, Matrix ***sub_matrices_b,
		int *len_submat_b, Matrix *b, Matrix **local_b_submatrix,
		int round) {
	Matrix *ptr_m;
	*local_b_submatrix = NULL;	
	switch(rank) {
		case 0:	
			//*b = new_Matrix(5, 5);
			//randomlyFillMatrix(*b);
			puts("b:");
			print_matrix(b);
			*sub_matrices_b = Explode_B_Into_Columns(
					b, numprocs, len_submat_b);
			//printf("len=%d\n", *len_submat_b);
			//print_matrix_list_bis(*sub_matrices_b, *len_submat_b);
			for (int i = numprocs - 1; i > 0; i--) {
				ptr_m = (*sub_matrices_b)[i];
				My_MPI_Send(ptr_m, (rank + 1) % numprocs, round);
			}
			*local_b_submatrix = matrixcpy((*sub_matrices_b)[0]);
			break;
		default:
			for (int i = 0; i < numprocs - rank; i++) {
				Transmit_SubMatrix(i, numprocs, 
						rank, local_b_submatrix, round);
			}
			break;
	}
}
/* .-------------------------. 
 * |    IPC COMMUNICATION    |
 * .-------------------------.
 */
void Propagate_Number(int rank, int numprocs,
		unsigned int *nb, int round) {
	int buff[1], size_msg;
	MPI_Status status;
	switch(rank) {
		case 0:
			*buff = *nb;
			MPI_Send(buff, 1, MPI_INT, 1, round, MPI_COMM_WORLD);
			break;
		default:
			MPI_Probe(rank - 1, round, MPI_COMM_WORLD, &status);
			MPI_Get_count(&status, MPI_INT, &size_msg);
			if (size_msg == 1) {
				MPI_Recv(buff, 1, MPI_INT, rank - 1, round,
						MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				*nb = *buff;
				if (rank != numprocs - 1)
					MPI_Send(buff, 1, MPI_INT, (rank + 1) % numprocs,
							round, MPI_COMM_WORLD);
			}
			break;
	}	
}
void Transmit_SubMatrix(
		int iter, int numprocs, 
		int rank, Matrix **local_submatrix, int round) {
	MPI_Status status;
	int size_msg, dimensions[2], sender = rank - 1;
	unsigned int *buffer;

	MPI_Probe(sender, round, MPI_COMM_WORLD, &status);
	MPI_Get_count(&status, MPI_INT, &size_msg);
	if (size_msg == 2) {
		MPI_Recv(dimensions, size_msg, MPI_INT,
				sender, round, MPI_COMM_WORLD,
				MPI_STATUS_IGNORE);
	}
	MPI_Probe(sender, round, MPI_COMM_WORLD, &status);
	MPI_Get_count(&status, MPI_INT, &size_msg);
	buffer = calloc(size_msg, sizeof(unsigned int));
	if (!buffer) {
		fprintf(stderr, "buffer: calloc error\n");
		exit(10);
	}
	MPI_Recv(buffer, size_msg, MPI_INT, sender, round,
			MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	//printf("received at rank: %d\n", rank);
	//print_raw_array(buffer, size_msg);
	if (iter == numprocs - rank - 1) {
		*local_submatrix = new_Matrix(
			dimensions[0], dimensions[1]);
		unsigned_int_cpy(
				(*local_submatrix)->data, 
				buffer, size_msg);
		//printf("%d finally got:\n", rank);
		//print_matrix(*local_submatrix);
	}
	if (rank != numprocs - 1) {
		MPI_Send(dimensions, 2, MPI_INT, (rank + 1) % numprocs, 
				round, MPI_COMM_WORLD);
		MPI_Send(buffer, size_msg, MPI_INT, (rank + 1) % numprocs,
				round, MPI_COMM_WORLD);
	}
	free(buffer);
}
void MPI_Send_Local_Result(Local_Result *local_res,
		int recipient, int round) {
	unsigned int index[1];
	index[0] = local_res->index;
	MPI_Send(index, 1, MPI_INT, recipient, round, MPI_COMM_WORLD);
	My_MPI_Send(local_res->mat, recipient, round);
}
void My_MPI_Send(Matrix *m, int recipient, int round) {
	unsigned int dimensions[2];
	dimensions[0] = m->height;
	dimensions[1] = m->width;	
	MPI_Send(dimensions, 2, MPI_INT, recipient,
			round, MPI_COMM_WORLD);
	MPI_Send(m->data, m->size, MPI_INT, recipient,
			round, MPI_COMM_WORLD);
}
/*
void MPI_Recv_Local_Result_List(int rank, int numprocs,
		Local_Result ***local_res_list, int sender, int round) {
	Local_Result *temp;
	MPI_Recv_Local_Result(temp, sender, round);
}*/
void MPI_Recv_Local_Result(Local_Result **local_res, int sender,
		int round) {
	unsigned int index[1];
	MPI_Status status;
	int size_msg;
	Local_Result *received = new_Local_Result_Default();

	MPI_Probe(sender, round, MPI_COMM_WORLD, &status);
	MPI_Get_count(&status, MPI_INT, &size_msg);
	if (size_msg == 1) {
		MPI_Recv(index, 1, MPI_INT, sender, round, MPI_COMM_WORLD,
				MPI_STATUS_IGNORE);
		received->index = index[0];
		My_MPI_Recv(&(received->mat), sender, round);
		*local_res = received;
	}
}
void My_MPI_Recv(Matrix **local_a_submatrix, int sender, int round) {
	unsigned int dimensions[2];
	unsigned int *buffer;
	int size_msg;
	MPI_Status status;
	/* Get Matrix' Dimensions */	
	MPI_Probe(sender, round, MPI_COMM_WORLD, &status);
	MPI_Get_count(&status, MPI_INT, &size_msg);
	if (size_msg == 2) {
		MPI_Recv(dimensions, 2, MPI_INT, sender, round, 
				MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	/* Get Matrix Raw Content */
	MPI_Probe(sender, round, MPI_COMM_WORLD, &status);
	MPI_Get_count(&status, MPI_INT, &size_msg);
	buffer = calloc(size_msg, sizeof(unsigned int));
	if (!buffer) {
		fprintf(stderr, "buffer: calloc error\n");
		exit(10);
	}
	
	MPI_Recv(buffer, size_msg, MPI_INT, sender, round, 
			MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	if (*local_a_submatrix)
		Destroy_Matrix(*local_a_submatrix);
	*local_a_submatrix = new_Matrix(
			dimensions[0], dimensions[1]);
	unsigned_int_cpy(
			(*local_a_submatrix)->data, 
			buffer, size_msg);
}

/* .------------------------------------------. 
 * |    CIRCULATION OF LOCAL A SUBMATRICES    |
 * .------------------------------------------.
 */
void Make_Local_A_Submatrices_Circulate(
		int rank, int numprocs, Matrix **local_a_submatrix, 
		int round) {
	Matrix *save;
	int sender = rank - 1;
	switch(rank % 2) {
		case 0:
			if (rank == 0) sender = numprocs - 1;
			My_MPI_Send(*local_a_submatrix,
					(rank + 1) % numprocs, round);
			My_MPI_Recv(local_a_submatrix, sender, round);
			break;	
		default:
			save = matrixcpy(*local_a_submatrix);
			My_MPI_Recv(local_a_submatrix, sender, round);
			My_MPI_Send(save, (rank + 1) % numprocs, round);
			break;
	}
}
/* .-------------------------. 
 * |    LOCAL COMPUTATIONS   |
 * .-------------------------.
 */
void Do_Local_Computation(int rank, int numprocs,
		Matrix *local_a_submatrix, Matrix *local_b_submatrix,
		Local_Result *local_res, int iter) {
	Matrix *res = parallelMultiply(
			local_a_submatrix, local_b_submatrix);
	int index = rank;
	while (iter) {
		if (index) index--;
		else index = numprocs - 1;
		iter--;
	}
	local_res[index].index = index;
	local_res[index].mat = res;
}
void Local_Computation_Each_Proc(
		int numprocs, int rank, Matrix **local_a_submatrix,
		Matrix *local_b_submatrix, Local_Result *local_res,
		int *round) {
	for (int i = 0; i < numprocs; i++, (*round)++) {
		Do_Local_Computation(rank, numprocs, *local_a_submatrix,
				local_b_submatrix, local_res, i);
		Make_Local_A_Submatrices_Circulate(
				rank, numprocs, local_a_submatrix, *round);
	}
}
void Initialization_Local_Result(
		Local_Result **local_res, int numprocs) {
	*local_res = calloc(numprocs, sizeof(Local_Result));
	if (!*local_res) {
		fprintf(stderr, "local_res: calloc error\n");
		exit(13);
	}
	//puts("ending init Local_Result");
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
