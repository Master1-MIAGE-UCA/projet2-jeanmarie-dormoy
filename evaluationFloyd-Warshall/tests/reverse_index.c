#include<stdio.h>
#include<unistd.h>
#include<stdlib.h>

int flatA[500];
int flatB[500];
int flatA2[500];
int flatB2[500];

typedef struct matrix2D {
    int **data;
    int height;
    int width;
} matrix2D;

typedef struct Matrix {
    unsigned int *data;
    int width;
    int height;
    int size;
} Matrix;

#define GET(mat, i, j) mat->data[i * mat->width + j]
#define SET(mat, i, j, val) GET(mat, i, j) = val

matrix2D *new_matrix2D(int l, int c){
    int **tab = calloc(l, sizeof(int*)); 
    for (int i = 0; i< l; ++i) tab[i] = calloc(c, sizeof(int));
    int oua = 1;
    for (int i = 0; i <l; ++i) {
        for (int j=0; j< c ; ++j) {
            tab[i][j] = oua;
            ++oua;
        }
    }
    matrix2D *m = calloc(1, sizeof(matrix2D));
    m->data = tab;
    m->height = l;
    m->width = c;
    return m;
}

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

Matrix *convert(matrix2D *a){
	Matrix *res = new_Matrix(a->height, a->width);
	for(int i=0; i<a->height; i++){
		for(int j=0; j<a->width; j++){
			res->data[i * a->width + j] = a->data[i][j];
			//res->data[j * a->height + i] = a->data[i][j];
		}
	}
	return res;
}

Matrix *convert_reverse(matrix2D *a){
	Matrix *res = new_Matrix(a->height, a->width);
	for(int i=0; i<a->height; i++){
		for(int j=0; j<a->width; j++){
			//flatA[i * a->width + j] = a->data[i][j];
			res->data[j * a->height + i] = a->data[i][j];
		}
	}
	return res;
}

void convert_reverse2(matrix2D *a){
	for(int i=0; i<a->height; i++){
		for(int j=0; j<a->width; j++){
			flatA2[i * a->width + j] = a->data[i][j];
			flatB2[j * a->height + i] = a->data[i][j];
		}
	}
}

void convert2(int **matrixA, int** matrixB, int l, int c){
	for(int i=0; i<l; i++){
		for(int j=0; j<c; j++){
			flatA2[i * c + j] = matrixA[i][j];
			flatB2[j * l + i] = matrixB[i][j];
		}
	}
}

void printflat(int *a, int dim){
    for (int i = 0; i < dim; ++i) {
        printf("%d ", a[i]);
    }
    puts("");
}

void print_matrix2D(matrix2D *a){
    for (int i = 0; i < a->height; ++i) {
        for (int j= 0; j < a->width ; j++)
            printf("%5d ", a->data[i][j]);
        puts("");
    }
    puts("");
}


matrix2D *sequentialMultiply(matrix2D* a, matrix2D* b) {
    matrix2D *res = new_matrix2D(a->height, b->width);
	for(int i=0; i < a->height; i++){
		for(int j=0; j < b->width; j++){
			res->data[i][j] =0;
			//printf("i= %d, j= %d\n", i, j);
			for(int k=0; k< a->width; k++){
				res->data[i][j] += 
					a->data[i][k] * b->data[k][j];
			}
			//printf("res[i][j]= %d\n", res->data[i][j]);
		}
	}
	return res;
}

Matrix *parallelMultiply(Matrix *a, Matrix *b) {
	int i, j, k, iOff,sum; int jOff;
	int a_index, b_index;
	if (a->width != b->height) {
		fprintf(
				stderr, 
				"matrix product: a->width=%d b->height=%d\n",
				a->width, b->width);
		exit(7);
	}
	Matrix *res = new_Matrix(a->height, b->width);

	for(i=0; i < a->height; i++){
		iOff = i * a->width;
		for(j=0; j < b->width; j++){
			jOff = j * b->height;
			//printf("j= %d\tjOff=%d\n", j, jOff);
			//res->data[iOff + j] = 0;
			sum = 0;
			for(k=0; k< a->width; k++){
				//a_index = iOff + k + i;
				//b_index = jOff + k + j;
				a_index = iOff + k;
				b_index = jOff + k;
				//if (i > 0)
				//	a_index += 1;
				if (1)
					b_index = jOff + j + k - j;
				sum += 
					a->data[a_index] * b->data[b_index];
				/*printf(
						"sum+=\n\ta[ %d] * b[ %d]\n",
						a_index, b_index);*/
			}
			SET(res, i, j, sum);
			//printf("iOff=%d j=%d i=%d\n", iOff, j, i);
			//printf("data[ res_index=%d] = %d\n", res_index, sum);
			//GET(res, i, j) = sum;
		}
	}
	return res;
}

void pretty_matrix(Matrix *m) {
	int len = m->width * m->height;
	for (int i = 0; i < len; ++i) {
		if (i % m->width == 0)
			printf("\n%5u ", m->data[i]);
		else
			printf("%5u ", m->data[i]);
	}
	puts("");
}
int main() {
    matrix2D *A = new_matrix2D(1, 16), *res = NULL;
	Matrix *convA, *convB, *convRes;

    print_matrix2D(A);
    convA = convert(A);
	puts("convA:");
    pretty_matrix(convA);
	puts("---------------------------------");
    
    matrix2D *B = new_matrix2D(16, 8);
    print_matrix2D(B);
    convB = convert_reverse(B);
	puts("convB:");
    pretty_matrix(convB);

    puts("---------------------------------");
	puts("EXPECTED:");
    res = sequentialMultiply(A, B);
    print_matrix2D(res);

	puts("GOT:");
	free(res);
	convRes = parallelMultiply(convA, convB);
	pretty_matrix(convRes);
	free(convRes);	
}
