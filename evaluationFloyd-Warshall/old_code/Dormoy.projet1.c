/********************************************************************/
/*					AUTHOR: Jean-Marie DORMOY						*/
/********************************************************************/
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <limits.h>

#define _XOPEN_SOURCE 500
#define MAX(a,b) ((a) > (b) ? a : b)
#define MIN(a,b) ((a) < (b) ? a : b)
#define BUF_SIZE	100000

struct Compare { int val; int index; };
#pragma omp declare reduction(maximum_ind_start : struct Compare : \
		omp_out = omp_in.val > omp_out.val ? omp_in : \
		omp_in.val == omp_out.val && omp_in.index < omp_out.index ? omp_in : omp_out)

//#ifdef _OPENMP
//	#pragma omp parallel sections
//#endif
static int arr_size;

void print_array(int *arr) {
	for(int i = 0; i < arr_size; i++) 
		if (i == arr_size -1)
			printf("%d\n", arr[i]);
		else
			printf("%d ", arr[i]); 
}

void intcpy(int *source, int *dest){
	int *p_source = source;
	while (source < p_source + arr_size)
		*dest++ = *source++;
}

void sum_prefix(int *src, int *dest) {
	int power = log2(arr_size - 1);
	for (int d = 0; d <= power; d++) {
		int incr = pow(2, d+1);
		#pragma omp parallel for
		for (int i = 0; i <= arr_size -1; i += incr) {
			dest[i + (int) pow(2, d+1) -1] += dest[i + (int) pow(2, d) -1];
		}
		//print_array(dest);
	}
	dest[arr_size - 1] = 0;
	//puts("------------");
	//print_array(dest);
	for (int d = power; d >= 0; d--) {
		int incr = pow(2, d+1);
		#pragma omp parallel for
		for (int i = 0; i <= arr_size -1; i += incr) {
			int temp = dest[i + (int) pow(2, d) -1];
			dest[i + (int) pow(2, d) - 1] = dest[i+ (int) pow(2, d+1) -1];
			dest[i + (int) pow(2, d+1) -1] = temp + dest[i + (int) pow(2, d+1) -1];
		}
		//print_array(dest);
	}
	for (int i = 0; i < arr_size-1; i++)
		dest[i] = dest[i+1];
	//print_array(dest);
	dest[arr_size-1] = dest[arr_size-2] + src[arr_size-1];
	//print_array(dest);
}

void sum_suffix(int *src, int *dest) {
	int power = log2(arr_size -1);
	for (int d = 0; d <= power; d++) {
		int incr = pow(2, d+1);
		#pragma omp parallel for
		for (int i = 0; i <= arr_size -1; i += incr) {
			//dest[i + (int) pow(2, d) -1] += dest[i + (int) pow(2, d+1) -1];
			dest[i] = dest[i] + dest[i + (int) pow(2,d)];
		}
		//print_array(dest);
	}
	dest[0] = 0;
	//puts("------------");
	for (int d = power; d >= 0; d--) {
		int incr = pow(2, d+1);
		#pragma omp parallel for
		for (int i = 0; i <= arr_size -1; i += incr) {
			int temp = dest[i+ (int) pow(2, d)];
			dest[i + (int) pow(2, d)] = dest[i];
			dest[i] = dest[i] + temp;
		}
		//print_array(dest);
	}
	for (int i = arr_size -1; i > 0; i--)
		dest[i] = dest[i-1];
	//print_array(dest);
	dest[0] = dest[1] + src[0];
	//print_array(dest);
}

void max_prefix(int *src, int *dest) {
	int power = log2(arr_size - 1);
	for (int d = 0; d <= power; d++) {
		int incr = pow(2, d+1);
		#pragma omp parallel for
		for (int i = 0; i <= arr_size -1; i += incr) {
			int a = dest[i + (int) pow(2, d) -1];
			int b = dest[i + (int) pow(2, d+1) -1];
			dest[i + (int) pow(2, d+1) -1] = MAX(a, b);
		}
		//print_array(dest);
	}
	dest[arr_size - 1] = INT_MIN;
	//puts("------------WOUAH");
	//print_array(dest);
	for (int d = power; d >= 0; d--) {
		int incr = pow(2, d+1);
		#pragma omp parallel for
		for (int i = 0; i <= arr_size -1; i += incr) {
			int temp = dest[i + (int) pow(2, d) -1];
			dest[i + (int) pow(2, d) - 1] = dest[i+ (int) pow(2, d+1) -1];
			dest[i + (int) pow(2, d+1) -1] = MAX(temp, dest[i + (int) pow(2, d+1) -1]);
		}
		//print_array(dest);
	}
	
	for (int i = 0; i < arr_size-1; i++)
		dest[i] = dest[i+1];
	//print_array(dest);
	dest[arr_size-1] = MAX(dest[arr_size-2], src[arr_size-1]);
	//print_array(dest);
}

void max_suffix(int *src, int *dest) {
	int power = log2(arr_size -1);
	for (int d = 0; d <= power; d++) {
		int incr = pow(2, d+1);
		#pragma omp parallel for
		for (int i = 0; i <= arr_size -1; i += incr) {
			//dest[i + (int) pow(2, d) -1] += dest[i + (int) pow(2, d+1) -1];
			dest[i] = MAX(dest[i], dest[i + (int) pow(2,d)]);
		}
		//print_array(dest);
	}
	dest[0] = INT_MIN;
	
	//puts("------------WOOUAH2");
	for (int d = power; d >= 0; d--) {
		int incr = pow(2, d+1);
		#pragma omp parallel for
		for (int i = 0; i <= arr_size -1; i += incr) {
			int temp = dest[i+ (int) pow(2, d)];
			dest[i + (int) pow(2, d)] = dest[i];
			dest[i] = MAX(dest[i], temp);
		}
		//print_array(dest);
	}
	for (int i = arr_size -1; i > 0; i--)
		dest[i] = dest[i-1];
	//print_array(dest);
	dest[0] = MAX(dest[1], src[0]);
	//print_array(dest);
}

void fill_m(int *psum, int *ssum, int *pmax, int *smax, int *m, int *q) {
	int *ms = malloc(arr_size * sizeof(int));
	if (!ms) {
		fprintf(stderr, "malloc error");
		exit(3);
	}
	int *mp = malloc(arr_size * sizeof(int));
	if (!mp) {
		fprintf(stderr, "malloc error");
		exit(3);
	}
	#pragma omp parallel for
	for (int i = 0; i < arr_size; i++) {
		ms[i] = pmax[i] - ssum[i] + q[i];
		mp[i] = smax[i] - psum[i] + q[i];
		m[i] = ms[i] + mp[i] - q[i];
	}
	free(ms);
	free(mp);
}
	
void get_subarray(int *m, int *index_max, int *max) {
	struct Compare max_start;
	
	//Calcul du max et de sa 1re occurence
	max_start.val = m[0];
	max_start.index = 0;
	#pragma omp parallel for reduction(maximum_ind_start:max_start)
	for (int i = 1; i < arr_size; i++) {
		if (m[i] > max_start.val) {
			max_start.val = m[i];
			max_start.index = i;
		} else if (m[i] == max_start.val && i < max_start.index) {
			max_start.index = i;
		}
	}
    //printf("ind_min: %d\n", max_start.index);
	*index_max = max_start.index;
	*max = max_start.val;
}

int main(int argc, char **argv) {
	/********************************************************************/
	/*						INITIALIZATION PART							*/
	/********************************************************************/
	char *input_text = 0, *token, *save, *buffer;
    int *array, i, *psum, *ssum, *pmax, *smax, *m, max_start, max;
	FILE *fptr; 
	size_t cur_len = 0;
	buffer = malloc(BUF_SIZE * sizeof(char));
	if (!buffer) {
		fprintf(stderr, "malloc error");
		exit(3);
	}

	if (argc != 2) {
		fprintf(stderr, "usage: ./Dormoy fichier\n");
		exit(1);
	}
    if (!(fptr = fopen(argv[1], "r"))) {
        printf("Error! opening file");
        exit(2);
    }
	//input_text = malloc(1000 * sizeof(char));
	//fscanf(fptr, "%[^\n]", input_text);
	
	while (fgets(buffer, sizeof(buffer), fptr)) {
		size_t buf_len = strlen(buffer);
		char *extra = realloc(input_text, (buf_len + cur_len + 1) * sizeof(char));
		//if (!extra) break;
		if (!extra) {
			fprintf(stderr, "malloc error");
			exit(3);
		}
		input_text = extra;
		strcpy(input_text + cur_len, buffer);
		cur_len += buf_len;
	}
	size_t len = strlen(input_text);
	//printf("len = %d\n", len);
	if ( len > 0 && input_text[len-1] == '\n') {
		input_text[--len] = '\0';
	}
	save = malloc((len + 1) * sizeof(char));
	if (!save) {
		fprintf(stderr, "malloc error");
		exit(3);
	}
	strcpy(save, input_text);
    //printf("Data from the file:\n|%s|\n\n\n-----------------save:\n|%s|\n", input_text, save);
	
	token = strtok(input_text, " ");
	while (token) {
		arr_size +=1;
		token = strtok(NULL, " ");
	}
	
	i = 0;
	array= malloc(arr_size * sizeof(int));
	if (!array) {
		fprintf(stderr, "malloc error");
		exit(3);
	}
	token = strtok(save, " ");
	while (token) {
		array[i] = atoi(token);
		i += 1;
		token = strtok(NULL, " ");
	}
	//print_array(array);
	
	psum = malloc(arr_size * sizeof(int));
	if (!psum) {
		fprintf(stderr, "malloc error");
		exit(3);
	}
	ssum = malloc(arr_size * sizeof(int));
	if (!ssum) {
		fprintf(stderr, "malloc error");
		exit(3);
	}
	pmax = malloc(arr_size * sizeof(int));
	if (!pmax) {
		fprintf(stderr, "malloc error");
		exit(3);
	}
	smax = malloc(arr_size * sizeof(int));
	if (!smax) {
		fprintf(stderr, "malloc error");
		exit(3);
	}
	m = malloc(arr_size * sizeof(int));
	if (!m) {
		fprintf(stderr, "malloc error");
		exit(3);
	}
	intcpy(array, psum);
	intcpy(array, ssum);
	//print_array(psum);
	
	/********************************************************************/
	
	struct timeval start, end;
  	gettimeofday(&start, NULL);
	
	//puts("Q:");	
	//print_array(array);
	//puts("--------------");
	//puts("PSUM:");
	sum_prefix(array, psum);
	//print_array(psum);
	//puts("SSUM:");
	sum_suffix(array, ssum);
	//print_array(ssum);

	intcpy(psum, smax);
	max_suffix(psum, smax);
	//puts("SMAX:");
	//print_array(smax);
	intcpy(ssum, pmax);
	max_prefix(ssum, pmax);
	//puts("PMAX:");
	//print_array(pmax);
	fill_m(psum, ssum, pmax, smax, m, array);
	//puts("M:");
	//print_array(m);

	get_subarray(m, &max_start, &max);
	//printf("start: %d\nmax: %d\n", max_start, max);

//	for (int sum = array[max_start], i = max_start; sum < max; sum += array) {
//
//	}
	i = max_start;
	int sum = array[i];
	printf("%d ", max);
	while (sum < max) {
		printf("%d ", array[i]);
		i += 1;
		sum += array[i];
	}
	printf("%d\n", array[i]);

	gettimeofday(&end, NULL);
	//printf("%ld\n", ((end.tv_sec * 1000000 + end.tv_usec)
	//	  - (start.tv_sec * 1000000 + start.tv_usec)));
    
	fclose(fptr);
	free(input_text);
	free(array);
	free(save);
	
	free(psum);
	free(ssum);
	free(pmax);
	free(smax);
	free(m);
	free(buffer);
	return 0;
}
