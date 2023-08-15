#include<iostream>
#include<cuda.h>

#include<sys/time.h>
using namespace std;
#define NTHREADS_X 32
#define NTHREADS_Y 32
// kernel for transpose
__global__ void transpose(int *d_matrixOut, int *d_matrixIn, int r, int c) {
    __shared__ int tile[32][32];
    int blockoffset1=blockIdx.x * blockDim.y ;
    int blockoffset2=blockIdx.y * blockDim.x ;
    int u = blockoffset2 + threadIdx.x;
    int v = blockoffset1+ threadIdx.y;
    int x = blockoffset1 + threadIdx.x;
    int y = blockoffset2 + threadIdx.y;
    /*
	For Calculating the Transpose of the matrix we have created a shared Memory name Tile where we will store our result first in row major 
	format but when transferring our result from shared memory we will use column major format since latency of shared memory is low
	we will get our result faster as compare to direct global memory transfer
	*/
    if(x<c) {
      if(y<r){
        int indexin=y*c + x;
        tile[threadIdx.y][threadIdx.x] = d_matrixIn[indexin];
      }
    }
    __syncthreads();



    if(u<r){
      if(v<c){
        int outindex=v*r+u;
    d_matrixOut[outindex] = tile[threadIdx.x][threadIdx.y];
    }
  }
        
}
__global__ void matrixMul(int *a, int *b, int *c, int a_columns, int c_row,
        int c_ncolumns, int nBlocks)
{
	int i;
	int  z;
	int sum = 0;

    /* The number of Multiplication 
     */
    int nMultiplications =a_columns;

    /*
	The block will multiply the NTHREADS Y values throughout each iteration. 
	If the number of a columns is greater than NTHREADS Y, 
	then this value may be less than NTHREADS Y. This variable is used to manage that.
     */
    int multiplicationsInBlock = NTHREADS_Y;
	int xoffset=blockIdx.x * blockDim.x;
    int column = xoffset + threadIdx.x;
	int yoffset= blockIdx.y * blockDim.y; 
    int line =  yoffset + threadIdx.y;
/*
We are tanking two shared Memory of size 32*32 in each Block , since there latency is low  , they will help us in getting results faster
*/
    __shared__ int s_a[NTHREADS_Y][NTHREADS_X];
    __shared__ int s_b[NTHREADS_Y][NTHREADS_X];


    int  a_row;
	int  a_Col;
	int  b_row;
	int  b_Col;

    for (z = 0; z < nBlocks; z++)
    {
		/*
         We are Loading Matrix A into shared Memory, Now here we have to keep coalsced Memory access in our mind for that we are accessing
		 our memory using threadIdx.y and threadIdx.x and we are acessing 32 Banks at once of one Blocks.
		 */
		int offset1=blockIdx.y * NTHREADS_Y ;
        a_row = (offset1 + threadIdx.y);
		int offset2=z * NTHREADS_X ;
        a_Col = (offset2+ threadIdx.x);
        if (a_row < c_row)
		{ 
			if(a_Col < a_columns)
        	{
			int cal=(a_columns * a_row);
			int temp=cal+a_Col;
            s_a[threadIdx.y][threadIdx.x] = a[temp];
        	}
		}

        /*
         We are Loading Matrix B into shared Memory, Now here we have to keep coalsced Memory access in our mind for that we are accessing
		 our memory using threadIdx.y and threadIdx.x and we are acessing 32 Banks at once of one Blocks.
		 */
		int offset3=z * NTHREADS_Y ;
         b_row = (offset3+ threadIdx.y);
		int offset4=blockIdx.x * NTHREADS_X;
        b_Col = (offset4 + threadIdx.x);
        if ( b_row < a_columns)
		{

		 if(b_Col < c_ncolumns)
    	    {	
			int inside=(c_ncolumns *  b_row);
            s_b[threadIdx.y][threadIdx.x] = b[ inside + b_Col ];
        	}
		}
        __syncthreads();

        /*
		Now we are doing our actual Matrix Multiplication here we have to keep in mind since shared memory latency is less and Global
		Memory latency is high thats why we have first store our Matrix into shared Memory A and shared Memory B and then we have done the 
		actual Matrix Multiplication again at last we are sending our results to Global Matrix and after global Matrix we will send our results
		to CPU Memory using CudaMemCpy

         */
        if (column < c_ncolumns)
			if( line < c_row)
        {
            if (nMultiplications < 32)
            {
                multiplicationsInBlock = nMultiplications;
            }

            for (i = 0; i < multiplicationsInBlock; i++)
            {
                sum += s_a[threadIdx.y][i] * s_b[i][threadIdx.x];
            }

            nMultiplications -= NTHREADS_Y;
        }

        __syncthreads();
    }

    /* 
	We are Now checking the actual.valid position of our thread in Matrix C
     */
    if (column < c_ncolumns)
	{	if(line < c_row)
    	{
		int inside=line * c_ncolumns;
        c[ inside + column] = sum;
    	}
	}
}
__global__ void add_matrices(int *d_matrixA, int *d_matrixB) {
    int id= blockIdx.x*blockDim.x + threadIdx.x;
    d_matrixA[id] += d_matrixB[id];
}
// function to compute the output matrix
void computE(int p, int q, int r, int *h_matrixA, int *h_matrixB, 
	         int *h_matrixC, int *h_matrixD, int *h_matrixE){
	// Device variables declarations...
	int *d_matrixA, *d_matrixB, *d_matrixC, *d_matrixD, *d_matrixE,*d_transD,*d_matrixTemp1,*d_matrixTemp2;
	struct timeval t1, t2;
	double seconds, microSeconds;
	// allocate memory...
	cudaMalloc(&d_matrixA, p * q * sizeof(int));
	cudaMalloc(&d_matrixB, q * r * sizeof(int));
	cudaMalloc(&d_matrixC, p * q * sizeof(int));
	cudaMalloc(&d_matrixD, r * q * sizeof(int));
	cudaMalloc(&d_matrixE, p * r * sizeof(int));

	// copy the values...
	cudaMemcpy(d_matrixA, h_matrixA, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixB, h_matrixB, q * r * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixC, h_matrixC, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixD, h_matrixD, r * q * sizeof(int), cudaMemcpyHostToDevice);

	/* ****************************************************************** */
	cudaMalloc(&d_transD, r * q * sizeof(int));//Calculating transpose of matrix D
	dim3 grid1((q+31)/32, (r+31)/32);
	dim3 block1(32, 32);
	transpose<<<grid1, block1>>>(d_transD, d_matrixD, r, q);
    cudaFree(d_matrixD); 
	d_matrixD = d_transD;

	/* Write your code here */
	/* Configure and launch kernels */

	/* ****************************************************************** */
    cudaMalloc(&d_matrixTemp1, p * r * sizeof(int));
    cudaMalloc(&d_matrixTemp2, p * r * sizeof(int));
	dim3 grid2((r+31)/32, (p+31)/32);
	dim3 block2(32, 32);
    matrixMul<<<grid2,block2 >>>( d_matrixA, d_matrixB, d_matrixTemp1,q, p,r,(int)ceil( (double) q / NTHREADS_X));
	matrixMul<<<grid2,block2>>>( d_matrixC, d_matrixD,d_matrixTemp2, q, p, r,(int)ceil( (double) q / NTHREADS_X));
	add_matrices<<<p, r>>>(d_matrixTemp1, d_matrixTemp2);
	cudaMemcpy(h_matrixE, d_matrixTemp1, p * r * sizeof(int), cudaMemcpyDeviceToHost);

    
	// copy the result back...
	//cudaMemcpy(h_matrixE, d_matrixE, p * r * sizeof(int), cudaMemcpyDeviceToHost);

	// deallocate the memory...
	cudaFree(d_matrixA);
	cudaFree(d_matrixB);
	cudaFree(d_matrixC);
	cudaFree(d_matrixD);
	cudaFree(d_matrixE);
	cudaFree(d_matrixTemp1);
	cudaFree(d_matrixTemp2);
}

// function to read the input matrices from the input file
void readMatrix(FILE *inputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fscanf(inputFilePtr, "%d", &matrix[i*cols+j]);
		}
	}
}

// function to write the output matrix into the output file
void writeMatrix(FILE *outputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fprintf(outputFilePtr, "%d ", matrix[i*cols+j]);
		}
		fprintf(outputFilePtr, "\n");
	}
}

int main(int argc, char **argv) {
	// variable declarations
	int p, q, r;
	int *matrixA, *matrixB, *matrixC, *matrixD, *matrixE;
	struct timeval t1, t2;
    double seconds, microSeconds;

	// get file names from command line
	char *inputFileName = argv[1];
	char *outputFileName = argv[2];

	// file pointers
	FILE *inputFilePtr, *outputFilePtr;
    
    inputFilePtr = fopen(inputFileName, "r");
	if(inputFilePtr == NULL) {
	    printf("Failed to open the input file.!!\n"); 
		return 0;
	}

	// read input values
	fscanf(inputFilePtr, "%d %d %d", &p, &q, &r);

	// allocate memory and read input matrices
	matrixA = (int*) malloc(p * q * sizeof(int));
	matrixB = (int*) malloc(q * r * sizeof(int));
	matrixC = (int*) malloc(p * q * sizeof(int));
	matrixD = (int*) malloc(r * q * sizeof(int));
	readMatrix(inputFilePtr, matrixA, p, q);
	readMatrix(inputFilePtr, matrixB, q, r);
	readMatrix(inputFilePtr, matrixC, p, q);
	readMatrix(inputFilePtr, matrixD, r, q);

	// allocate memory for output matrix
	matrixE = (int*) malloc(p * r * sizeof(int));

	// call the compute function
    gettimeofday(&t1, NULL);
	computE(p, q, r, matrixA, matrixB, matrixC, matrixD, matrixE);
	cudaDeviceSynchronize();
	gettimeofday(&t2, NULL);

	// print the time taken by the compute function
	seconds = t2.tv_sec - t1.tv_sec;
	microSeconds = t2.tv_usec - t1.tv_usec;
	printf("Time taken (ms): %.3f\n", 1000*seconds + microSeconds/1000);

	// store the result into the output file
	outputFilePtr = fopen(outputFileName, "w");
	writeMatrix(outputFilePtr, matrixE, p, r);

	// close files
	fclose(inputFilePtr);
	fclose(outputFilePtr);

	// deallocate memory
	free(matrixA);
	free(matrixB);
	free(matrixC);
	free(matrixD);
	free(matrixE);

	return 0;
}
	
