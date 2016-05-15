

#include <stdio.h>

__global__ void multMatriz(float *da, float *db, float *dc, int num){
	float sum=0;
	for (unsigned int k = 0; k<num; k++)
			sum += da[threadIdx.y * num + k] * db[k * num + threadIdx.x];
	dc[threadIdx.y*num + threadIdx.x] = (float) sum;
}

#define n 32
#define SIZE n*n*sizeof(float)

int main(){

	int N=32;
	float *A, *B, *C;
	float *da, *db, *dc;
	int m;
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);	
	dim3 dimGrid(1, 1);
	dim3 dimBlock(N, N);
	
	A=(float *)malloc(SIZE);
	B=(float *)malloc(SIZE);
	C=(float *)malloc(SIZE);
	for(m=0;m<N*N;m++){
		A[m]=(float)m+1;
		B[m]=(float)m+2;
		C[m]=(float)0;
	}
	
	cudaMalloc((void**)&da, SIZE);
	cudaMalloc((void**)&db, SIZE);
	cudaMalloc((void**)&dc, SIZE);
	
	cudaMemcpy(da,A, SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(db,B, SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(dc,C, SIZE, cudaMemcpyHostToDevice);
	
	cudaEventRecord(start, 0);
	multMatriz<<<dimGrid , dimBlock >>>(da,db,dc,N);
	//cudaThreadSynchronize();
	cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
	
	cudaMemcpy(C,dc, SIZE, cudaMemcpyDeviceToHost);
	
	
	
	for(m=0;m<N*N;m++){
		printf("%08.0f",A[m]);
		printf("%c",( (m%N)<(N-1) ) ? '\t':'\n');
		
	}
	printf("\n\n");
	
	for(m=0;m<N*N;m++){
		printf("%08.0f",B[m]);
		printf("%c",( (m%N)<(N-1) ) ? '\t':'\n');
		
	}
	printf("\n\n");
	
	for(m=0;m<N*N;m++){
		printf("%08.0f",C[m]);
		printf("%c",( (m%N)<(N-1) ) ? '\t':'\n');
		
	}
	printf("\n\n");
	
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime,start,stop);
	printf("Tiempo %4.6f milseg\n\n",elapsedTime);
	
	
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);
	free(A);
	free(B);
	free(C);
	
	return 0;
}