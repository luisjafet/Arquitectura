
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

# define NPOINTS 2000
# define MAXITER 2000

#define SIZE NPOINTS*NPOINTS*sizeof(int)

struct complex{
  double real;
  double imag;
};



__global__ void mandelbrot(int npoints, int max, int *num){


	double  ztemp;
	struct complex z, c;
	int iter;
	
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	
	while (i<npoints) {
    while (j<npoints) {
      c.real = -2.0+2.5*(double)(i)/(double)(npoints)+1.0e-7;
      c.imag = 1.125*(double)(j)/(double)(npoints)+1.0e-7;
      z=c;
      for (iter=0; iter<max; iter++){
		ztemp=(z.real*z.real)-(z.imag*z.imag)+c.real;
		z.imag=z.real*z.imag*2+c.imag; 
		z.real=ztemp; 
		
		if ((z.real*z.real+z.imag*z.imag)>4.0e0) {
		  num[i +j*npoints]=1; 
		  break;
		}
		//__syncthreads();
      }
	  j+=gridDim.y * blockDim.y;
	  
	  
    }
	i+=gridDim.x*blockDim.x;
	j=threadIdx.y + blockIdx.y * blockDim.y;
  }
  
  
}


int main() {
	
	int i;
	double total=0;
	int *dnum, *numoutside;
	double area, error;

	dim3 BLOQUES (16,16);
	dim3 HILOS(16,16);
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
 
	numoutside =(int *)malloc(SIZE);
	for(i=0; i<NPOINTS*NPOINTS;i++)
		numoutside[i]=0;
		
 	cudaMalloc((void**)&dnum, SIZE);
	cudaMemcpy(dnum,numoutside, SIZE, cudaMemcpyHostToDevice);
	
    cudaEventRecord(start, 0);
	mandelbrot<<<BLOQUES,HILOS>>>(NPOINTS, MAXITER, dnum);
	cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
	
	cudaMemcpy(numoutside,dnum,SIZE, cudaMemcpyDeviceToHost) ;
	for(i=0; i<NPOINTS*NPOINTS;i++)
		total += numoutside[i];

	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime,start,stop);
	
	

    area=2.0*2.5*1.125*(double)(NPOINTS*NPOINTS-total)/(double)(NPOINTS*NPOINTS);
    error=area/(double)NPOINTS;
	
	
    printf("Area of Mandlebrot set = %12.8f +/- %12.8f\n",area,error);
	printf("Tiempo %4.6f milseg\n\n",elapsedTime);
	
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	cudaFree(dnum);
	free(numoutside);

    return 0;
}