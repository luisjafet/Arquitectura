#include <stdio.h>
#include <stdlib.h>
#include <time.h>


/* Utilidad para checar errores de CUDA */
void checkCUDAError(const char*);

/* Kernel para sumar dos vectores en un sólo bloque de hilos */
__global__ void vect_add(int *d_a, int *d_b, int *d_c)
{
    /* Part 2B: Implementación del kernel para realizar la suma de los vectores en el GPU */
	
	int id = threadIdx.x;
	d_c[id]= d_a[id] + d_b[id];
	
}

/* Versión de múltiples bloques de la suma de vectores */
__global__ void vect_add_multiblock(int *d_a, int *d_b, int *d_c)
{
    /* Part 2C: Implementación del kernel pero esta vez permitiendo múltiples bloques de hilos. */
	
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	d_c[id]= d_a[id] + d_b[id];
}

/* Numero de elementos en el vector */
#define ARRAY_SIZE 1024
#define sz  ARRAY_SIZE * sizeof(int)

/*
 * Número de bloques e hilos
 * Su producto siempre debe ser el tamaño del vector (arreglo).
 */
#define bloques  1
#define hilos 1024
#define bloques2  4
#define hilos2 256
#define bloques3  8
#define hilos3 128


int main(int argc, char *argv[])
{
    int *a, *b, *c; /* Arreglos del CPU */
    int *d_a, *d_b, *d_c;/* Arreglos del GPU */
	float elapsedTime;

    int i;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
    //size_t sz = ARRAY_SIZE * sizeof(int);

    /*
     * Reservar memoria en el cpu
     */
    a = (int *) malloc(sz);
    b = (int *) malloc(sz);
    c = (int *) malloc(sz);

    /*
     * Parte 1A:Reservar memoria en el GPU
     */
	cudaMalloc((void**)&d_a, sz);
	cudaMalloc((void**)&d_b, sz);
	cudaMalloc((void**)&d_c, sz);
    

    /* inicialización */
    for (i = 0; i < ARRAY_SIZE; i++) {
        a[i] = i;
        b[i] = ARRAY_SIZE - i;
        c[i] = 0;
    }

    /* Parte 1B: Copiar los vectores del CPU al GPU */
    
	cudaMemcpy( d_a,a, sz, cudaMemcpyHostToDevice);
	cudaMemcpy( d_b,b, sz, cudaMemcpyHostToDevice);

    /* Parte 2A: Configurar y llamar los kernels */
    /* dim3 dimGrid( ); */
    /* dim3 dimBlock( ); */
    cudaEventRecord(start, 0);
	vect_add<<<bloques , hilos>>>(d_a,d_b,d_c );
	cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
	

    /* Esperar a que todos los threads acaben y checar por errores */
    cudaThreadSynchronize();
    checkCUDAError("kernel invocation");

    /* Part 1C: copiar el resultado de nuevo al CPU */
    cudaMemcpy(c,d_c, sz, cudaMemcpyDeviceToHost );

    checkCUDAError("memcpy");
	
	
    /* print out the result */
    printf("Results: ");
    for (i = 0; i < ARRAY_SIZE; i++) {
      printf("%04d\t ", c[i]);
    }
	
	cudaEventElapsedTime(&elapsedTime,start,stop);
	printf("\nTiempo %4.6f milseg\n\n",elapsedTime);
	printf("Presione cualquier caracter para continuar ejecucuion de 4 bloques 256 hilos....  ");
	getchar();
	printf("\n\n");
	
	cudaEventRecord(start,0);
	vect_add_multiblock<<<bloques2,hilos2>>>(d_a,d_b,d_c );
	cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    checkCUDAError("kernel invocation");

    /* Part 1C: copiar el resultado de nuevo al CPU */
    cudaMemcpy(c,d_c, sz, cudaMemcpyDeviceToHost );

    checkCUDAError("memcpy");

    printf("Results: ");
    for (i = 0; i < ARRAY_SIZE; i++) {
      printf("%04d\t ", c[i]);
    }
	cudaEventElapsedTime(&elapsedTime,start,stop);
	printf("\nTiempo %4.6f milseg\n\n",elapsedTime);
	
	
	printf("Presione cualquier caracter para continuar ejecucuion de 8 bloques 64 hilos....  ");
	getchar();
	getchar();
	printf("\n\n");
    
	cudaEventRecord(start,0);
	vect_add_multiblock<<<bloques3,hilos3>>>(d_a,d_b,d_c );
	cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    checkCUDAError("kernel invocation");

    /* Part 1C: copiar el resultado de nuevo al CPU */
    cudaMemcpy(c,d_c, sz, cudaMemcpyDeviceToHost );

    checkCUDAError("memcpy");

    printf("Results: ");
    for (i = 0; i < ARRAY_SIZE; i++) {
      printf("%04d\t ", c[i]);
    }
	cudaEventElapsedTime(&elapsedTime,start,stop);
	printf("\nTiempo %4.6f milseg\n\n",elapsedTime);
	
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

    /* Parte 1D: Liberar los arreglos */
    cudaFree( d_a);
	cudaFree( d_b);
	cudaFree( d_c);

    free(a);
    free(b);
    free(c);

    return 0;
}


/* Utility function to check for and report CUDA errors */
void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }                         
}
