#include "cuda_runtime.h"
#include "device_launch_parameters.h"
/*
* Plantilla para la multiplicación de matrices
* con memoria compartida
* Jose Incera. Adaptado del código
* de Robert Hochberg
* Abril 2016
*
* Based nearly entirely on the code from the CUDA C Programming Guide
*/

#include <time.h>
#include <stdio.h>
#include <stdlib.h>

// Estructura Matriz.
typedef struct{
	int nRen;
	int nCol;
	float *elementos;
	int salto; // stride para recorrer columnas
} Matriz;

// dimensión de un bloque
// El tamaño es TAM_BLOQUE * TAM_BLOQUE
#define TAM_BLOQUE 64

// Prototipo de función
__global__ void MatMultKernel(const Matriz, const Matriz, Matriz);

__global__ void multMatriz(Matriz , Matriz , Matriz  );

// Por facilidad, las dimensiones de la matriz son múltiplos de TAM_BLOQUE
void MatMultShared(const Matriz A, const Matriz B, Matriz C) {

	// Carga A y B en memoria GPU
	Matriz d_A;
	d_A.nRen = d_A.salto = A.nRen;
	d_A.nCol = A.nCol;
	size_t tam = A.nRen * A.nCol * sizeof(float);

	cudaError_t err = cudaMalloc(&(d_A.elementos), tam);  //  AGREGA LOS ARGUMENTOS QUE CORRESPONDAN
	cudaMemcpy(d_A.elementos, A.elementos, tam, cudaMemcpyHostToDevice); //  AGREGA LOS ARGUMENTOS QUE CORRESPONDAN

	Matriz d_B;
	d_B.nRen = d_B.salto = B.nRen;
	d_B.nCol = B.nCol;
	tam = B.nRen * B.nCol * sizeof(float);

	cudaMalloc(&(d_B.elementos), tam); //  AGREGA LOS ARGUMENTOS QUE CORRESPONDAN
	cudaMemcpy(d_B.elementos, B.elementos, tam, cudaMemcpyHostToDevice);  //  AGREGA LOS ARGUMENTOS QUE CORRESPONDAN

	// Asigna espacio para C en GPU
	Matriz d_C;
	d_C.nRen = d_C.salto = C.nRen;
	d_C.nCol = C.nCol;
	tam = C.nRen * C.nCol * sizeof(float);
	cudaMalloc(&(d_C.elementos), tam);  //  AGREGA LOS ARGUMENTOS QUE CORRESPONDAN

	// Llama al kernel
	dim3 dimBlock(TAM_BLOQUE, TAM_BLOQUE);
	dim3 dimGrid(B.nRen / dimBlock.x, A.nCol / dimBlock.y);

	//  Descomenta y AGREGA LOS ARGUMENTOS QUE CORRESPONDAN
	MatMultKernel <<<dimGrid , dimBlock >>> (d_A, d_B, d_C);

	// Espera a que todos terminen
	cudaThreadSynchronize();

	// Lee C from del GPU
	cudaMemcpy(C.elementos, d_C.elementos, tam, cudaMemcpyDeviceToHost);// (void **)  AGREGA LOS ARGUMENTOS QUE CORRESPONDAN

	// Libera memoria GPU
	cudaFree(d_A.elementos);//  AGREGA LOS ARGUMENTOS QUE CORRESPONDAN
	cudaFree(d_B.elementos);//  AGREGA LOS ARGUMENTOS QUE CORRESPONDAN
	cudaFree(d_C.elementos);//  AGREGA LOS ARGUMENTOS QUE CORRESPONDAN
}

void MatMultGlobal(const Matriz A, const Matriz B, Matriz C) {

	// Carga A y B en memoria GPU
	Matriz d_A;
	d_A.nRen = d_A.salto = A.nRen;
	d_A.nCol = A.nCol;
	size_t tam = A.nRen * A.nCol * sizeof(float);

	cudaError_t err = cudaMalloc(&(d_A.elementos), tam);  //  AGREGA LOS ARGUMENTOS QUE CORRESPONDAN
	cudaMemcpy(d_A.elementos, A.elementos, tam, cudaMemcpyHostToDevice); //  AGREGA LOS ARGUMENTOS QUE CORRESPONDAN

	Matriz d_B;
	d_B.nRen = d_B.salto = B.nRen;
	d_B.nCol = B.nCol;
	tam = B.nRen * B.nCol * sizeof(float);

	cudaMalloc(&(d_B.elementos), tam); //  AGREGA LOS ARGUMENTOS QUE CORRESPONDAN
	cudaMemcpy(d_B.elementos, B.elementos, tam, cudaMemcpyHostToDevice);  //  AGREGA LOS ARGUMENTOS QUE CORRESPONDAN

	// Asigna espacio para C en GPU
	Matriz d_C;
	d_C.nRen = d_C.salto = C.nRen;
	d_C.nCol = C.nCol;
	tam = C.nRen * C.nCol * sizeof(float);
	cudaMalloc(&(d_C.elementos), tam);  //  AGREGA LOS ARGUMENTOS QUE CORRESPONDAN

	// Llama al kernel
	// dim3 dimBlock(1, 1);
	// dim3 dimGrid(B.nRen, A.nCol);

	//  Descomenta y AGREGA LOS ARGUMENTOS QUE CORRESPONDAN
	// multMatriz << <dimGrid, dimBlock >> >(d_A, d_B, d_C); 
	// multMatriz <<<dimGrid, dimBlock >>> (d_A, d_B, d_C, d_A.nRen);

	// Invoke kernel
	dim3 dimBlock(TAM_BLOQUE, TAM_BLOQUE);
	dim3 dimGrid((B.nCol + dimBlock.x - 1) / dimBlock.x, (A.nRen + dimBlock.y - 1) / dimBlock.y);
	multMatriz << <dimGrid, dimBlock >> >(d_A, d_B, d_C);

	// Espera a que todos terminen
	cudaThreadSynchronize();

	// Lee C from del GPU
	cudaMemcpy(C.elementos, d_C.elementos, tam, cudaMemcpyDeviceToHost);// (void **)  AGREGA LOS ARGUMENTOS QUE CORRESPONDAN

	// Libera memoria GPU
	cudaFree(d_A.elementos);//  AGREGA LOS ARGUMENTOS QUE CORRESPONDAN
	cudaFree(d_B.elementos);//  AGREGA LOS ARGUMENTOS QUE CORRESPONDAN
	cudaFree(d_C.elementos);//  AGREGA LOS ARGUMENTOS QUE CORRESPONDAN
}

// Toma un elemento de la matriz
__device__ float GetElement(const Matriz A, int ren, int col) {
	return A.elementos[ren* A.salto + col];
}

// Pon un elemento en la matriz
__device__ void SetElement(Matriz A, int ren, int col, float value) {
	A.elementos[ren* A.salto + col] = value;
}

// Toma una submatriz de A de tamaño TAM_BLOQUExTAM_BLOQUE
// localizada col sub-matrices a la derecha y ren sub-matrices abajo
// desde la esquina superior izquierda
__device__ Matriz LeeSubMatriz(Matriz A, int ren, int col) {
	Matriz Asub;
	Asub.nRen = TAM_BLOQUE;
	Asub.nCol = TAM_BLOQUE;
	Asub.salto = A.salto;
	Asub.elementos = &A.elementos[A.salto * TAM_BLOQUE * ren + TAM_BLOQUE * col];
	return Asub;
}

// Kernel multiplicación de Matriz
__global__ void MatMultKernel(Matriz A, Matriz B, Matriz C) {

	// Renglon y columna del bloque
	int blockRen = blockIdx.y;
	int blockCol = blockIdx.x;

	// Cada bloque calcula una submatriz Csub de C
	Matriz Csub = LeeSubMatriz(C, blockRen, blockCol);

	// Cada thread calcula un elemento de Csub
	// acumulando elementos en valorC
	float valorC = 0.0;

	// Thread ren y col dentro de Csub
	int ren = threadIdx.y;
	int col = threadIdx.x;

	// Loop sobre todas las sub-matrices de A y B necesarias
	// para calcular Csub
	// Multiplica cada par de sub-matrices y acumula resultados
	for (int m = 0; m < (A.nRen / TAM_BLOQUE); ++m) {

		// Toma sub-Matriz Asub de A
		Matriz Asub = LeeSubMatriz(A, blockRen, m);//  AGREGA LOS ARGUMENTOS QUE CORRESPONDAN

		// Toma sub-Matriz Bsub de B
		Matriz Bsub = LeeSubMatriz(B, m, blockCol);//  AGREGA LOS ARGUMENTOS QUE CORRESPONDAN

		// La memoria compartida donde se almacenan Asub y Bsub
		__shared__ float As[TAM_BLOQUE][TAM_BLOQUE];
		__shared__ float Bs[TAM_BLOQUE][TAM_BLOQUE];

		// Transfiere Asub y Bsub de memoria global a shared
		// Cada thread carga un elemento de cada submatriz
		As[ren][col] = GetElement(Asub, ren, col);
		Bs[ren][col] = GetElement(Bsub, ren, col);

		// Punto de sincronización: Espera a que todas las
		// sub-matrices se hayan cargado antes de continuar
		__syncthreads();

		// Multiplica Asub y Bsub
		for (int e = 0; e < TAM_BLOQUE; ++e)
		{
			// Descomenta y agrega la operación apropiada
			valorC += As[ren][e] * Bs[e][col];

			// Punto de sincronización antes de iniciar otra iteración
			__syncthreads();
		}
	}

	// Escribe Csub a memoria global
	// Cada thread escribe un elemento
	SetElement(Csub, ren, col, valorC);
}

__global__ void multMatriz(Matriz A, Matriz B, Matriz C){

	// int blockRen = blockIdx.y;
	// int blockCol = blockIdx.x;
	/*
	float sum = 0;
	for (unsigned int k = 0; k<num; k++)
		sum += A.elementos[blockRen * num + k] * B.elementos[k * num + blockCol];
	C.elementos[blockRen*num + blockCol] = sum;
	*/
	float Cvalue = 0.0;

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row > A.nRen || col > B.nCol) return;
	for (int e = 0; e < A.nCol; ++e)
		Cvalue += (A.elementos[row * A.nCol + e]) * (B.elementos[e * B.nCol + col]);
	C.elementos[row * C.nCol + col] = Cvalue;
}

int main(int argc, char* argv[]){
	
	Matriz A, B, C, CGlobal, CSecuencial;
	int a1, a2, b1, b2;
	a1 = atoi(argv[1]);			/* nCol de A */
	a2 = atoi(argv[2]);			/* nRen  de A */
	b1 = a2;		         	/* nCol de B */
	b2 = atoi(argv[3]);			/* nRen  de B */

	// printf("\n A(%d, %d) B(%d, %d)", a1, a2, b1, b2);

	A.nCol = a1;
	A.nRen = a2;
	A.elementos = (float*)malloc(A.nRen * A.nCol * sizeof(float));

	B.nCol = b1;
	B.nRen = b2;
	B.elementos = (float*)malloc(B.nRen * B.nCol * sizeof(float));

	C.nCol = A.nCol;
	C.nRen = B.nRen;
	C.elementos = (float*)malloc(C.nRen * C.nCol * sizeof(float));

	CGlobal.nCol = A.nCol;
	CGlobal.nRen = B.nRen;
	CGlobal.elementos = (float*)malloc(CGlobal.nRen * CGlobal.nCol * sizeof(float));

	CSecuencial.nCol = A.nCol;
	CSecuencial.nRen = B.nRen;
	CSecuencial.elementos = (float*)malloc(CSecuencial.nRen * CSecuencial.nCol * sizeof(float));

	// Llena las matrices con valores aleatorios
	for (int i = 0; i < A.nCol; i++)
	for (int j = 0; j < A.nRen; j++)
		A.elementos[i*A.nRen + j] = (rand() % 3);

	for (int i = 0; i < B.nCol; i++)
	for (int j = 0; j < B.nRen; j++)
		B.elementos[i*B.nRen + j] = (rand() % 2);

	// Multiplicacion secuencial
	clock_t begin = clock();
	int n = A.nRen;
	for (int i = 0; i<B.nRen; i++){
		for (int j = 0; j<A.nCol; j++) {
			float sum = 0;
			for (int k = 0; k < n; k++)
				sum += A.elementos[i * n + k] * B.elementos[k * n + j];
			CSecuencial.elementos[i * n + j] = sum;
		}
	}
	clock_t end = clock();

	double diffticks = end - begin;
	double diffmsSecuencial = (diffticks * 10) / CLOCKS_PER_SEC;

	begin = clock();  // Para medir cuánto tarda
	MatMultShared(A, B, C);
	end = clock();  // Checa el tiempo inmediatamente después de terminar

	diffticks = end - begin;
	double diffmsShared = (diffticks * 10) / CLOCKS_PER_SEC;

	begin = clock();  // Para medir cuánto tarda
	MatMultGlobal(A, B, CGlobal);
	end = clock();  // Checa el tiempo inmediatamente después de terminar

	diffticks = end - begin;
	double diffmsGlobal = (diffticks * 10) / CLOCKS_PER_SEC;

	

	// Imprime hasta porciones de 10x10 de las tres matrices
	
	printf("A: \n");
	for(int i = 0; i < min(10, A.nCol); i++){
	for(int j = 0; j < min(10, A.nRen); j++)
	printf("%f ", A.elementos[i*A.nRen + j]);
	printf("\n");
	}
	printf("B: \n");
	for(int i = 0; i < min(10, B.nCol); i++){
	for(int j = 0; j < min(10, B.nRen); j++)
	printf("%f ", B.elementos[i*B.nRen + j]);
	printf("\n");
	}
	printf("C Secuencial: \n");
	for (int i = 0; i < min(10, CSecuencial.nCol); i++){
		for (int j = 0; j < min(10, CSecuencial.nRen); j++)
			printf("%f ", CSecuencial.elementos[i*CSecuencial.nRen + j]);
		printf("\n");
	}
	printf("C Shared: \n");
	for(int i = 0; i < min(10, C.nCol); i++){
	for(int j = 0; j < min(10, C.nRen); j++)
	printf("%f ", C.elementos[i*C.nRen + j]);
	printf("\n");
	}
	printf("C Global: \n");
	for (int i = 0; i < min(10, CGlobal.nCol); i++){
		for (int j = 0; j < min(10, CGlobal.nRen); j++)
			printf("%f ", CGlobal.elementos[i*CGlobal.nRen + j]);
		printf("\n");
	}
	printf("\n");
	
	printf("\Multiplicacion de matrices A(%d, %d) x B(%d, %d) \nTiempo de ejecución (Seg) \nSecuencial: %f, MCompartida: %f, MGlobal: %f\n\n", a1, a2, b1, b2, diffmsSecuencial, diffmsShared, diffmsGlobal);


	/*
	free(&A);
	free(&B);
	free(&C);
	*/
}