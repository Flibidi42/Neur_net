#include "class.hpp"
#include <cmath>
#include "../common/book.h"
#define threadsPerBlock 16

using namespace std;

float frand_a_b(float a, float b)
{
    return ( rand()/(float)RAND_MAX ) * (b-a) + a;
}

__global__ void dot( float *a, float *b, float *c, int *N, float *add) {
	 __shared__ float cache[threadsPerBlock];
	 int tid = threadIdx.x + blockIdx.x * blockDim.x;
	 int cacheIndex = threadIdx.x;
	 float temp = 0;
	 while (tid < *N) {
		b[tid] += *add * a[tid];
		temp += a[tid] * b[tid];
		tid += blockDim.x * gridDim.x;
	 }
	 cache[cacheIndex] = temp;
	 
	 __syncthreads();
	 
	 int i = blockDim.x/2;
	 while (i != 0) {
		 if (cacheIndex < i)
			cache[cacheIndex] += cache[cacheIndex + i];
		 __syncthreads();
		 i /= 2;
	 }
	if (cacheIndex == 0)
		c[blockIdx.x] = cache[0];
}

__global__ float para_learning(float* input, float error_factor, float* val_test, float* m_weight_old, float* m_weight, int* nb_branch, float *bias){
	
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	float y = 0.f;
    float add = 0.f;
	 for(int i = 0; i<m_nb_branchs; i++)
    {
        m_weight_old[i] = m_weight[i];
    }
    y = test(input);
    add = y*(1-y)*error_factor;
    for(int i = 0; i<m_nb_branchs; i++)
    {
        m_weight[i] += add*input[i]*learn_rate;
    }
	m_bias += add*learn_rate;
	return add;	
	
}

Neur::Neur(int nb_branchs)
{
    m_nb_branchs = nb_branchs;
    m_weight = new float[nb_branchs];
    for(int i = 0; i < nb_branchs; i++)
    {
        m_weight[i] = 0.5;
    }
	m_weight_old = new float[nb_branchs];
    for(int i = 0; i < nb_branchs; i++)
    {
        m_weight_old[i] = 0.5;
    }
}

Neur::Neur()
{
    m_nb_branchs = 1;
    m_weight = new float[1];
    for(int i = 0; i < 1; i++)
    {
        m_weight[i] = frand_a_b(0.f, 1.f);
    }
	m_weight_old = new float[1];
    for(int i = 0; i < 1; i++)
    {
        m_weight_old[i] = frand_a_b(0.f, 1.f);
    }
	m_bias = 0;	
}

void Neur::setBranchs(int nb){
	
	
	float *tempo = new float[nb];
	for(int i = 0; i<nb; i++){
		if(i<m_nb_branchs)
			tempo[i] = m_weight[i];
		else
			tempo[i] = 0;
	}
	delete m_weight;
	m_weight = tempo;
	delete tempo;
	tempo = new float[nb];
	for(int i = 0; i<nb; i++){
		tempo[i] = m_weight[i];
	}
	delete m_weight_old;
	m_weight_old = tempo;
	m_nb_branchs = nb;
}

float Neur::learning(float* input, float error_factor)
{
	/* Linear neuron
	int nb_blocks = ((m_nb_branchs+(threadsPerBlock-1))/threadsPerBlock);
	
	float *c = (float *)malloc(nb_blocks*sizeof(float));
	float *dev_a, *dev_b, *dev_partial_c, *dev_add; 
	int *dev_N;	
	
	HANDLE_ERROR( cudaMalloc( (void**)&dev_a, m_nb_branchs*sizeof(float) ) );
	HANDLE_ERROR( cudaMemcpy( dev_a, input, m_nb_branchs*sizeof(float), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_b, m_nb_branchs*sizeof(float) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_partial_c, ((m_nb_branchs+15)/16)*sizeof(float) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_N, sizeof(int) ) );
	HANDLE_ERROR( cudaMemcpy( dev_N, &m_nb_branchs, sizeof(int), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_add, sizeof(int) ) );
	
    float out = 0.f;
    float error = m_expect - out;
    float add = 0.f;
    while(error > m_accept || -error > m_accept)
    {
        add = error*learn_rate;
		HANDLE_ERROR( cudaMemcpy( dev_add, &add, sizeof(float), cudaMemcpyHostToDevice ) );
        out = 0.f;
		HANDLE_ERROR( cudaMemcpy( dev_b, m_weight, m_nb_branchs*sizeof(float), cudaMemcpyHostToDevice ) );
		
		dot<<<nb_blocks,threadsPerBlock>>>( dev_a, dev_b, dev_partial_c, dev_N, dev_add);
		
		HANDLE_ERROR( cudaMemcpy(c, dev_partial_c, nb_blocks*sizeof(float), cudaMemcpyDeviceToHost ) );
		
		for(int i = 0; i<nb_blocks; i++){
            out += c[i];
        }
		
		HANDLE_ERROR( cudaMemcpy(m_weight, dev_b, m_nb_branchs*sizeof(float), cudaMemcpyDeviceToHost ) );
		
		cout << "Out reel:" << out << endl;
		
        error =(m_expect - out);
        cout << "Error : " << error << endl;
    }
	/*cudaFree( dev_a );
	cudaFree( dev_b );
	cudaFree( dev_partial_c );
	cudaFree( dev_add );
	cudaFree( dev_N );
	free(c);*/
	 //Logistic neuron
	 for(int i = 0; i<m_nb_branchs; i++)
    {
        m_weight_old[i] = m_weight[i];
    }
    float y = 0.f;
    float add = 0.f;
    y = test(input);
    add = y*(1-y)*error_factor;
    for(int i = 0; i<m_nb_branchs; i++)
    {
        m_weight[i] += add*input[i]*learn_rate;
    }
	m_bias += add*learn_rate;
	return add;
}

float Neur::test(float *input)
{
	int nb_blocks = ((m_nb_branchs+(threadsPerBlock-1))/threadsPerBlock);
	
	float *c = (float *)malloc(nb_blocks*sizeof(float));
	float *dev_a, *dev_b, *dev_partial_c, *dev_add; 
	int *dev_N;
	
	HANDLE_ERROR( cudaMalloc( (void**)&dev_a, m_nb_branchs*sizeof(float) ) );
	HANDLE_ERROR( cudaMemcpy( dev_a, input, m_nb_branchs*sizeof(float), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_b, m_nb_branchs*sizeof(float) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_partial_c, ((m_nb_branchs+(threadsPerBlock-1))/threadsPerBlock)*sizeof(float) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_N, sizeof(int) ) );
	HANDLE_ERROR( cudaMemcpy( dev_N, &m_nb_branchs, sizeof(int), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_add, sizeof(int) ) );
	
    float out = 0.f;
    float add = 0.f;
	HANDLE_ERROR( cudaMemcpy( dev_add, &add, sizeof(float), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_b, m_weight, m_nb_branchs*sizeof(float), cudaMemcpyHostToDevice ) );
		
	dot<<<nb_blocks,threadsPerBlock>>>( dev_a, dev_b, dev_partial_c, dev_N, dev_add);
		
	HANDLE_ERROR( cudaMemcpy(c, dev_partial_c, nb_blocks*sizeof(float), cudaMemcpyDeviceToHost ) );
		
	for(int i = 0; i<nb_blocks; i++){
        out += c[i];
    }

	cudaFree( dev_a );
	cudaFree( dev_b );
	cudaFree( dev_partial_c );
	cudaFree( dev_add );
	cudaFree( dev_N );
	free(c);
	return sigmo(out);
}

float Neur::sigmo(float val){
    return 1/(1+exp(-val));
}
