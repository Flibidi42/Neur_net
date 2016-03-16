#include "class.hpp"

using namespace std;

__global__ void para_learning(float* add, float* y, float* error_factor, float* bias, float* nb_branchs, float* input, float* N, float** w, float** w_old){ // N, error_factor and input : non array
	
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	 while (tid < *N) {
		 for(int i = 0; i<m_nb_branchs; i++)
		{
			m_weight_old[i] = m_weight[i];
		}
		add[tid] = y*(1-y)*error_factor;
		for(int i = 0; i<nb_branchs[tid] ; i++)
		{
			m_weight[tid] [i] += add[tid] *input[i]*learn_rate;
		}
		m_bias += add*learn_rate;
		tid += blockDim.x * gridDim.x;
	 }	
}

Net::Net(int width, int nb_input){
	m_width = width;
	input_layer = new Neur[width];
	output_layer = new Neur(width);
	for(int i = 0; i < width; i++){
		input_layer[i].setBranchs(nb_input);
	}
	m_nb_input = nb_input;
}

float Net::learning(float m_expect, float *input){
	
	float *transition = new float[m_width];
    float output = 0.f;
    for(int  i = 0; i<m_width; i++)
    {
        transition[i]  = input_layer[i].test(input);
    }
    output = output_layer->test(transition);
    float error = m_expect - output;
    float add = output_layer->learning(transition, error);
    
	//Parallel
	float  **y, **bias, **error_factor, w, w_old;
	float **dev_y, **dev_bias, *dev_nb_branchs, **dev_error_factor,  *dev_input, *dev_N, *dev_w, *dev_w_old;
	
	y = (float*)malloc(sizeof(float*) * m_width);
	bias = (float*)malloc(sizeof(float*) * m_width);
	error_factor = (float*)malloc(sizeof(float) * m_width);
	w = (float**)malloc(sizeof(float*) * m_width);
	w_old = (float**)malloc(sizeof(float*) * m_width);
	
	for(int  i = 0; i<m_width; i++)
    {
		w[i] = (float*) malloc (sizeof(float) * m_nb_input);
		w_old[i] = (float*) malloc (sizeof(float) * m_nb_input);
    }
	
	//alloc
	HANDLE_ERROR( cudaMalloc( (void***)&dev_y, m_width*sizeof(float) ) );
	HANDLE_ERROR( cudaMalloc( (void***)&dev_bias, m_width*sizeof(float) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_nb_branchs, sizeof(float) ) );
	HANDLE_ERROR( cudaMalloc( (void***)&dev_error_factor, sizeof(float) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_input, m_nb_branchs*sizeof(float) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_N, sizeof(float) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_w, m_width*sizeof(float) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_w_old, m_width*sizeof(float) ) );
	
	//copy
	HANDLE_ERROR( cudaMemcpy( dev_nb_branchs, &m_nb_input, sizeof(float), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_input, input, m_nb_input * sizeof(float), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_N, &m_width, sizeof(float), cudaMemcpyHostToDevice ) );
	
	for(int  i = 0; i<m_width; i++)
    {
		y[i] = input_layer[i].test(input);
		bias[i] = input_layer[i].m_bias;
		
    }
	
    error = m_expect - test(input);
	delete transition;
    return 0.5*error*error;
}

float Net::test(float *input){
	float *transition = new float[m_width];
	float output = 0.f;
	for(int  i = 0; i<m_width; i++){
		transition[i]  = input_layer[i].test(input);
	}
	output = output_layer->test(transition);
	delete transition;
	return output;
}

void Net::getState(){
	
	for(int i = 0; i < m_width; i++){
		cout << "Neuron nb "<< i <<" : ";
		for(int j = 0; j< m_nb_input; j++){
			cout << input_layer[i].get_weight(j) << " ";
		}
		cout << endl;
	}
	
	cout << "Output Neuron : ";
	for(int j = 0; j< m_width; j++){
		cout << output_layer->get_weight(j) << " ";
	}
	cout << endl;
	
}