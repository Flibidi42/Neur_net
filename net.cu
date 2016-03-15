#include "class.hpp"

using namespace std;

Net::Net(int width, int nb_input, int depth = 1){
	m_width = width;
	m_depth = depth;
	hid_layers = new Neur*[depth];
	for(int i = 0; i < width; i++){
		hid_layers[i] = new Neur[width];
	}
	output_layer = new Neur(width);
	for(int i = 0; i < width; i++){
		for(int j = 0; j<width; j++){
			hid_layers[i][j].setBranchs(nb_input);
		}
	}
	m_nb_input = nb_input;
}

float Net::learning(float m_expect, float *input){
	float**transition = new float*[m_depth];
	for(int  i = 0; i<m_depth; i++)
    {
		transition[i] = new float[m_width];
	}
    float output = 0.f;
	for(int j = 0; j<m_width; j++){
		transition[0][j]  = hid_layers[0][j].test(input);
	}
    for(int  i = 1; i<m_depth; i++)
    {
		for(int j = 0; j<m_width; j++){
			transition[i][j]  = hid_layers[i][j].test(transition[i-1]);
		}
    }
    output = output_layer->test(transition[m_depth-1]);
    float error = m_expect - output;
    float add = output_layer->learning(transition, error);
    for(int  i = 0; i<m_width; i++)
    {
        hid_layers[m_depth-1][i].learning(input, add*output_layer->get_weight(i));
    }
    error = m_expect - test(input);
	delete transition[];
    return 0.5*error*error;
}

float Net::test(float *input){
	float *transition = new float[m_depth];
	for(int  i = 0; i<m_depth; i++)
    {
		transition[i] = new float[m_width];
	}
    float output = 0.f;
	for(int j = 0; j<m_width; j++){
		transition[0][j]  = hid_layers[0][j].test(input);
	}
    for(int  i = 1; i<m_depth; i++)
    {
		for(int j = 0; j<m_width; j++){
			transition[i][j]  = hid_layers[i][j].test(transition[i-1]);
		}
    }
    output = output_layer->test(transition[m_depth-1]);
	delete transition[];
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