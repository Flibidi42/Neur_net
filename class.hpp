#ifndef CLASS_HPP_INCLUDED
#define CLASS_HPP_INCLUDED

#include <iostream>
#include <cstdlib>
#include <ctime>

const float learn_rate = 0.1f;

class Neur
{
private:
    int m_nb_branchs;
    float *m_weight;
    float sigmo(float val);
	float m_bias;
	Neur** hid_layers;
	int m_depth;
	
public:
	Neur();
    Neur(int nb_branchs);
	void setBranchs(int nb);
    float learning(float* input, float error_factor);
    float test(float *input);
	float get_weight(int i){
		if(i < m_nb_branchs)
			return m_weight[i];
		else
			return 0;
	};
};

class Net{
	
	private:
		Neur* input_layer;
		Neur* output_layer;
		int m_width;
		int m_nb_input;
	
	public:
		Net(int width, int nb_input);
		~Net();
		float learning(float m_expect, float *input);
		float test(float *input);
		void getState();
};

#endif // CLASS_HPP_INCLUDED
