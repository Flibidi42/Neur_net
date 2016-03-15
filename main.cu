#include "class.hpp"
#define size_tab 8
#define nb_train 1000
#define size_layer 16

using namespace std;


float compare(float *tab, float *pat, int taille){

    int j = 0;
    bool testing = false; // 1 0 1
    for(int  i = 0; i<size_tab; i++){ // 0 1 1 0 1
        if(!testing && tab[i] == pat[j]){
            testing = true;
            j++;
            if(taille == 1)
                return 1;
        }
        else if(testing && tab[i] == pat[j]){
            j++;
            if(taille == j)
                return 1;
        }
        else if(testing && tab[i] != pat[j] && tab[i] != pat[0]){
            j = 0;
            testing = false;
        }
        else if(testing && tab[i] != pat[j] && tab[i]){
            j = 1;
        }
    }
    return 0;
}

void pat_gene(float *pat, float *pat_test, int taille){

    int step = rand()%(size_tab - taille);
    int j = 0;
    for(int  i = 0; i<size_tab; i++){
        if(i >= step && i<step+taille){
            pat_test[i] = pat[j];
            j++;
        }
        else
            pat_test[i] = rand() % 2;
    }
}

int main()
{
	srand(time(NULL));
    Net my_net(2, size_tab);
	float tab[size_tab];
	
	int test = 0;
	
	for(int  i = 0; i< nb_train; i++){
		tab[0] = rand() % 2;
		tab[1] = rand() % 2;
		test = (tab[0] == 1 || tab[1] == 1)?1:0;
		my_net.learning(test, tab);
	}
	
	my_net.getState();
	
	tab[0] = 0;
	tab[1] = 0;
	cout << "Test : with 0 0 : " << my_net.test(tab) << endl;
	
	tab[0] = 0;
	tab[1] = 1;
	cout << "Test : with 0 1 : " << my_net.test(tab) << endl;
	
	tab[0] = 1;
	tab[1] = 0;
	cout << "Test : with 1 0 : " << my_net.test(tab) << endl;
	
	tab[0] = 1;
	tab[1] = 1;
	cout << "Test : with 1 1 : " << my_net.test(tab) << endl;

    return 0;
}

