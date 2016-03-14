#include "class.hpp"
#define size_tab 2
#define nb_train 10000

using namespace std;


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
		cout << "Error : " << my_net.learning(test, tab) << endl;	
	}
	
	my_net.getState();
	
	tab[0] = 0;
	tab[1] = 0;
	cout << "Test : with 0 0 : " << my_net.test(tab) << endl;
	cout << 9 << endl;
	
	tab[0] = 0;
	tab[1] = 1;
	cout << "Test : with 0 1 : " << my_net.test(tab) << endl;
	
	tab[0] = 1;
	tab[1] = 0;
	cout << "Test : with 1 0 : " << my_net.test(tab) << endl;
	
	tab[0] = 1;
	tab[1] = 1;
	cout << "Test : with 1 1 : " << my_net.test(tab) << endl;
	
	cout << 10 << endl;
    return 0;
}
