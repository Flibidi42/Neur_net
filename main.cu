#include "class.hpp"
#define size_tab 8
#define nb_train 10000
#define size_layer 8

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
    Net my_net(size_layer, size_tab);
    int taille = 4;
    float *pat = new float[taille];
    pat[0] = 1;
    pat[1] = 0;
    pat[2] = 1;
    pat[3] = 1;
    float tab[size_tab];
    cout << compare(tab, pat, taille) << endl;

    float pat_test[size_tab];
    for(int  j = 0; j< nb_train; j++)
    {
        if(rand()%7 < 2){
            pat_gene(pat, pat_test, taille);
            cout << "Erreur : " << my_net.learning(1.f, pat_test) << " avec ";
            for(int  i = 0; i<size_tab; i++){
                cout << pat_test[i] << " ";
            }
        }
        else{
            for(int  i = 0; i<size_tab; i++){
                tab[i] = rand() % 2;
            }
            cout << "Erreur : " << my_net.learning(compare(tab, pat, taille), pat_test) << " avec ";
            for(int  i = 0; i<size_tab; i++){
                cout << tab[i] << " ";
            }
        }
        cout << endl;
    }



    cout << "Tab : ";
    tab[0] = 1;
    tab[1] = 1;
    tab[2] = 1;
    tab[3] = 1;
    tab[4] = 0;
    tab[5] = 0;
    tab[6] = 0;
    tab[7] = 0;
    for(int  i = 0; i<size_tab; i++){
        cout << tab[i] << " ";
    }
    cout << endl << "Test :" << my_net.test(tab) << endl;

    cout << "Tab : ";
    tab[0] = 0;
    tab[1] = 0;
    tab[2] = 1;
    tab[3] = 1;
    tab[4] = 0;
    tab[5] = 0;
    tab[6] = 0;
    tab[7] = 0;
    for(int  i = 0; i<size_tab; i++){
        cout << tab[i] << " ";
    }
    cout << endl << "Test :" << my_net.test(tab) << endl;

    cout << "Tab : ";
    tab[0] = 0;
    tab[1] = 0;
    tab[2] = 1;
    tab[3] = 1;
    tab[4] = 1;
    tab[5] = 1;
    tab[6] = 0;
    tab[7] = 0;
    for(int  i = 0; i<size_tab; i++){
        cout << tab[i] << " ";
    }
    cout << endl << "Test :" << my_net.test(tab) << endl;

    cout << "Tab : ";
    tab[0] = 1;
    tab[1] = 0;
    tab[2] = 1;
    tab[3] = 1;
    tab[4] = 0;
    tab[5] = 1;
    tab[6] = 1;
    tab[7] = 1;
    for(int  i = 0; i<size_tab; i++){
        cout << tab[i] << " ";
    }
    cout << endl << "Test :" << my_net.test(tab) << endl;

    return 0;
}

