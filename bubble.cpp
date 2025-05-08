#include <iostream>
#include <omp.h>

using namespace std;

// Sequential Bubble Sort
void bubble(int array[], int n) {
    for (int i = 0; i < n - 1; i++){
        for (int j = 0; j < n - i - 1; j++){
            if (array[j] > array[j + 1]) 
                swap(array[j], array[j + 1]);
        }
    }
}

// Parallel Bubble Sort using Odd-Even Transposition Sort
void pBubble(int array[], int n){
    #pragma omp parallel
    {
        for(int i = 0; i < n; ++i){    
            // Odd indexed phase
            #pragma omp for
            for (int j = 1; j < n - 1; j += 2){
                if (array[j] > array[j + 1]){
                    swap(array[j], array[j + 1]);
                }
            }

            #pragma omp barrier

            // Even indexed phase
            #pragma omp for
            for (int j = 0; j < n - 1; j += 2){
                if (array[j] > array[j + 1]){
                    swap(array[j], array[j + 1]);
                }
            }

            #pragma omp barrier
        }
    }
}

// Print the array
void printArray(int arr[], int n){
    for(int i = 0; i < n; i++) 
        cout << arr[i] << " ";
    cout << "\n";
}

int main(){
    int n = 10;
    int arr[n], brr[n];
    double start_time, end_time;

    // Create an array with numbers from n to 1
    for(int i = 0, j = n; i < n; i++, j--) {
        arr[i] = j;
        brr[i] = j;  // Copy for parallel sort
    }

    // Sequential Bubble Sort
    start_time = omp_get_wtime();
    bubble(arr, n);
    end_time = omp_get_wtime();     
    cout << "Sequential Bubble Sort took: " << end_time - start_time << " seconds.\n";
    printArray(arr, n);

    // Parallel Bubble Sort
    start_time = omp_get_wtime();
    pBubble(brr, n);
    end_time = omp_get_wtime();     
    cout << "Parallel Bubble Sort took: " << end_time - start_time << " seconds.\n";
    printArray(brr, n);

    return 0;
}



//  g++ -fopenmp -o bubblesort bubblesort.cpp
//  ./bubblesort
