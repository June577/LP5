#include <iostream>
#include <omp.h>
#include <vector>

using namespace std;

void merge(vector<int> &arr, int low, int mid, int high) {
    int n1 = mid - low + 1;
    int n2 = high - mid;

    vector<int> left(n1);
    vector<int> right(n2);

    for (int i = 0; i < n1; i++) left[i] = arr[low + i];
    for (int j = 0; j < n2; j++) right[j] = arr[mid + 1 + j];

    int i = 0, j = 0, k = low;

    while (i < n1 && j < n2) {
        if (left[i] <= right[j]) arr[k++] = left[i++];
        else arr[k++] = right[j++];
    }

    while (i < n1) arr[k++] = left[i++];
    while (j < n2) arr[k++] = right[j++];
}

void parallelMergeSort(vector<int> &arr, int low, int high) {
    if (low < high) {
        int mid = (low + high) / 2;

        #pragma omp parallel sections
        {
            #pragma omp section
            parallelMergeSort(arr, low, mid);

            #pragma omp section
            parallelMergeSort(arr, mid + 1, high);
        }
        merge(arr, low, mid, high);
    }
}

void mergeSort(vector<int> &arr, int low, int high) {
    if (low < high) {
        int mid = (low + high) / 2;
        mergeSort(arr, low, mid);
        mergeSort(arr, mid + 1, high);
        merge(arr, low, mid, high);
    }
}

void printArray(const vector<int> &arr) {
    for (int val : arr)
        cout << val << " ";
    cout << endl;
}

int main() {
    int n = 10;
    vector<int> arr(n), brr(n);
    double start_time, end_time;

    // Initialize array with n to 1
    for (int i = 0, j = n; i < n; i++, j--) {
        arr[i] = j;
        brr[i] = j;
    }

    start_time = omp_get_wtime();
    mergeSort(arr, 0, n - 1);
    end_time = omp_get_wtime();
    cout << "Time taken by sequential algorithm: " << end_time - start_time << " seconds\n";
    cout << "Sorted array (sequential): ";
    printArray(arr);

    start_time = omp_get_wtime();
    parallelMergeSort(brr, 0, n - 1);
    end_time = omp_get_wtime();
    cout << "Time taken by parallel algorithm: " << end_time - start_time << " seconds\n";
    cout << "Sorted array (parallel): ";
    printArray(brr);

    return 0;
}


//  g++ -fopenmp -o merge merge.cpp
//  ./merge
