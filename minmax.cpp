#include <iostream>
#include <omp.h>

using namespace std;

int findMin(int arr[], int n)
{
    int minVal = arr[0];
#pragma omp parallel for reduction(min : minVal)
    for (int i = 0; i < n; i++)
    {
        if (arr[i] < minVal)
            minVal = arr[i];
    }
    return minVal;
}

int findMax(int arr[], int n)
{
    int maxVal = arr[0];
#pragma omp parallel for reduction(max : maxVal)
    for (int i = 0; i < n; i++)
    {
        if (arr[i] > maxVal)
            maxVal = arr[i];
    }
    return maxVal;
}

int findSum(int arr[], int n)
{
    int sum = 0;
#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < n; i++)
    {
        sum += arr[i];
    }
    return sum;
}

double findAverage(int arr[], int n)
{
    return static_cast<double>(findSum(arr, n)) / n;
}

int main()
{
    int n = 6;
    int arr[] = {15, 42, 7, 88, 23, 9};

    cout << "Parallel Reduction Results:" << endl;
    cout << "Minimum Value: " << findMin(arr, n) << endl;
    cout << "Maximum Value: " << findMax(arr, n) << endl;
    cout << "Sum: " << findSum(arr, n) << endl;
    cout << "Average: " << findAverage(arr, n) << endl;

    return 0;
}

// g++ -fopenmp reduction.cpp -o reduction
// ./reduction
