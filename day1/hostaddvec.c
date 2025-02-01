#include <stdio.h>
#include <stdlib.h>
#include <omp.h>



void addtwovecs(double *arr1,double *arr2, double*arr3,int n)
{

    int i;
    for(i=0;i<n;i++)
    {
        arr3[i]=arr1[i]+arr2[i];
    }
}

int main()
{
    double arr1[5]={1,2,3,4,5};
    double arr2[5]={6,7,8,9,10};
    double *arr3;
    arr3=malloc(5*sizeof(double));
    addtwovecs(arr1,arr2,arr3,5);
    int i;
    for(i=0;i<5;i++)
    {
        printf("%f\n",arr3[i]);
    }
    return 0;
}