
/*test.cc*/
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <iostream>
#include <algorithm>

#include <time.h>
#include "cuda_call.h"
#include "switcherKernel.h"
#include "params.h"
#include "cnpy.h"

#define VISUAL 0
#if VISUAL
#include "plotGrid.h"
#endif

using namespace std;
int main() 
{

#if VISUAL
    plotGrid* pg = new plotGrid;
#endif

    // number of reps
    int numBlocks = 256;
    // length of grid
    int Nx = 8;
    int N = Nx * Nx;
    int N2 = 0.5 * N;
    int N4 = 0.5 * N2;
    int N_ALL = N * numBlocks;

    dim3 threadGrid(Nx, Nx);
    curandState *devRands;
    CUDA_CALL(cudaMalloc((void **)&devRands, N_ALL * sizeof(curandState)));

    srand (time(NULL));
    initRands(threadGrid, numBlocks, devRands, rand());


    float* d_wg;
    CUDA_CALL(cudaMalloc((void**)&d_wg, sizeof(float) *  (N_ALL) ));
    int* d_states;
    CUDA_CALL(cudaMalloc((void**)&d_states, sizeof(int) * N_ALL));
    int* d_states2;
    CUDA_CALL(cudaMalloc((void**)&d_states2, sizeof(int) * N_ALL));


    int* d_blockTotals;
    CUDA_CALL(cudaMalloc((void**)&d_blockTotals, sizeof(int) * numBlocks));

    float* h_wg = new float [N_ALL];
    int* h_states = new int[N_ALL];
    int* h_blockTotals = new int[numBlocks];
    int* h_blockTimes = new int[numBlocks];
    int wgCount = 11;

    const unsigned int shape[] = {wgCount,2};

    float* results = new float[wgCount*2];


    for (int NL=4;NL<36;NL+=4)
    {
        for (int i=0;i<wgCount*2;i++)
            results[i]=0.0f;


        cout<<"~~~~~~~~~~~~~~~~~~"<<endl<<NL<<endl<<"~~~~~~~~~~~~~~~~~~"<<endl;

        char fileName[30];
        sprintf(fileName, "output/time-%d.npy", int(2.0*NL));
        for (int G=0;G<wgCount;G++)
        {
            float wg = 0.5 - 0.025 * float(G);
            for (int i=0;i<N_ALL;i++)
                h_wg[i]=wg;

            CUDA_CALL(cudaMemcpy(d_wg, h_wg, (N_ALL) * sizeof(float), cudaMemcpyHostToDevice));


            for (int b=0;b<numBlocks;b++)
                h_blockTimes[b] = -1;
            int maxTime = 10000000;
            int checkTime = 1;


            CUDA_CALL(cudaMemset (d_states, 0, sizeof(int) * (N_ALL)));
            CUDA_CALL(cudaMemset (d_blockTotals, 0, sizeof(int) * (numBlocks)));




            for (int t=0;t<maxTime;t++)
            {

                advanceTimestep(threadGrid, numBlocks, devRands, d_wg, d_states, Nx, NL, t);
                /*
                   CUDA_CALL(cudaMemcpy(h_states, d_states, (N_ALL) * sizeof(int), cudaMemcpyDeviceToHost));
                   int countUp = 0;
                   for (int i=0;i<N_ALL;i++)
                   if (h_states[i]>0)
                   countUp++;
                   cout<<"~~~~~~~~~~~~~~~~~~~~~~~~~~~"<<endl<<countUp<<endl;
                //            */
#if VISUAL
                CUDA_CALL(cudaMemcpy(h_states, d_states, (N_ALL) * sizeof(int), cudaMemcpyDeviceToHost));
                pg->draw(Nx, h_states);
#endif
                if (t%checkTime == 0 ) 
                {
                    // cout<<t<<endl;
                    countStates(N, numBlocks, d_states, d_blockTotals, N_ALL);

                    CUDA_CALL(cudaMemcpy(h_blockTotals, d_blockTotals, (numBlocks) * sizeof(int), cudaMemcpyDeviceToHost));
                    bool allDone = true;
                    for (int b=0;b<numBlocks;b++)
                    {
                        if (h_blockTotals[b]>0.5*N)
                        {
                            //                    cout<<"block total : "<<h_blockTotals[b]<<endl;
                            if (h_blockTimes[b]<0)
                                h_blockTimes[b]=t;
                        }
                        else
                        {
                            //           cout<<b<<" block done"<<endl;
                            allDone = false;
                        }
                    }
                    if (allDone)
                    {
                        break;
                        /*
                           for (int b=0;b<numBlocks;b++)
                           h_blockTimes[b] = -1;
                           CUDA_CALL(cudaMemset (d_states, 0, sizeof(int) * (N_ALL)));
                           */
                    } 
                }

            }

            float avTime = 0.0f;
            int count=0;
            for (int b=0;b<numBlocks;b++)
                if (h_blockTimes[b]>0)
                {
                    avTime += (float)h_blockTimes[b];
                    count++;
                }
            results[G*2] = wg;
            if (count>0)
                results[G*2+1] = avTime/(float)count;
            else
                results[G*2+1] = maxTime;
            if (avTime/(float)count > 100*checkTime)
                checkTime = checkTime * 10;
            if (checkTime > 10000)
                checkTime = 10000;

            cout<<results[G*2]<<" "<<results[G*2+1]<<endl;
            cnpy::npy_save(fileName,results,shape,2,"w");
        }
    }
    return 0;
}
