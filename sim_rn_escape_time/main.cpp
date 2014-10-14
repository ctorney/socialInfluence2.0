
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
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#define VISUAL 0
#if VISUAL
#include "plotGrid.h"
#endif

using namespace std;
int main() 
{
    const gsl_rng_type * T;
    gsl_rng * r;



    gsl_rng_env_setup();

    T = gsl_rng_default;
    r = gsl_rng_alloc (T);

#if VISUAL
    plotGrid* pg = new plotGrid;
#endif

    // number of reps
    int numBlocks = 4*256;
    // length of grid
    int Nx = 8;
    int N = Nx * Nx;
    int N2 = 0.5 * N;
    int N4 = 0.5 * N2;
    int N_ALL = N * numBlocks;
    int nList[N];

    for (int i=0;i<N;i++)
    {
        nList[i] = i; 
    }


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
    int* d_net;
    CUDA_CALL(cudaMalloc((void**)&d_net, sizeof(int) * N * N));


    int* d_blockTotals;
    CUDA_CALL(cudaMalloc((void**)&d_blockTotals, sizeof(int) * numBlocks));

    float* h_wg = new float [N_ALL];
    int* h_states = new int[N_ALL];
    int* h_net = new int[N*N];
    int* h_blockTotals = new int[numBlocks];
    int* h_blockTimes = new int[numBlocks];
    int wgCount = 1;

    const unsigned int shape[] = {wgCount,2};

    float* results = new float[wgCount*2];


    for (int NL=28;NL<32;NL+=4)
    {
        for (int i=0;i<wgCount*2;i++)
            results[i]=0.0f;

        cout<<"~~~~~~~~~~~~~~~~~~"<<endl<<NL<<endl<<"~~~~~~~~~~~~~~~~~~"<<endl;

        char fileName[30];
        sprintf(fileName, "output/time-%d-last.npy", int(NL));
        for (int G=0;G<wgCount;G++)
        {
            memset(h_net, 0, sizeof(int)*N*N);
         /*   int nList2[N];
            for (int i=0;i<N;i++)
            {
                nList2[i] = i; 
            }
            gsl_ran_shuffle (r, nList2, N, sizeof (int));
            memset(h_net, 0, sizeof(int)*N*N);
            for (int ii=0;ii<N;ii++)
            {
                int i = nList2[ii];
                gsl_ran_shuffle (r, nList, N, sizeof (int));
                for (int j=0;j<N;j++)
                {
                    if (h_net[i*N]==NL) break;
                    int k = nList[j];
                    if (h_net[k*N]==NL) continue;
                    if (k==i) continue;
                    h_net[i*N]++;
                    h_net[k*N]++;
                    h_net[i*N + h_net[i*N]]=k;
                    h_net[k*N + h_net[k*N]]=i;

                }
            }*/
            float p = 0.5*NL/float(N-1);
            for (int i=0;i<N;i++)
            {
                for (int j=0;j<N;j++)
                {
                    if (i==j) continue;
                    if (gsl_rng_uniform (r)< p)
                    {
                        h_net[i*N]++;
                        h_net[j*N]++;
                        h_net[i*N + h_net[i*N]]=j;
                        h_net[j*N + h_net[j*N]]=i;

                    }


                }
            }
            // */
            CUDA_CALL(cudaMemcpy(d_net, h_net, (N*N) * sizeof(int), cudaMemcpyHostToDevice));
            //float wg = 0.25 + 0.0125 * float(G);
            float wg = 0.25 - 0.0125 * float(G);
 //           wg = 0.25;
            for (int i=0;i<N_ALL;i++)
                h_wg[i]=wg;

            CUDA_CALL(cudaMemcpy(d_wg, h_wg, (N_ALL) * sizeof(float), cudaMemcpyHostToDevice));


            for (int b=0;b<numBlocks;b++)
                h_blockTimes[b] = -1;
            int maxTime = 1000000000;
            int checkTime = 10;


            CUDA_CALL(cudaMemset (d_states, 0, sizeof(int) * (N_ALL)));
            CUDA_CALL(cudaMemset (d_blockTotals, 0, sizeof(int) * (numBlocks)));




            for (int t=0;t<maxTime;t++)
            {

                advanceTimestep(threadGrid, numBlocks, devRands, d_wg, d_states, d_net, Nx, NL, t);
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
