/* Program to calculate the Mandlebrot set with MPI parallelism
   D. Acreman */

#include <stdio.h>
#include <complex.h>
#include <stdbool.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char *argv[]){

  /* Declare variables */
  /* Loop indices for real and imaginary axes*/
  int i, j;
  /* Loop index for iteration*/
  int k;
  /* Maximum number of iterations*/
  const int maxIter=100;

  /* Number of points on real and imaginary axes*/
  const int nRe = 12000;
  const int nIm =  8000;

  /* Domain size */
  const float z_Re_min = -2.0;
  const float z_Re_max =  1.0;
  const float z_Im_min = -1.0;
  const float z_Im_max =  1.0;
  
  /* Real and imaginary components of z*/
  float z_Re, z_Im;
  /* Value of Z at current iteration*/
  float complex z;
  /*Value of z at iteration zero*/
  float complex z0;

  int nIter[nRe+1][nIm+1];

  const bool doIO = false; 
  const bool verbose = false;
  
  /* Rank of this MPI process */
  int myRank;
  /* Total number of MPI processes*/
  int nProcs; 

  MPI_Init(&argc, &argv);

  /* Record start time */
  double start_time, end_time;
  start_time=MPI_Wtime();

  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &nProcs); 

  if (verbose) {printf("Process %d of %d starting calculation\n", myRank, nProcs);}

  /* Initialise to nIter zero as we will use a reduce to collate results into this array*/ 
  for (i=0; i<nRe+1; i++){
    for (j=0; j<nIm+1; j++){
      nIter[i][j]=0;
    }
  }

  /* Loop over real and imaginary axes interleaving iterations between MPI processes */
  for (i=myRank; i<nRe+1; i+=nProcs){
    for (j=0; j<nIm+1; j++){
      z_Re = ( ( (float) i)/ ( (float) nRe)) * (z_Re_max - z_Re_min) + z_Re_min;
      z_Im = ( ( (float) j)/ ( (float) nIm)) * (z_Im_max - z_Im_min) + z_Im_min;
      z0 = z_Re + z_Im*I;
      z  = z0;

      /* Iterate up to a maximum number or bail out if mod(z) > 2 */
      k=0;
      while(k<maxIter){
        nIter[i][j] = k;
        if (cabs(z) > 2.0)
          break;
        z = z*z + z0;
        k++;
      }
    }
  }

  /* call MPI_reduce to collate all results on process 0*/
  int sendBuffer[(nRe+1)*(nIm+1)];
  int receiveBuffer[(nRe+1)*(nIm+1)];
  int index=0;
  /* Pack nIter into a 1D buffer for sending*/
  for (i=0; i<nRe+1; i++){
    for (j=0; j<nIm+1; j++){
      sendBuffer[index]=nIter[i][j];
      index++;
    }
  }

  MPI_Reduce(&sendBuffer, &receiveBuffer, (nRe+1)*(nIm+1), MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD); 
  
  /* Unpack receive buffer into nIter */
  index=0;
  for (i=0; i<nRe+1; i++){
    for (j=0; j<nIm+1; j++){
      nIter[i][j]=receiveBuffer[index];
      index++;
    }
  }

  /* Write out results */
  if (doIO && myRank==0 ){
    if (verbose) {printf("Writing out results from process %d \n", myRank);}
    FILE *outfile;
    outfile=fopen("mandelbrot.dat","w");

    for (i=0; i<nRe+1; i++){
      for (j=0; j<nIm+1; j++){
	z_Re = ( ( (float) i)/ ( (float) nRe)) * (z_Re_max - z_Re_min) + z_Re_min;
	z_Im = ( ( (float) j)/ ( (float) nIm)) * (z_Im_max - z_Im_min) + z_Im_min;
	fprintf(outfile,"%f %f %d \n",z_Re, z_Im, nIter[i][j]);
      }
    }
    fclose(outfile);
  } else {
    if (myRank==0){
      printf("Centre: nIter=%d\n",nIter[nRe/2][nIm/2]);
	}
  }

  /* Record end time. The barrier synchronises the process so they all measure the same time */
  MPI_Barrier(MPI_COMM_WORLD);
  end_time=MPI_Wtime();

  /* Report elapsed time */
  if (myRank==0){
    printf("STATS (num procs, elapsed time): %d %f\n", nProcs, end_time-start_time);
  }

  MPI_Finalize();

  return 0;

}
