/******************************************************************
Description: Program to calculate the Mandelbrot set using 
             a manager-worker pattern

Notes: compile with: mpicc -o mandelbrot_mpi_mw mandelbrot_mpi_mw.c -lm
       run with:     sbatch run_mandelbrot_mw.sh

Contact: D. Acreman, Exeter University
******************************************************************/

#include <stdio.h>
#include <complex.h>
#include <stdbool.h>
#include <math.h>
#include <mpi.h>

/* Number of intervals on real and imaginary axes*/
#define N_RE 12000
#define N_IM  8000

/* Number of iterations at each z value */
int nIter[N_RE+1][N_IM+1];

/* Points on real and imaginary axes*/
float z_Re[N_RE+1], z_Im[N_IM+1];

/* Domain size */
const float z_Re_min = -2.0; /* Minimum real value*/
const float z_Re_max =  1.0; /* Maximum real value */
const float z_Im_min = -1.0; /* Minimum imaginary value */
const float z_Im_max =  1.0; /* Maximum imaginary value */

/* Set to true to write out results*/
const bool doIO = false;
const bool verbose = false;

/******************************************************************************************/

/* Calculate number of iterations for a given i index (real axis) and loop over all j values 
   (imaginary axis). The nIter array is populated with the results. */

void calc_vals(int i){

  /* Maximum number of iterations*/
  const int maxIter=100;

  /* Value of Z at current iteration*/
  float complex z;
  /*Value of z at iteration zero*/
  float complex z0;

  int j,k;
  
  /* Loop over imaginary axis */
  for (j=0; j<N_IM+1; j++){
    z0 = z_Re[i] + z_Im[j]*I;
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

/******************************************************************************************/

void write_to_file(char filename[]){

  int i, j;
  FILE *outfile;
  
  outfile=fopen(filename,"w");
  for (i=0; i<N_RE+1; i++){
    for (j=0; j<N_IM+1; j++){
      fprintf(outfile,"%f %f %d \n",z_Re[i], z_Im[j], nIter[i][j]);
    }
  }
  fclose(outfile);
  
}

/******************************************************************************************/

int main(int argc, char *argv[]){

  /* MPI related variables */
  int myRank;        /* Rank of this MPI process */
  int nProcs;        /* Total number of MPI processes*/
  int nextProc;      /* Next process to send work to */
  MPI_Status status; /* Status from MPI calls */
  int endFlag=-9999; /* Flag to indicate completion*/
  int missingData=-6666; /* Missing data flag */

  const int BUFFSIZE=N_IM+3; /* N_IM+1 points + rank + i value */
  int buffer[BUFFSIZE];
  int this_i;

  /* Timing variables */
  double start_time, end_time;
  
  /* Loop indices */
  int i,j;
  
  MPI_Init(&argc, &argv);

  /* Record start time */
  start_time=MPI_Wtime();

  /* Get job size and rank information */
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &nProcs); 

  if (myRank==0 && verbose){
    printf("Calculating Mandelbrot set with %d processes\n", nProcs);
  }
  
  /* Initialise to nIter to a missing data value */ 
  for (i=0; i<N_RE+1; i++){
    for (j=0; j<N_IM+1; j++){
      nIter[i][j]=-3;
    }
  }

  /* Points on real axis */
  for (i=0; i<N_RE+1; i++){
    z_Re[i] = ( ( (float) i)/ ( (float) N_RE)) * (z_Re_max - z_Re_min) + z_Re_min;
  }

  /* Points on imaginary axis */
  for (j=0; j<N_IM+1; j++){
    z_Im[j] = ( ( (float) j)/ ( (float) N_IM)) * (z_Im_max - z_Im_min) + z_Im_min;
  }

	 
  // Manager process
  if ( myRank == 0 ){
    // Hand out work to worker processes
    for (i=0; i<N_RE+1; i++){
      // Receive request for work and data
      MPI_Recv(&buffer, BUFFSIZE, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

      /* Put results into nIter */
      nextProc=buffer[0];
      this_i=buffer[1];
      if (this_i != missingData){
	for (j=0; j<N_IM+1; j++){
	  nIter[this_i][j]=buffer[j+2];
	}
      }

      // Send i value to requesting process
      MPI_Send(&i,        1, MPI_INT, nextProc,       100,         MPI_COMM_WORLD);
    }
    // Tell all the worker processes to finish (once for each worker process = nProcs-1)
    for (i=0; i<nProcs-1; i++){
      // Receive request for work
      MPI_Recv(&buffer, BUFFSIZE, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

      /* Put results into nIter */
      nextProc=buffer[0];
      this_i=buffer[1];
      if (this_i != missingData){
	for (j=0; j<N_IM+1; j++){
	  nIter[this_i][j]=buffer[j+2];
	}
      }

      // Send endFlag to finish
      MPI_Send(&endFlag,     1, MPI_INT, nextProc,       100,         MPI_COMM_WORLD);
    }
  }

  // Worker Processes
  else {

    /* Initialise first send buffer. Values after [1] will be set to missing data value */
    buffer[0]=myRank;
    buffer[1]=missingData;
    for(j=2; j<BUFFSIZE; j++){
      buffer[j]=missingData;
    }

    while(true){

      // Send request for work and data
      MPI_Send(&buffer, BUFFSIZE, MPI_INT, 0, 100+myRank, MPI_COMM_WORLD);
      // Receive i value to work on
      MPI_Recv(&i,      1, MPI_INT, 0, 100       , MPI_COMM_WORLD, &status); 

      if (i==endFlag){
	break;
      } else {
	calc_vals(i);
      }

      /* Put results in buffer ready to send to manager */
      buffer[0]=myRank;
      buffer[1]=i;
      for(j=0; j<N_IM+1; j++){
	buffer[j+2]= nIter[i][j];
      }

    } // while(true)
  } // else worker process

  /* Write out results */
  if (doIO && myRank==0 ){
    if (verbose) {printf("Writing out results from process %d \n", myRank);}
    write_to_file("mandelbrot.dat");
  } else {
    if (myRank==0){
      printf("Centre: nIter=%d\n",nIter[N_RE/2][N_IM/2]);
	}
  }

  /* Record end time */
  MPI_Barrier(MPI_COMM_WORLD);
  end_time=MPI_Wtime();

  /* Record end time. The barrier synchronises the process so they all measure the same time */
  if (myRank==0){
    printf("STATS (num procs, elapsed time): %d %f\n", nProcs, end_time-start_time);
  }

  MPI_Finalize();

  return 0;

}


