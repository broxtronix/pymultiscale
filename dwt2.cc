/*
 * This C code was originally part of the waveslim package for R.
 * 
 *   http://cran.r-project.org/web/packages/waveslim/index.html
 *
 * It is licensed under the GPL (version 2 or above) by its original
 * author, Brandon Whitcher <bwhitcher@gmail.com>.  It is now part of
 * this Python written by Michael Broxton <broxton@stanford.edu>.
 *
 */

#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include "dwt.h"

/***************************************************************************
 ***************************************************************************
   2D DWT 
 ***************************************************************************
 ***************************************************************************/

void two_D_dwt(double *X, int *M, int *N, int *L, double *h, double *g, 
	       double *LL, double *LH, double *HL, double *HH)
{
  int i, j, k;
  double *data, *Wout, *Vout, *Low, *High;
  
  /*
   *  Perform one-dimensional DWT on columns (length M).
   */
  
  Wout = (double *) malloc((*M) * sizeof(double));
  Vout = (double *) malloc((*M) * sizeof(double));
  
  /*
   *  Create temporary "matrices" to store DWT of columns.
   */
  Low = (double *) malloc((*N*(*M/2)) * sizeof(double));
  High = (double *) malloc((*N*(*M/2)) * sizeof(double));
  
  for(i = 0; i < *N; i++) {
    /*
     *  Must take column from X and place into vector for DWT.
     */
    data = (double *) malloc((*M) * sizeof(double));
    for(j = 0; j < *M; j++) {
      data[j] = X[i*(*M)+j];
      /* printf("X[%d][%d] = %f\n", i, j, X[i*(*M)+j]); */
    }
    /*
     *  Perform DWT and read into temporary matrices.
     */
    dwt(data, M, L, h, g, Wout, Vout);
    for(k = 0; k < (int) *M/2; k++) {
      Low[i*(*M/2)+k] = Vout[k]; 
      High[i*(*M/2)+k] = Wout[k];
      /* printf("Low[%d][%d] = %f\n", i, k, Low[i*(*M/2)+k]);
	 printf("High[%d][%d] = %f\n", i, k, High[i*(*M/2)+k]); */
    }
    free(data);
  }
  
  free(Wout);
  free(Vout);
  
  /*
   *  Perform one-dimensional DWT on rows (length N).
   */
  
  Wout = (double *) malloc((*N) * sizeof(double));
  Vout = (double *) malloc((*N) * sizeof(double));
  
  for(i = 0; i < (int) *M/2; i++) {
    /*
     *  Must take row from "Low" and place into vector for DWT.
     */
    data = (double *) malloc((*N) * sizeof(double));
    for(j = 0; j < *N; j++) {
      data[j] = Low[i+j*(*M/2)];
      /* printf("Low[%d][%d] = %f\n", i, j, Low[i+j*(*M/2)]); */
    }
    /*
     *  Perform DWT and read into final "Low" matrices.
     */
    dwt(data, N, L, h, g, Wout, Vout);
    for(k = 0; k < (int) *N/2; k++) {
      LL[i+k*(*N/2)] = Vout[k]; 
      HL[i+k*(*N/2)] = Wout[k];
      /* printf("LL[%d][%d] = %f\n", i, k, LL[i+k*(*N/2)]);
	 printf("LH[%d][%d] = %f\n", i, k, HL[i+k*(*N/2)]); */
    }
    free(data);
    
    /*
     *  Must take row from "High" and place into vector for DWT.
     */
    data = (double *) malloc((*N) * sizeof(double));
    for(j = 0; j < *N; j++) {
      data[j] = High[i+j*(*M/2)];
      /* printf("High[%d][%d] = %f\n", j, i, High[i+j*(*M/2)]); */
    }
    /*
     *  Perform DWT and read into final "High" matrices.
     */
    dwt(data, N, L, h, g, Wout, Vout);
    for(k = 0; k < (int) *N/2; k++) {
      LH[i+k*(*N/2)] = Vout[k]; 
      HH[i+k*(*N/2)] = Wout[k];
      /* printf("HL[%d][%d] = %f\n", i, k, LH[i+k*(*N/2)]);
	 printf("HH[%d][%d] = %f\n", i, k, HH[i+k*(*N/2)]); */
    }
    free(data);
  }
  
  free(Wout);
  free(Vout);
  free(Low);
  free(High);
}

/***************************************************************************
 ***************************************************************************
   printdvec()
 ***************************************************************************
 ***************************************************************************/
/*
void printdvec(double *v, int n)
{ 
  int i;

  for(i = 0; i <= n-1; i++)
    printf("%f ", v[i]);
  printf("\n");
}
*/
/***************************************************************************
 ***************************************************************************
   2D iDWT
 ***************************************************************************
 ***************************************************************************/

void two_D_idwt(double *LL, double *LH, double *HL, double *HH, int *M, 
		int *N, int *L, double *h, double *g, double *image)
{
  int i, j, k;
  /*
  int debug = 0;
  */
  double *Win, *Vin, *Low, *High, *Xout;
  
  Low = (double *) malloc((*M)*2*(*N) * sizeof(double));
  High = (double *) malloc((*M)*2*(*N) * sizeof(double));
  
  Win = (double *) malloc((*N) * sizeof(double));
  Vin = (double *) malloc((*N) * sizeof(double));
  Xout = (double *) malloc(2*(*N) * sizeof(double));
  
  for(i = 0; i < *M; i++) {
    /*
     *  Must take row from LL and HL and place into vectors for iDWT.
     */
    for(j = 0; j < *N; j++) {
      Win[j] = HL[i+j*(*M)];
      Vin[j] = LL[i+j*(*M)];
    }
    
    idwt(Win, Vin, N, L, h, g, Xout);
    
    for(k = 0; k < 2*(*N); k++) {
      Low[i+k*(*M)] = Xout[k];
      /* if(debug) printf("Low[%d][%d] = %f\n", k, i, Low[i+k*(*M)]); */
    }
    
    /*
     *  Must take row from LH and HH and place into vectors for iDWT.
     */
    for(j = 0; j < *N; j++) {
      Win[j] = HH[i+j*(*M)];
      Vin[j] = LH[i+j*(*M)];
    }
    
    idwt(Win, Vin, N, L, h, g, Xout);
    
    for(k = 0; k < 2*(*N); k++) {
      High[i+k*(*M)] = Xout[k];
      /* if(debug) printf("High[%d][%d] = %f\n", k, i, High[i+k*(*M)]); */
    }
    
  }
  
  free(Vin);
  free(Win);
  free(Xout);
  
  Vin = (double *) malloc((*M) * sizeof(double));
  Win = (double *) malloc((*M) * sizeof(double));
  Xout = (double *) malloc(2*(*M) * sizeof(double));
  
  for(i = 0; i < 2*(*N); i++) {
    /*
     *  Must take columns from High and Low and place into vectors for iDWT.
     */
    for(k = 0; k < *M; k++) {
      Vin[k] = Low[i*(*M)+k];
      Win[k] = High[i*(*M)+k];
    }
    
    idwt(Win, Vin, M, L, h, g, Xout);
    
    for(j = 0; j < 2*(*M); j++) 
      image[i*2*(*M)+j] = Xout[j];
  }
  
  free(Vin);
  free(Win);
  free(Xout);
  free(Low);
  free(High);
}

/***************************************************************************
 ***************************************************************************
   2D MODWT 
 ***************************************************************************
 ***************************************************************************/

void two_D_modwt(double *X, int *M, int *N, int *J, int *L, double *h, 
		 double *g, double *LL, double *LH, double *HL, double *HH)
{
  int i, j, k, index;
  /*
  int debug = 0;
  */
  double *data, *Wout, *Vout, *Low, *High;
  
  /*
   *  Perform one-dimensional MODWT on columns (length M).
   */
  
  Wout = (double *) malloc((*M) * sizeof(double));
  Vout = (double *) malloc((*M) * sizeof(double));
  
  /*
   *  Create temporary "matrices" to store MODWT of columns.
   */
  Low = (double *) malloc((*N*(*M)) * sizeof(double));
  High = (double *) malloc((*N*(*M)) * sizeof(double));
  
  for(i = 0; i < *N; i++) {
    /*
     *  Must take column from X and place into vector for MODWT.
     */
    data = (double *) malloc((*M) * sizeof(double));
    for(j = 0; j < *M; j++) {
      /* index = i * (*N) + j; */
      index = i * (*M) + j;
      data[j] = X[index];
      /* if(debug) printf("X[%d][%d] = %f\n", i, j, X[index]); */
    }
    /*
     *  Perform MODWT and read into temporary matrices.
     */
    modwt(data, M, J, L, h, g, Wout, Vout);
    for(k = 0; k < *M; k++) {
      /* index = i * (*N) + k; */
      index = i * (*M) + k;
      Low[index] = Vout[k]; 
      High[index] = Wout[k];
      /*
       *	if(debug) {
       *	printf("Low[%d][%d] = %f\n", i, k, Low[index]);
       *	printf("High[%d][%d] = %f\n", i, k, High[index]);
       *        }
       */
    }
    free(data);
  }
  
  free(Wout);
  free(Vout);
  
  /*
   *  Perform one-dimensional MODWT on rows (length N).
   */
  
  Wout = (double *) malloc((*N) * sizeof(double));
  Vout = (double *) malloc((*N) * sizeof(double));
  
  for(i = 0; i < *M; i++) {
    /*
     *  Must take row from "Low" and place into vector for DWT.
     */
    data = (double *) malloc((*N) * sizeof(double));
    for(j = 0; j < *N; j++) {
      index = i + j * (*M);
      data[j] = Low[index];
      /* if(debug) printf("Low[%d][%d] = %f\n", i, j, Low[index]); */
    }
    /*
     *  Perform MODWT and read into final "Low" matrices.
     */
    modwt(data, N, J, L, h, g, Wout, Vout);
    for(k = 0; k < *N; k++) {
      index = i + k * (*M);
      LL[index] = Vout[k]; 
      LH[index] = Wout[k];
      /*
       *	if(debug) {
       *	printf("LL[%d][%d] = %f\n", i, k, LL[index]);
       *	printf("LH[%d][%d] = %f\n", i, k, LH[index]);
       *        }
       */
    }
    free(data);

    /*
     *  Must take row from "High" and place into vector for MODWT.
     */
    data = (double *) malloc((*N) * sizeof(double));
    for(j = 0; j < *N; j++) {
      index = i + j * (*M);
      data[j] = High[index];
      /* if(debug) printf("High[%d][%d] = %f\n", j, i, High[index]); */
    }
    /* 
     *  Perform MODWT and read into final "High" matrices.
     */
    modwt(data, N, J, L, h, g, Wout, Vout);
    for(k = 0; k < *N; k++) {
      index = i + k * (*M);
      HL[index] = Vout[k]; 
      HH[index] = Wout[k];
      /* 
       *	 if(debug) {
       *	 printf("HL[%d][%d] = %f\n", i, k, HL[index]);
       *	 printf("HH[%d][%d] = %f\n", i, k, HH[index]);
       *         }
       */
    }
    free(data);
  }
  
  free(Wout);
  free(Vout);
  free(Low);
  free(High);
}

/***************************************************************************
 ***************************************************************************
   2D iMODWT
 ***************************************************************************
 ***************************************************************************/

void two_D_imodwt(double *LL, double *LH, double *HL, double *HH, int *M, 
		  int *N, int *J, int *L, double *h, double *g, 
		  double *image)
{
  int i, j, k, index;
  double *Win, *Vin, *Low, *High, *Xout;

  Low = (double *) malloc((*M)*(*N) * sizeof(double));
  High = (double *) malloc((*M)*(*N) * sizeof(double));

  Win = (double *) malloc((*N) * sizeof(double));
  Vin = (double *) malloc((*N) * sizeof(double));
  Xout = (double *) malloc((*N) * sizeof(double));
  
  for(i = 0; i < *M; i++) {
    /*
     *  Must take row from LL and LH and place into vectors for iMODWT.
     */
    for(j = 0; j < *N; j++) {
      index = i + j * (*M);
      Win[j] = LH[index];
      Vin[j] = LL[index];
    }
    
    imodwt(Win, Vin, N, J, L, h, g, Xout);
    
    for(k = 0; k < *N; k++) {
      index = i + k * (*M);
      Low[index] = Xout[k];
    }

    /*
     *  Must take row from HL and HH and place into vectors for iMODWT.
     */
    for(j = 0; j < *N; j++) {
      index = i + j * (*M);
      Win[j] = HH[index];
      Vin[j] = HL[index];
    }

    imodwt(Win, Vin, N, J, L, h, g, Xout);

    for(k = 0; k < *N; k++) {
      index = i + k * (*M);
      High[index] = Xout[k];
    }
  }

  free(Vin);
  free(Win);
  free(Xout);

  Vin = (double *) malloc((*M) * sizeof(double));
  Win = (double *) malloc((*M) * sizeof(double));
  Xout = (double *) malloc((*M) * sizeof(double));

  for(i = 0; i < *N; i++) {
    /*
     *  Must take columns from High and Low and place into vectors for iMODWT.
     */
    for(k = 0; k < *M; k++) {
      /* index = i * (*N) + k; */
      index = i * (*M) + k;
      Vin[k] = Low[index];
      Win[k] = High[index];
    }

    imodwt(Win, Vin, M, J, L, h, g, Xout);

    for(j = 0; j < *M; j++) {
      /* index = i * (*N) + j; */
      index = i * (*M) + j;
      image[index] = Xout[j];
    }
  }

  free(Vin);
  free(Win);
  free(Xout);
  free(Low);
  free(High);
}
