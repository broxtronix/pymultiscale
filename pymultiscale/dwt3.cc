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
   3D DWT 
 ***************************************************************************
 ***************************************************************************/

void three_D_dwt(double *X, int *NX, int *NY, int *NZ, int *L, 
		 double *h, double *g, double *LLL, double *HLL, 
		 double *LHL, double *LLH, double *HHL, double *HLH, 
		 double *LHH, double *HHH)
{
  int i, j, k, l, index;
  /*
  int printall = 0;
  */
  double *data, *Wout, *Vout, *Xl, *Xh, *Yll, *Ylh, *Yhl, *Yhh;

  /* printf("Original Data (N = %d)...\n", *NX * (*NY) * (*NZ));
     printdvec(X, *NX * (*NY) * (*NZ)); */

  /*
   *  Perform one-dimensional DWT on first dimension (length NX).
   */
  Wout = (double *) malloc((*NX) * sizeof(double));
  Vout = (double *) malloc((*NX) * sizeof(double));
  data = (double *) malloc((*NX) * sizeof(double));

  /*
   *  Create temporary "hyperrectangles" to store DWT of X-dimension.
   */
  Xl = (double *) malloc((*NZ*(*NY)*(*NX/2)) * sizeof(double));
  Xh = (double *) malloc((*NZ*(*NY)*(*NX/2)) * sizeof(double));
  
  for(i = 0; i < *NZ*(*NY); i++) {
    /*
     *  Must take column from X-dimension and place into vector for DWT.
     */
    for(j = 0; j < *NX; j++) {
      index = i * (*NX) + j;
      data[j] = X[index];
      /* printf("X[%d][%d] = %f\n", i, j, X[index]); */
    }
    /*
     *  Perform DWT and read into temporary matrices.
     */
    dwt(data, NX, L, h, g, Wout, Vout);
    for(j = 0; j < (int) *NX/2; j++) {
      index = i * (*NX/2) + j;
      Xl[index] = Vout[j]; 
      Xh[index] = Wout[j];
      /* printf("Low[%d][%d] = %f\n", i, j, Low[index]);
	 printf("High[%d][%d] = %f\n", i, j, High[index]); */
    }
  }

  free(Wout);
  free(Vout);
  free(data);

  /*
    printf("X Low...\n");
    printdvec(Xl, (*NX/2) * (*NY) * (*NZ));
    printf("X High...\n");
    printdvec(Xh, (*NX/2) * (*NY) * (*NZ));
  */

  /*
   *  Perform one-dimensional DWT on second dimension (length NY).
   */
  Wout = (double *) malloc((*NY) * sizeof(double));
  Vout = (double *) malloc((*NY) * sizeof(double));
  data = (double *) malloc((*NY) * sizeof(double));

  /*
   *  Create temporary "hyperrectangles" to store DWT of X-dimension.
   */
  Yll = (double *) malloc((*NZ*(*NY/2)*(*NX/2)) * sizeof(double));
  Ylh = (double *) malloc((*NZ*(*NY/2)*(*NX/2)) * sizeof(double));
  Yhl = (double *) malloc((*NZ*(*NY/2)*(*NX/2)) * sizeof(double));
  Yhh = (double *) malloc((*NZ*(*NY/2)*(*NX/2)) * sizeof(double));

  k = 0;
  l = 0;
  for(i = 0; i < *NZ * (int) *NX/2; i++) {
    /* 
     * Must adjust for 3D array structure.
     *   k: vertical dimension (Z) adjustment when reading in data
     *   l: vertical dimension (Z) adjustment when writing wavelet coeffs.
     */
    if(i > 0 && fmod(i, (int) *NX/2) == 0.0) {
      k = k + (*NY - 1) * ((int) *NX/2);
      l = l + ((int) *NY/2 - 1) * ((int) *NX/2);
    }
    /*
      printf("fmod(%d, %d) = %f\n", i, (int) *NX/2, fmod(i, (int) *NX/2));
      printf("i = %d\tk = %d\tl = %d\n", i, k, l);
    */
    /*
     *  Must take row from "Xl" and place into vector for DWT.
     */
    for(j = 0; j < *NY; j++) {
      index = i + j * ((int) *NX/2) + k;
      data[j] = Xl[index];
    }
    /*
     *  Perform DWT and read into temporary "Yll" and "Yhl" hyperrectangles.
     */
    dwt(data, NY, L, h, g, Wout, Vout);
    for(j = 0; j < (int) *NY/2; j++) {
      index = i + j * ((int) *NX/2) + l;
      Yll[index] = Vout[j]; 
      Ylh[index] = Wout[j];
      /* 
	 if(printall == 1)
	 printf("Y.LL[%d][%d] = %f\nY.HL[%d][%d] = %f\n", i, j, 
	 Yll[index], i, j, Ylh[index]);
      */
    }

    /*
     *  Must take row from "Xh" and place into vector for DWT.
     */
    for(j = 0; j < *NY; j++) {
      index = i + j * ((int) *NX/2) + k;
      data[j] = Xh[index];
    }
    /*
     *  Perform DWT and read into temporary "Yhl" and "Yhh" hyperrectangles.
     */
    dwt(data, NY, L, h, g, Wout, Vout);
    for(j = 0; j < (int) *NY/2; j++) {
      index = i + j * ((int) *NX/2) + l;
      Yhl[index] = Vout[j]; 
      Yhh[index] = Wout[j];
      /* 
	 if(printall == 1)
	 printf("Y.LH[%d][%d] = %f\nY.HH[%d][%d] = %f\n", i, j, 
	 Yhl[index], i, j, Yhh[index]);
      */
    }
  }

  free(Wout);
  free(Vout);
  free(data);
  free(Xl);
  free(Xh);

  /*
    printf("Y Low-Low...\n");
    printdvec(Yll, (*NX/2) * (*NY/2) * (*NZ));
    printf("Y High-Low...\n");
    printdvec(Yhl, (*NX/2) * (*NY/2) * (*NZ));
    printf("Y Low-High...\n");
    printdvec(Ylh, (*NX/2) * (*NY/2) * (*NZ));
    printf("Y High-High...\n");
    printdvec(Yhh, (*NX/2) * (*NY/2) * (*NZ));
  */

  /*
   *  Perform one-dimensional DWT on third dimension (length NZ).
   */
  Wout = (double *) malloc((*NZ) * sizeof(double));
  Vout = (double *) malloc((*NZ) * sizeof(double));
  data = (double *) malloc((*NZ) * sizeof(double));

  for(i = 0; i < (int) *NY/2 * (int) *NX/2; i++) {
    /*
     *  Must take vertical column from "Yll" and place into vector for DWT.
     */
    for(j = 0; j < *NZ; j++) {
      index = i + j * ((int) *NY/2 * (int) *NX/2);
      data[j] = Yll[index];
    }
    /*
     *  Perform DWT and read into final "LLL" and "LLH" hyperrectangles.
     */
    dwt(data, NZ, L, h, g, Wout, Vout);
    for(j = 0; j < (int) *NZ/2; j++) {
      index = i + j * ((int) *NY/2 * (int) *NX/2);
      LLL[index] = Vout[j]; 
      LLH[index] = Wout[j];
      /*
	if(printall == 1)
	printf("LLL[%d][%d] = %f\nLLH[%d][%d] = %f\n", i, j, 
	LLL[index], i, j, LLH[index]);
      */
    }
    /*
     *  Must take row from "Yhl" and place into vector for DWT.
     */
    for(j = 0; j < *NZ; j++) {
      index = i + j * ((int) *NY/2 * (int) *NX/2);
      data[j] = Yhl[index];
    }
    /*
     *  Perform DWT and read into final "HLL" and "HLH" hyperrectangles.
     */
    dwt(data, NZ, L, h, g, Wout, Vout);
    for(j = 0; j < (int) *NZ/2; j++) {
      index = i + j * ((int) *NY/2 * (int) *NX/2);
      HLL[index] = Vout[j]; 
      HLH[index] = Wout[j];
      /* printf("HLL[%d][%d] = %f\n", i, j, HLL[index]);
	 printf("HLH[%d][%d] = %f\n", i, j, HLH[index]); */
    }
    /*
     *  Must take row from "Ylh" and place into vector for DWT.
     */
    for(j = 0; j < *NZ; j++) {
      index = i + j * ((int) *NY/2 * (int) *NX/2);
      data[j] = Ylh[index];
    }
    /*
     *  Perform DWT and read into final "LHL" and "LHH" hyperrectangles.
     */
    dwt(data, NZ, L, h, g, Wout, Vout);
    for(j = 0; j < (int) *NZ/2; j++) {
      index = i + j * ((int) *NY/2 * (int) *NX/2);
      LHL[index] = Vout[j]; 
      LHH[index] = Wout[j];
      /* printf("LHH[%d][%d] = %f\n", i, j, LHH[index]);
	 printf("LHL[%d][%d] = %f\n", i, j, LHL[index]); */
    }
    /*
     *  Must take row from "Yhh" and place into vector for DWT.
     */
    for(j = 0; j < *NZ; j++) {
      index = i + j * ((int) *NY/2 * (int) *NX/2);
      data[j] = Yhh[index];
    }
    /*
     *  Perform DWT and read into final "HHL" and "HHH" hyperrectangles.
     */
    dwt(data, NZ, L, h, g, Wout, Vout);
    for(j = 0; j < (int) *NZ/2; j++) {
      index = i + j * ((int) *NY/2 * (int) *NX/2);
      HHL[index] = Vout[j]; 
      HHH[index] = Wout[j];
      /* printf("HHH[%d][%d] = %f\n", i, j, HHH[index]);
	 printf("HHL[%d][%d] = %f\n", i, j, HHL[index]); */
    }
  }

  free(Wout);
  free(Vout);
  free(data);
  free(Yll);
  free(Ylh);
  free(Yhl);
  free(Yhh);
}

/***************************************************************************
 ***************************************************************************
   3D iDWT
 ***************************************************************************
 ***************************************************************************/

void three_D_idwt(double *LLL, double *HLL, double *LHL, double *LLH, 
		  double *HHL, double *HLH, double *LHH, double *HHH, 
		  int *NX, int *NY, int *NZ, int *L, double *h, 
		  double *g, double *image)
{
  int i, j, k, l;
  /*
  int printall = 0;
  */
  double *Win, *Vin, *Xl, *Xh, *Yll, *Ylh, *Yhl, *Yhh, *Xout;

  /*
   *  Create temporary "hyperrectangles" to store iDWT of Z-dimension.
   */
  Yll = (double *) malloc((2*(*NZ)*(*NY)*(*NX)) * sizeof(double));
  Ylh = (double *) malloc((2*(*NZ)*(*NY)*(*NX)) * sizeof(double));
  Yhl = (double *) malloc((2*(*NZ)*(*NY)*(*NX)) * sizeof(double));
  Yhh = (double *) malloc((2*(*NZ)*(*NY)*(*NX)) * sizeof(double));

  Win = (double *) malloc((*NZ) * sizeof(double));
  Vin = (double *) malloc((*NZ) * sizeof(double));
  Xout = (double *) malloc(2*(*NZ) * sizeof(double));
  
  for(i = 0; i < *NY * (*NX); i++) {
    /*
     *  Must take row from LLL and LLH and place into vectors for iDWT.
     */
    for(j = 0; j < *NZ; j++) {
      Win[j] = LLH[i + j * (*NY) * (*NX)];
      Vin[j] = LLL[i + j * (*NY) * (*NX)];
    }
    idwt(Win, Vin, NZ, L, h, g, Xout);
    for(j = 0; j < 2 * (*NZ); j++)
      Yll[i + j * (*NY) * (*NX)] = Xout[j];

    /*
     *  Must take row from HLL and HLH and place into vectors for iDWT.
     */
    for(j = 0; j < *NZ; j++) {
      Win[j] = HLH[i + j * (*NY) * (*NX)];
      Vin[j] = HLL[i + j * (*NY) * (*NX)];
    }
    idwt(Win, Vin, NZ, L, h, g, Xout);
    for(j = 0; j < 2 * (*NZ); j++)
      Yhl[i + j * (*NY) * (*NX)] = Xout[j];

    /*
     *  Must take row from LHL and LHH and place into vectors for iDWT.
     */
    for(j = 0; j < *NZ; j++) {
      Win[j] = LHH[i + j * (*NY) * (*NX)];
      Vin[j] = LHL[i + j * (*NY) * (*NX)];
    }
    idwt(Win, Vin, NZ, L, h, g, Xout);
    for(j = 0; j < 2 * (*NZ); j++)
      Ylh[i + j * (*NY) * (*NX)] = Xout[j];

    /*
     *  Must take row from HHL and HHH and place into vectors for iDWT.
     */
    for(j = 0; j < *NZ; j++) {
      Win[j] = HHH[i + j * (*NY) * (*NX)];
      Vin[j] = HHL[i + j * (*NY) * (*NX)];
    }
    idwt(Win, Vin, NZ, L, h, g, Xout);
    for(j = 0; j < 2 * (*NZ); j++)
      Yhh[i + j * (*NY) * (*NX)] = Xout[j];
  }

  free(Vin);
  free(Win);
  free(Xout);

  /*
    printf("Y Low-Low...\n");
    printdvec(Yll, (*NX) * (*NY) * 2 * (*NZ));
    printf("Y High-Low...\n");
    printdvec(Yhl, (*NX) * (*NY) * 2 * (*NZ));
    printf("Y Low-High...\n");
    printdvec(Ylh, (*NX) * (*NY) * 2 * (*NZ));
    printf("Y High-High...\n");
    printdvec(Yhh, (*NX) * (*NY) * 2 * (*NZ));
  */

  Xl = (double *) malloc((2*(*NZ)*2*(*NY)*(*NX)) * sizeof(double));
  Xh = (double *) malloc((2*(*NZ)*2*(*NY)*(*NX)) * sizeof(double));

  Vin = (double *) malloc((*NY) * sizeof(double));
  Win = (double *) malloc((*NY) * sizeof(double));
  Xout = (double *) malloc(2*(*NY) * sizeof(double));

  k = 0;
  l = 0;
  for(i = 0; i < 2 * (*NZ) * (*NX); i++) {
    /* 
     * Must adjust for 3D array structure.
     *   k: vertical dimension (Z) adjustment when reading in data
     *   l: vertical dimension (Z) adjustment when writing wavelet coeffs.
     */
    if(i > 0 && fmod(i, *NX) == 0.0) {
      k = k + (*NY - 1) * (*NX);
      l = l + (2 * (*NY) - 1) * (*NX);
    }
    /* printf("k = %d \t l = %d\n", k, l); */

    /*
     *  Must take columns from Yll and Ylh and place into vectors for iDWT.
     */
    for(j = 0; j < *NY; j++) {
      Vin[j] = Yll[i + j * (*NX) + k];
      Win[j] = Ylh[i + j * (*NX) + k];
    }
    idwt(Win, Vin, NY, L, h, g, Xout);
    for(j = 0; j < 2 * (*NY); j++) 
      Xl[i + j * (*NX) + l] = Xout[j];
    /*
     *  Must take columns from Yhl and Yhh and place into vectors for iDWT.
     */
    for(j = 0; j < *NY; j++) {
      Vin[j] = Yhl[i + j * (*NX) + k];
      Win[j] = Yhh[i + j * (*NX) + k];
    }
    idwt(Win, Vin, NY, L, h, g, Xout);
    for(j = 0; j < 2 * (*NY); j++) 
      Xh[i + j * (*NX) + l] = Xout[j];
  }

  /*
    printf("X Low...\n");
    printdvec(Xl, (*NX) * 2 * (*NY) * 2 * (*NZ));
    printf("X High...\n");
    printdvec(Xh, (*NX) * 2 * (*NY) * 2 * (*NZ));
  */

  free(Vin);
  free(Win);
  free(Xout);

  free(Yll);
  free(Ylh);
  free(Yhl);
  free(Yhh);

  Vin = (double *) malloc((*NX) * sizeof(double));
  Win = (double *) malloc((*NX) * sizeof(double));
  Xout = (double *) malloc(2*(*NX) * sizeof(double));

  for(i = 0; i < 2 * (*NZ) * 2 * (*NY); i++) {
    /*
     *  Must take columns from Xl and Xh and place into vectors for iDWT.
     */
    for(j = 0; j < *NX; j++) {
      Vin[j] = Xl[i * (*NX) + j];
      Win[j] = Xh[i * (*NX) + j];
    }
    idwt(Win, Vin, NX, L, h, g, Xout);
    for(j = 0; j < 2 * (*NX); j++)
      image[i * 2 * (*NX) + j] = Xout[j];
  }

  free(Vin);
  free(Win);
  free(Xout);
  free(Xl);
  free(Xh);
}

/***************************************************************************
 ***************************************************************************
   3D MODWT
 ***************************************************************************
 ***************************************************************************/

void three_D_modwt(double *X, int *NX, int *NY, int *NZ, int *J, int *L, 
		   double *h, double *g, double *LLL, double *HLL, 
		   double *LHL, double *LLH, double *HHL, double *HLH, 
		   double *LHH, double *HHH)
{
  int i, j, k, index;
  double *data, *Wout, *Vout, *Xl, *Xh, *Yll, *Ylh, *Yhl, *Yhh;

  /*
    printf("Original Data (N = %d)...\n", *NX * (*NY) * (*NZ));
    printdvec(X, *NX * (*NY) * (*NZ));
  */

  /*
   *  Perform one-dimensional MODWT on first dimension (length NX).
   */
  Wout = (double *) malloc((*NX) * sizeof(double));
  Vout = (double *) malloc((*NX) * sizeof(double));
  data = (double *) malloc((*NX) * sizeof(double));

  /*
   *  Create temporary "hyperrectangles" to store MODWT of X-dimension.
   */
  Xl = (double *) malloc((*NZ*(*NY)*(*NX)) * sizeof(double));
  Xh = (double *) malloc((*NZ*(*NY)*(*NX)) * sizeof(double));
  
  for(i = 0; i < *NZ*(*NY); i++) {
    /*
     *  Must take column from X-dimension and place into vector for DWT.
     */
    for(j = 0; j < *NX; j++) {
      index = i * (*NX) + j;
      data[j] = X[index];
    }
    /*
     *  Perform MODWT and read into temporary matrices.
     */
    modwt(data, NX, J, L, h, g, Wout, Vout);
    for(j = 0; j < *NX; j++) {
      index = i * (*NX) + j;
      Xl[index] = Vout[j]; 
      Xh[index] = Wout[j];
    }
  }

  free(Wout);
  free(Vout);
  free(data);

  /*
    printf("X Low...\n");
    printdvec(Xl, (*NX) * (*NY) * (*NZ));
    printf("X High...\n");
    printdvec(Xh, (*NX) * (*NY) * (*NZ));
  */

  /*
   *  Perform one-dimensional MODWT on second dimension (length NY).
   */
  Wout = (double *) malloc((*NY) * sizeof(double));
  Vout = (double *) malloc((*NY) * sizeof(double));
  data = (double *) malloc((*NY) * sizeof(double));

  /*
   *  Create temporary "hyperrectangles" to store MODWT of X-dimension.
   */
  Yll = (double *) malloc((*NZ*(*NY)*(*NX)) * sizeof(double));
  Ylh = (double *) malloc((*NZ*(*NY)*(*NX)) * sizeof(double));
  Yhl = (double *) malloc((*NZ*(*NY)*(*NX)) * sizeof(double));
  Yhh = (double *) malloc((*NZ*(*NY)*(*NX)) * sizeof(double));

  k = 0;
  for(i = 0; i < *NZ * (*NX); i++) {
    /* 
     * Must adjust for 3D array structure.
     *   k: vertical dimension (Z) adjustment when reading in data
     *   l: vertical dimension (Z) adjustment when writing wavelet coeffs.
     */
    if(i > 0 && fmod(i, *NX) == 0.0)
      k = k + (*NY - 1) * (*NX);
    /*
     *  Must take row from "Xl" and place into vector for DWT.
     */
    for(j = 0; j < *NY; j++) {
      index = i + j * (*NX) + k;
      data[j] = Xl[index];
    }
    /*
     *  Perform MODWT and read into temporary "Yll" and "Ylh" hyperrectangles.
     */
    modwt(data, NY, J, L, h, g, Wout, Vout);
    for(j = 0; j < *NY; j++) {
      index = i + j * (*NX) + k;
      Yll[index] = Vout[j]; 
      Ylh[index] = Wout[j];
    }
    /*
     *  Must take row from "Xh" and place into vector for DWT.
     */
    for(j = 0; j < *NY; j++) {
      index = i + j * (*NX) + k;
      data[j] = Xh[index];
    }
    /*
     *  Perform MODWT and read into temporary "Yhl" and "Yhh" hyperrectangles.
     */
    modwt(data, NY, J, L, h, g, Wout, Vout);
    for(j = 0; j < *NY; j++) {
      index = i + j * (*NX) + k;
      Yhl[index] = Vout[j]; 
      Yhh[index] = Wout[j];
    }
  }

  free(Wout);
  free(Vout);
  free(data);
  free(Xl);
  free(Xh);

  /*
    printf("Y Low-Low...\n");
    printdvec(Yll, (*NX) * (*NY) * (*NZ));
    printf("Y High-Low...\n");
    printdvec(Yhl, (*NX) * (*NY) * (*NZ));
    printf("Y Low-High...\n");
    printdvec(Ylh, (*NX) * (*NY) * (*NZ));
    printf("Y High-High...\n");
    printdvec(Yhh, (*NX) * (*NY) * (*NZ));
  */

  /*
   *  Perform one-dimensional MODWT on third dimension (length NZ).
   */
  Wout = (double *) malloc((*NZ) * sizeof(double));
  Vout = (double *) malloc((*NZ) * sizeof(double));
  data = (double *) malloc((*NZ) * sizeof(double));

  for(i = 0; i < *NY * (*NX); i++) {
    /*
     *  Must take vertical column from "Yll" and place into vector for MODWT.
     */
    for(j = 0; j < *NZ; j++) {
      index = i + j * (*NY) * (*NX);
      data[j] = Yll[index];
    }
    /*
     *  Perform MODWT and read into final "LLL" and "LLH" hyperrectangles.
     */
    modwt(data, NZ, J, L, h, g, Wout, Vout);
    for(j = 0; j < *NZ; j++) {
      index = i + j * (*NY) * (*NX);
      LLL[index] = Vout[j]; 
      LLH[index] = Wout[j];
    }
    /*
     *  Must take row from "Yhl" and place into vector for MODWT.
     */
    for(j = 0; j < *NZ; j++) {
      index = i + j * (*NY) * (*NX);
      data[j] = Yhl[index];
    }
    /*
     *  Perform MODWT and read into final "HLL" and "HLH" hyperrectangles.
     */
    modwt(data, NZ, J, L, h, g, Wout, Vout);
    for(j = 0; j < *NZ; j++) {
      index = i + j * (*NY) * (*NX);
      HLL[index] = Vout[j]; 
      HLH[index] = Wout[j];
    }
    /*
     *  Must take row from "Ylh" and place into vector for MODWT.
     */
    for(j = 0; j < *NZ; j++) {
      index = i + j * (*NY) * (*NX);
      data[j] = Ylh[index];
    }
    /*
     *  Perform MODWT and read into final "LHL" and "LHH" hyperrectangles.
     */
    modwt(data, NZ, J, L, h, g, Wout, Vout);
    for(j = 0; j < *NZ; j++) {
      index = i + j * (*NY) * (*NX);
      LHL[index] = Vout[j]; 
      LHH[index] = Wout[j];
    }
    /*
     *  Must take row from "Yhh" and place into vector for MODWT.
     */
    for(j = 0; j < *NZ; j++) {
      index = i + j * (*NY) * (*NX);
      data[j] = Yhh[index];
    }
    /*
     *  Perform MODWT and read into final "LHH" and "HHH" hyperrectangles.
     */
    modwt(data, NZ, J, L, h, g, Wout, Vout);
    for(j = 0; j < *NZ; j++) {
      index = i + j * (*NY) * (*NX);
      HHL[index] = Vout[j]; 
      HHH[index] = Wout[j];
    }
  }

  free(Wout);
  free(Vout);
  free(data);
  free(Yll);
  free(Ylh);
  free(Yhl);
  free(Yhh);
}

/***************************************************************************
 ***************************************************************************
   3D iMODWT
 ***************************************************************************
 ***************************************************************************/

void three_D_imodwt(double *LLL, double *HLL, double *LHL, double *LLH, 
		    double *HHL, double *HLH, double *LHH, double *HHH, 
		    int *NX, int *NY, int *NZ, int *J, int *L, double *h, 
		    double *g, double *image)
{
  int i, j, k, index;
  double *Win, *Vin, *Xl, *Xh, *Yll, *Ylh, *Yhl, *Yhh, *Xout;

  /*
   *  Create temporary "hyperrectangles" to store imodwt of Z-dimension.
   */
  Yll = (double *) malloc(((*NZ)*(*NY)*(*NX)) * sizeof(double));
  Ylh = (double *) malloc(((*NZ)*(*NY)*(*NX)) * sizeof(double));
  Yhl = (double *) malloc(((*NZ)*(*NY)*(*NX)) * sizeof(double));
  Yhh = (double *) malloc(((*NZ)*(*NY)*(*NX)) * sizeof(double));

  Win = (double *) malloc((*NZ) * sizeof(double));
  Vin = (double *) malloc((*NZ) * sizeof(double));
  Xout = (double *) malloc((*NZ) * sizeof(double));
  
  for(i = 0; i < *NY * (*NX); i++) {
    /*
     *  Must take row from LLL and LLH and place into vectors for imodwt.
     */
    for(j = 0; j < *NZ; j++) {
      index = i + j * (*NY) * (*NX);
      Win[j] = LLH[index];
      Vin[j] = LLL[index];
    }
    imodwt(Win, Vin, NZ, J, L, h, g, Xout);
    for(j = 0; j < *NZ; j++) {
      index = i + j * (*NY) * (*NX);
      Yll[index] = Xout[j];
    }
    /*
     *  Must take row from HLL and HLH and place into vectors for imodwt.
     */
    for(j = 0; j < *NZ; j++) {
      index = i + j * (*NY) * (*NX);
      Win[j] = HLH[index];
      Vin[j] = HLL[index];
    }
    imodwt(Win, Vin, NZ, J, L, h, g, Xout);
    for(j = 0; j < *NZ; j++) {
      index = i + j * (*NY) * (*NX);
      Yhl[index] = Xout[j];
    }
    /*
     *  Must take row from LHL and LHH and place into vectors for imodwt.
     */
    for(j = 0; j < *NZ; j++) {
      index = i + j * (*NY) * (*NX);
      Win[j] = LHH[index];
      Vin[j] = LHL[index];
    }
    imodwt(Win, Vin, NZ, J, L, h, g, Xout);
    for(j = 0; j < *NZ; j++) {
      index = i + j * (*NY) * (*NX);
      Ylh[index] = Xout[j];
    }
    /*
     *  Must take row from HHL and HHH and place into vectors for imodwt.
     */
    for(j = 0; j < *NZ; j++) {
      index = i + j * (*NY) * (*NX);
      Win[j] = HHH[index];
      Vin[j] = HHL[index];
    }
    imodwt(Win, Vin, NZ, J, L, h, g, Xout);
    for(j = 0; j < *NZ; j++) {
      index = i + j * (*NY) * (*NX);
      Yhh[index] = Xout[j];
    }
  }

  free(Vin);
  free(Win);
  free(Xout);

  /*
    printf("Y Low-Low...\n");
    printdvec(Yll, (*NX) * (*NY) * (*NZ));
    printf("Y High-Low...\n");
    printdvec(Yhl, (*NX) * (*NY) * (*NZ));
    printf("Y Low-High...\n");
    printdvec(Ylh, (*NX) * (*NY) * (*NZ));
    printf("Y High-High...\n");
    printdvec(Yhh, (*NX) * (*NY) * (*NZ));
  */

  Xl = (double *) malloc(((*NZ)*(*NY)*(*NX)) * sizeof(double));
  Xh = (double *) malloc(((*NZ)*(*NY)*(*NX)) * sizeof(double));

  Vin = (double *) malloc((*NY) * sizeof(double));
  Win = (double *) malloc((*NY) * sizeof(double));
  Xout = (double *) malloc((*NY) * sizeof(double));

  k = 0;
  for(i = 0; i < (*NZ) * (*NX); i++) {
    /* 
     * Must adjust for 3D array structure.
     *   k: vertical dimension (Z) adjustment when reading in data
     */
    if(i > 0 && fmod(i, *NX) == 0.0)
      k = k + (*NY - 1) * (*NX);
    /*
     *  Must take columns from Yll and Ylh and place into vectors for imodwt.
     */
    for(j = 0; j < *NY; j++) {
      index = i + j * (*NX) + k;
      Vin[j] = Yll[index];
      Win[j] = Ylh[index];
    }
    imodwt(Win, Vin, NY, J, L, h, g, Xout);
    for(j = 0; j < (*NY); j++) {
      index = i + j * (*NX) + k;
      Xl[index] = Xout[j];
    }
    /*
     *  Must take columns from Yhl and Yhh and place into vectors for imodwt.
     */
    for(j = 0; j < *NY; j++) {
      index = i + j * (*NX) + k;
      Vin[j] = Yhl[index];
      Win[j] = Yhh[index];
    }
    imodwt(Win, Vin, NY, J, L, h, g, Xout);
    for(j = 0; j < (*NY); j++) {
      index = i + j * (*NX) + k;
      Xh[i + j * (*NX) + k] = Xout[j];
    }
  }

  /*
    printf("X Low...\n");
    printdvec(Xl, (*NX) * (*NY) * (*NZ));
    printf("X High...\n");
    printdvec(Xh, (*NX) * (*NY) * (*NZ));
  */

  free(Vin);
  free(Win);
  free(Xout);
  free(Yll);
  free(Ylh);
  free(Yhl);
  free(Yhh);

  Vin = (double *) malloc((*NX) * sizeof(double));
  Win = (double *) malloc((*NX) * sizeof(double));
  Xout = (double *) malloc((*NX) * sizeof(double));

  for(i = 0; i < (*NZ) * (*NY); i++) {
    /*
     *  Must take columns from Xl and Xh and place into vectors for imodwt.
     */
    for(j = 0; j < *NX; j++) {
      index = i * (*NX) + j;
      Vin[j] = Xl[index];
      Win[j] = Xh[index];
    }
    imodwt(Win, Vin, NX, J, L, h, g, Xout);
    for(j = 0; j < (*NX); j++) {
      index = i * (*NX) + j;
      image[index] = Xout[j];
    }
  }

  free(Vin);
  free(Win);
  free(Xout);
  free(Xl);
  free(Xh);
}
