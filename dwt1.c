#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>

/*************************************************************************/

void dwt(double *Vin, int *M, int *L, double *h, double *g, 
	 double *Wout, double *Vout)
{

  int n, t, u;

  for(t = 0; t < *M/2; t++) {
    u = 2 * t + 1;
    Wout[t] = h[0] * Vin[u];
    Vout[t] = g[0] * Vin[u];
    for(n = 1; n < *L; n++) {
      u -= 1;
      if(u < 0) u = *M - 1;
      Wout[t] += h[n] * Vin[u];
      Vout[t] += g[n] * Vin[u];
    } 
  }
}

/*************************************************************************/

void idwt(double *Win, double *Vin, int *M, int *L, double *h, double *g, 
	  double *Xout)
{

  int i, j, l, t, u;
  int m = -2, n = -1;

  for(t = 0; t < *M; t++) {
    m += 2;
    n += 2;
    u = t;
    i = 1;
    j = 0;
    Xout[m] = h[i] * Win[u] + g[i] * Vin[u];
    Xout[n] = h[j] * Win[u] + g[j] * Vin[u];
    if(*L > 2) {
      for(l = 1; l < *L/2; l++) {
	u += 1;
	if(u >= *M) u = 0;
	i += 2;
	j += 2;
	Xout[m] += h[i] * Win[u] + g[i] * Vin[u];
	Xout[n] += h[j] * Win[u] + g[j] * Vin[u];
      }
    }
  }
}

/*************************************************************************/

void modwt(double *Vin, int *N, int *j, int *L, double *ht, double *gt, 
	   double *Wout, double *Vout)
{

  int k, n, t;

  for(t = 0; t < *N; t++) {
    k = t;
    Wout[t] = ht[0] * Vin[k];
    Vout[t] = gt[0] * Vin[k];
    for(n = 1; n < *L; n++) {
      k -= (int) pow(2.0, (double) *j - 1.0);
      if(k < 0) k += *N;
      Wout[t] += ht[n] * Vin[k];
      Vout[t] += gt[n] * Vin[k];
    }
  }

}

/*************************************************************************/

void imodwt(double *Win, double *Vin, int *N, int *j, int *L, 
	    double *ht, double *gt, double *Vout)
{

  int k, n, t;

  for(t = 0; t < *N; t++) {
    k = t;
    Vout[t] = (ht[0] * Win[k]) + (gt[0] * Vin[k]);
    for(n = 1; n < *L; n++) {
      k += (int) pow(2.0, (double) *j - 1.0);
      if(k >= *N) k -= *N;
      Vout[t] += (ht[n] * Win[k]) + (gt[n] * Vin[k]);
    }
  }
}

/***************************************************************************
 ***************************************************************************
   This DWT algorithm is shifted to the left by one in order to match with 
   the interval boundary conditions.
 ***************************************************************************
 ***************************************************************************/

void dwt_shift(double *Vin, int *M, int *L, double *h, double *g, 
	       double *Wout, double *Vout)
{

  int n, t, u;

  for(t = 0; t < *M/2; t++) {
    /* u = 2 * t + 1; */
    u = 2 * t + 2;
    Wout[t] = h[0] * Vin[u];
    Vout[t] = g[0] * Vin[u];
    for(n = 1; n < *L; n++) {
      u -= 1;
      if(u < 0) u = *M - 1;
      Wout[t] += h[n] * Vin[u];
      Vout[t] += g[n] * Vin[u];
    } 
  }
}

/***************************************************************************
 ***************************************************************************
   shifted iDWT
 ***************************************************************************
 ***************************************************************************/

void idwt_shift(double *Win, double *Vin, int M, int L, double *h, 
		double *g, double *Xout)
{

  int i, j, l, t, u;
  int m = -2, n = -1;

  for(t = 0; t < M; t++) {
    m += 2;
    n += 2;
    u = t;
    i = 1;
    j = 0;
    Xout[m] = h[i] * Win[u] + g[i] * Vin[u];
    Xout[n] = h[j] * Win[u] + g[j] * Vin[u];
    if(L > 2) {
      for(l = 1; l < L/2; l++) {
	u += 1;
	if(u >= M) u = 0;
	i += 2;
	j += 2;
	Xout[m] += h[i] * Win[u] + g[i] * Vin[u];
	Xout[n] += h[j] * Win[u] + g[j] * Vin[u];
      }
    }
  }
}

