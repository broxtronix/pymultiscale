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

extern void dwt(double *Vin, int *M, int *L, double *h, double *g, 
		double *Wout, double *Vout);
extern void idwt(double *Win, double *Vin, int *M, int *L, double *h, 
		 double *g, double *Xout);
extern void modwt(double *Vin, int *N, int *j, int *L, double *ht, 
		  double *gt, double *Wout, double *Vout);
extern void imodwt(double *Win, double *Vin, int *N, int *j, int *L, 
		   double *ht, double *gt, double *Vout);

extern void two_D_dwt(double *X, int *M, int *N, int *L, double *h, double *g, 
	       double *LL, double *LH, double *HL, double *HH);
extern void two_D_idwt(double *LL, double *LH, double *HL, double *HH, int *M, 
		int *N, int *L, double *h, double *g, double *image);
extern void two_D_modwt(double *X, int *M, int *N, int *J, int *L, double *h, 
		 double *g, double *LL, double *LH, double *HL, double *HH);
extern void two_D_imodwt(double *LL, double *LH, double *HL, double *HH, int *M, 
		  int *N, int *J, int *L, double *h, double *g, 
		  double *image);

extern void three_D_dwt(double *X, int *NX, int *NY, int *NZ, int *L, 
		 double *h, double *g, double *LLL, double *HLL, 
		 double *LHL, double *LLH, double *HHL, double *HLH, 
		 double *LHH, double *HHH);
extern void three_D_idwt(double *LLL, double *HLL, double *LHL, double *LLH, 
		  double *HHL, double *HLH, double *LHH, double *HHH, 
		  int *NX, int *NY, int *NZ, int *L, double *h, 
		  double *g, double *image);
extern void three_D_modwt(double *X, int *NX, int *NY, int *NZ, int *J, int *L, 
		   double *h, double *g, double *LLL, double *HLL, 
		   double *LHL, double *LLH, double *HHL, double *HLH, 
		   double *LHH, double *HHH);
extern void three_D_imodwt(double *LLL, double *HLL, double *LHL, double *LLH, 
		    double *HHL, double *HLH, double *LHH, double *HHH, 
		    int *NX, int *NY, int *NZ, int *J, int *L, double *h, 
		    double *g, double *image);
