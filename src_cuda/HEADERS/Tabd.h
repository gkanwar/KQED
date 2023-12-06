#ifndef TABD_H
#define TABD_H

typedef enum {
  KQED_TABD_XEQ0, KQED_TABD_YEQ0
} KQEDTabdKind;

__device__ // KQED_PRIVATE
int
Tabd_xeq0( const double yv[4] ,
	   const struct Grid_coeffs Grid ,
	   double tI[4][4][4] ,
	   double tII[4][4][4] ,
	   double tIII[4][4][4] ) ;

__device__ // KQED_PRIVATE
int
Tabd_yeq0( const double xv[4] ,
	   const struct Grid_coeffs Grid ,
	   double tI[4][4][4] ,
	   double tII[4][4][4] ,
	   double tIII[4][4][4] ) ;

#endif
