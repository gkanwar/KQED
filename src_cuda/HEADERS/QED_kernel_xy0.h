#ifndef QED_KERNEL_XY0_H
#define QED_KERNEL_XY0_H

#include "Tabd.h"

__device__
int
kernelQED_xoryeq0( const double qv[4] ,
		   const struct QED_kernel_temps t ,
		   double kerv[6][4][4][4] ,
                   KQEDTabdKind kind );
		   // int (*f)( const double qv[4] ,
		   //           const struct Grid_coeffs Grid ,
		   //           double vv[4][4][4] ,
		   //           double txv[4][4][4] ,
		   //           double tyv[4][4][4] ) ) ;

__device__
int
kernelQED_xoryeq0_axpy( const double qv[4] ,
                        const struct QED_kernel_temps t ,
                        double kerv[6][4][4][4] ,
                        double S , bool mulam_minus ,
                        KQEDTabdKind kind );
                            

#endif
