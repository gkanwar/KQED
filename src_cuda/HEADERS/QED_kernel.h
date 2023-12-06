#ifndef QED_KERNEL_H
#define QED_KERNEL_H

__device__
static inline size_t
ident(size_t idx)
{
  return idx;
}

__device__
static inline size_t
swap_mulam( const size_t idx )
{
  const size_t l[4] = { idx/64 , (idx/16)&3 , (idx/4)&3 , idx&3 } ;
  return l[1] + 4*(l[2]+4*(l[3]+4*l[0]) ) ;
}

__device__
int
kernelQED( const double xv[4] ,
	   const double yv[4] ,
	   struct QED_kernel_temps t ,
	   double kerv[6][4][4][4] ) ;

__device__
int
kernelQED_axpy( const double xv[4] ,
                const double yv[4] ,
                struct QED_kernel_temps t ,
                double S ,
                double kerv[6][4][4][4] ) ;

#endif
