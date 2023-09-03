# ax_cuda.m4: An m4 macro to detect and configure Cuda
#
# Copyright © 2008 Frederic Chateau <frederic.chateau@cea.fr>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
#
# As a special exception to the GNU General Public License, if you
# distribute this file as part of a program that contains a
# configuration script generated by Autoconf, you may include it under
# the same distribution terms that you use for the rest of that program.
#


#
# SYNOPSIS
#	AX_CUDA()
#
# DESCRIPTION
#	Checks the existence of Cuda binaries and libraries.
#	Options:
#	--with-cuda=(path|yes|no)
#		Indicates whether to use Cuda or not, and the path of a non-standard
#		installation location of Cuda if necessary.
#
#	This macro calls:
#		AC_SUBST(CUDA_CFLAGS)
#		AC_SUBST(CUDA_LIBS)
#		AC_SUBST(NVCC)
#		AC_SUBST(NVCCFLAGS)
#
AC_DEFUN([AX_CUDA],
[
AC_ARG_WITH([cuda],
    AS_HELP_STRING([--with-cuda@<:@=yes|no|DIR@:>@], [prefix where cuda is installed (default=yes)]),
[
	with_cuda=$withval
	if test "$withval" = "no"
	then
		want_cuda="no"
	elif test "$withval" = "yes"
	then
		want_cuda="yes"
	else
		want_cuda="yes"
		cuda_home_path=$withval
	fi
],
[
	want_cuda="yes"
])

AM_CONDITIONAL(USE_CUDA, test "x${want_cuda}" = xyes)

if test "$want_cuda" = "yes"
then
	# check that nvcc compiler is in the path
	if test -n "$cuda_home_path"
	then
	    nvcc_search_dirs="$PATH$PATH_SEPARATOR$cuda_home_path/bin"
	else
	    nvcc_search_dirs=$PATH
	fi

	AC_PATH_PROG([NVCC], [nvcc], [], [$nvcc_search_dirs])
	if test -n "$NVCC"
	then
		have_nvcc="yes"
	else
		have_nvcc="no"
	fi

	# test if nvcc version is >= 2.3
	NVCC_VERSION=`$NVCC --version | grep release | awk 'gsub(/,/, "") {print [$]5}'`
	AC_MSG_RESULT([nvcc version : $NVCC_VERSION])
	
	# test if architecture is 64 bits and NVCC version >= 2.3
        libdir=lib
	if test "x$host_cpu" = xx86_64 ; then
	   if test "x$NVCC_VERSION" \> "x2.2" ; then
              libdir=lib64
           fi
	fi

	# set CUDA flags
	if test -n "$cuda_home_path"
	then
	    CUDA_CFLAGS="-I$cuda_home_path/include"
	    CUDA_LIBS="-L$cuda_home_path/$libdir -lcudart"
	else
	    CUDA_CFLAGS="-I/usr/local/cuda/include"
	    CUDA_LIBS="-L/usr/local/cuda/$libdir -lcudart"
	fi

	# Env var CUDA_DRIVER_LIB_PATH can be used to set an alternate driver library path
	# this is usefull when building on a host where only toolkit (nvcc) is installed
	# and not driver. Driver libs must be placed in some location specified by this var.
	if test -n "$CUDA_DRIVER_LIB_PATH"
	then
	    CUDA_LIBS+=" -L$CUDA_DRIVER_LIB_PATH -lcuda"
	else
	    CUDA_LIBS+=" -lcuda"
	fi

	saved_CPPFLAGS=$CPPFLAGS
	saved_LIBS=$LIBS

	CPPFLAGS="$CPPFLAGS $CUDA_CFLAGS"
	LIBS="$LIBS $CUDA_LIBS"

	AC_LANG_PUSH(C)
	AC_MSG_CHECKING([for Cuda headers])
	AC_COMPILE_IFELSE(
	[
		AC_LANG_PROGRAM([@%:@include <cuda.h>], [])
	],
	[
		have_cuda_headers="yes"
		AC_MSG_RESULT([yes])
	],
	[
		have_cuda_headers="no"
		AC_MSG_RESULT([not found])
	])

	AC_MSG_CHECKING([for Cuda libraries])
	AC_LINK_IFELSE(
	[
		AC_LANG_PROGRAM([@%:@include <cuda.h>],
		[
			CUmodule cuModule;
			cuModuleLoad(&cuModule, "myModule.cubin");
			CUdeviceptr devPtr;
			CUfunction cuFunction;
			unsigned pitch, width = 250, height = 500;
			cuMemAllocPitch(&devPtr, &pitch,width * sizeof(float), height, 4);
			cuModuleGetFunction(&cuFunction, cuModule, "myKernel");
			cuFuncSetBlockShape(cuFunction, 512, 1, 1);
			cuParamSeti(cuFunction, 0, devPtr);
			cuParamSetSize(cuFunction, sizeof(devPtr));
			cuLaunchGrid(cuFunction, 100, 1);
		])
	],
	[
		have_cuda_libs="yes"
		AC_MSG_RESULT([yes])
	],
	[
		have_cuda_libs="no"
		AC_MSG_RESULT([not found])
	])
	AC_LANG_POP(C)

	CPPFLAGS=$saved_CPPFLAGS
	LIBS=$saved_LIBS
	
	if test "$have_cuda_headers" = "yes" -a "$have_cuda_libs" = "yes" -a "$have_nvcc" = "yes"
	then
		have_cuda="yes"
	else
		have_cuda="no"
		AC_MSG_ERROR([Cuda is requested but not available])
	fi
fi

AC_SUBST(CUDA_CFLAGS)
AC_SUBST(CUDA_LIBS)
AC_SUBST(NVCC)

# Fast math
AC_ARG_WITH([cuda-fast-math],
	[AS_HELP_STRING([--with-cuda-fast-math],
		[Tell nvcc to use -use_fast_math flag])],
	[
		if test "$withval" = "no"
		then
			want_fast_math="no"
		elif test "$withval" = "yes"
		then
			want_fast_math="yes"
		else
			with_fast_math="$withval"
			want_fast_math="yes"
		fi
	 ],
         [
		want_fast_math="yes"
	 ]
)

# Target GPU arch
target_arch=""
AC_ARG_WITH([target-gpu-arch],
	AS_HELP_STRING(
	[--with-target-gpu-arch@<:@=60,61,70,75,80,86@:>@],
	[Primary architecture to compile for (default=86)]),
[
	if test "$withval" = "86" ; then target_arch=86 
	elif  test "$withval" = "80" ; then target_arch=80
	elif  test "$withval" = "75" ; then target_arch=75
	elif  test "$withval" = "70" ; then target_arch=70
	elif  test "$withval" = "61" ; then target_arch=61
	elif  test "$withval" = "60" ; then target_arch=60
	else
		AC_MSG_ERROR([Requested target-gpu-arch must be in 60,61,70,75,80,86, not $withval])
	fi
	
], [ target_arch="86"] )
AC_MSG_NOTICE([target gpu architecture is sm$target_arch])

# Default nvcc flags
NVCCFLAGS=" -ccbin $CXX"
NVCCFLAGS+=" --gpu-architecture=sm_$target_arch"
NVCCFLAGS+=" -gencode=arch=compute_$target_arch,code=compute_$target_arch"

if test x$want_fast_math = xyes
then
	NVCCFLAGS+=" -use_fast_math"
fi

AC_MSG_NOTICE([Using NVCCFLAGS=$NVCCFLAGS])
AC_SUBST(NVCCFLAGS)
])
