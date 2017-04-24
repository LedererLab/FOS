%module hdim

%{
#include "../FOS/fos.h"
#include "../FOS/x_fos.h"
%}

%include <typemaps.i>
%include <eigen.i>

%eigen_typemaps(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>)
%eigen_typemaps(Eigen::Matrix<double, 1, Eigen::Dynamic>)
%eigen_typemaps(Eigen::Matrix<double, Eigen::Dynamic, 1>)

%include "../FOS/fos.h"
%include "../FOS/x_fos.h"

template < typename T >
class FOS {

public:
	FOS( Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > x, Eigen::Matrix< T, Eigen::Dynamic, 1 > y );
	void Algorithm();

	T ReturnLambda();
	Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > ReturnBetas();
	uint ReturnOptimIndex();
	Eigen::Matrix< T, Eigen::Dynamic, 1 > ReturnCoefficients();
	Eigen::Matrix< T, Eigen::Dynamic, 1 > ReturnSupport();

};

template < typename T >
class X_FOS {

public:

	X_FOS();
	void Process( const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >&x,
		     const Eigen::Matrix< T, Eigen::Dynamic, 1 >&y );

	T ReturnLambda();
	Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > ReturnBetas();
	uint ReturnOptimIndex();
	Eigen::Matrix< T, Eigen::Dynamic, 1 > ReturnCoefficients();
	Eigen::Matrix< int, Eigen::Dynamic, 1 > ReturnSupport();

};

%template(FOS_d) hdim::FOS<double>;
%template(X_FOS_d) hdim::experimental::X_FOS<double>;
