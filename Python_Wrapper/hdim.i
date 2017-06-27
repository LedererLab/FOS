%module hdim

%{
#include "../FOS/fos.hpp"
#include "../FOS/x_fos.hpp"
#include "../Solvers/solver.hpp"
#include "../Solvers/SubGradientDescent/subgradient_descent.hpp"
#include "../Solvers/SubGradientDescent/ISTA/ista.hpp"
#include "../Solvers/SubGradientDescent/FISTA/fista.hpp"
#include "../Solvers/CoordinateDescent/coordinate_descent.hpp"
%}

%include <typemaps.i>
%include <eigen.i>

%eigen_typemaps(Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>)
%eigen_typemaps(Eigen::Matrix<double,1,Eigen::Dynamic>)
%eigen_typemaps(Eigen::Matrix<double, Eigen::Dynamic,1>)

%eigen_typemaps(Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic>)
%eigen_typemaps(Eigen::Matrix<float,1,Eigen::Dynamic>)
%eigen_typemaps(Eigen::Matrix<float,Eigen::Dynamic,1>)

%eigen_typemaps(Eigen::Matrix<int,Eigen::Dynamic,1>)

%include "../FOS/fos.hpp"
%include "../FOS/x_fos.hpp"
%include "../Solvers/solver.hpp"
%include "../Solvers/SubGradientDescent/subgradient_descent.hpp"
%include "../Solvers/SubGradientDescent/ISTA/ista.hpp"
%include "../Solvers/SubGradientDescent/FISTA/fista.hpp"
%include "../Solvers/CoordinateDescent/coordinate_descent.hpp"

template < typename T >
class Solver {

  public:
    virtual ~Solver() {}

    virtual Eigen::Matrix< T, Eigen::Dynamic, 1 > operator()(
        const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0,
        T lambda,
        unsigned int num_iterations ) = 0;

    virtual Eigen::Matrix< T, Eigen::Dynamic, 1 > operator()(
        const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0,
        T lambda,
        T duality_gap_target ) = 0;
};

template < typename T >
class SubGradientSolver : public hdim::internal::Solver<T> {

  public:
    SubGradientSolver( T L = 0.1 );
    ~SubGradientSolver();
};

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
	~X_FOS();

	void operator()( const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& x,
									 const Eigen::Matrix< T, Eigen::Dynamic, 1 >& y,
									 SolverType s_type = SolverType::ista );

	T ReturnLambda();
	T ReturnIntercept();
	Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > ReturnBetas();
	unsigned int ReturnOptimIndex();
	Eigen::Matrix< T, Eigen::Dynamic, 1 > ReturnCoefficients();
	Eigen::Matrix< int, Eigen::Dynamic, 1 > ReturnSupport();

};

template < typename T >
class ISTA : public hdim::internal::SubGradientSolver<T> {

	public:
		Eigen::Matrix< T, Eigen::Dynamic, 1 >  operator()(
			const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > & X,
			const Eigen::Matrix< T, Eigen::Dynamic, 1 > & Y,
			const Eigen::Matrix< T, Eigen::Dynamic, 1 > & Beta_0,
			T lambda,
			unsigned int num_iterations );

		Eigen::Matrix< T, Eigen::Dynamic, 1 >  operator()(
			const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > & X,
			const Eigen::Matrix< T, Eigen::Dynamic, 1 > & Y,
			const Eigen::Matrix< T, Eigen::Dynamic, 1 > & Beta_0,
			T lambda,
			T duality_gap_target );
};

template < typename T >
class FISTA : public hdim::internal::SubGradientSolver<T> {

	public:
		Eigen::Matrix< T, Eigen::Dynamic, 1 >  operator()(
			const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > & X,
			const Eigen::Matrix< T, Eigen::Dynamic, 1 > & Y,
			const Eigen::Matrix< T, Eigen::Dynamic, 1 > & Beta_0,
			T lambda,
			unsigned int num_iterations );

		Eigen::Matrix< T, Eigen::Dynamic, 1 >  operator()(
			const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > & X,
			const Eigen::Matrix< T, Eigen::Dynamic, 1 > & Y,
			const Eigen::Matrix< T, Eigen::Dynamic, 1 > & Beta_0,
			T lambda,
			T duality_gap_target );
};

template < typename T >
class CoordinateDescentSolver : public hdim::internal::Solver<T> {

  public:
    CoordinateDescentSolver(const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
                            const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
                            const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0 );
    ~CoordinateDescentSolver();

		Eigen::Matrix< T, Eigen::Dynamic, 1 >  operator()(
			const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > & X,
			const Eigen::Matrix< T, Eigen::Dynamic, 1 > & Y,
			const Eigen::Matrix< T, Eigen::Dynamic, 1 > & Beta_0,
			T lambda,
			unsigned int num_iterations );

		Eigen::Matrix< T, Eigen::Dynamic, 1 >  operator()(
			const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > & X,
			const Eigen::Matrix< T, Eigen::Dynamic, 1 > & Y,
			const Eigen::Matrix< T, Eigen::Dynamic, 1 > & Beta_0,
			T lambda,
			T duality_gap_target );

};

template < typename T >
class LazyCoordinateDescent : public internal::Solver<T> {

  public:
    LazyCoordinateDescent( const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
                          const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
                          const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0 );
    ~LazyCoordinateDescent();

    Eigen::Matrix< T, Eigen::Dynamic, 1 > operator()(
        const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0,
        T lambda,
        T duality_gap_target );

    Eigen::Matrix< T, Eigen::Dynamic, 1 > operator()(
        const Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& X,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Y,
        const Eigen::Matrix< T, Eigen::Dynamic, 1 >& Beta_0,
        T lambda,
        unsigned int num_iterations );

};

%template(FOS_d) hdim::FOS<double>;
%template(FOS_f) hdim::FOS<float>;

%template(X_FOS_d) hdim::experimental::X_FOS<double>;
%template(X_FOS_f) hdim::experimental::X_FOS<float>;

%template(Solver_f) hdim::internal::Solver<float>;
%template(Solver_d) hdim::internal::Solver<double>;

%template(SGD_f) hdim::internal::SubGradientSolver<float>;
%template(SGD_d) hdim::internal::SubGradientSolver<double>;

%template(ISTA_f) hdim::ISTA<float>;
%template(ISTA_d) hdim::ISTA<double>;

%template(FISTA_f) hdim::FISTA<float>;
%template(FISTA_d) hdim::FISTA<double>;

%template(CD_f) hdim::CoordinateDescentSolver<float>;
%template(CD_d) hdim::CoordinateDescentSolver<double>;

%template(Lazy_CD_f) hdim::LazyCoordinateDescent<float>;
%template(Lazy_CD_d) hdim::LazyCoordinateDescent<double>;
