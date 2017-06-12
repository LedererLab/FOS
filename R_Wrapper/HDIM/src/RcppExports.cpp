// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <Rcpp.h>

using namespace Rcpp;

// FOS
Rcpp::List FOS(const Rcpp::NumericMatrix& X, const Rcpp::NumericVector& Y, const std::string solver_type);
RcppExport SEXP HDIM_FOS(SEXP XSEXP, SEXP YSEXP, SEXP solver_typeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Rcpp::NumericMatrix& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const std::string >::type solver_type(solver_typeSEXP);
    rcpp_result_gen = Rcpp::wrap(FOS(X, Y, solver_type));
    return rcpp_result_gen;
END_RCPP
}
// CoordinateDescent
Rcpp::NumericVector CoordinateDescent(Rcpp::NumericMatrix X, Rcpp::NumericVector Y, Rcpp::NumericVector Beta_0, double lambda, unsigned int num_iterations);
RcppExport SEXP HDIM_CoordinateDescent(SEXP XSEXP, SEXP YSEXP, SEXP Beta_0SEXP, SEXP lambdaSEXP, SEXP num_iterationsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type X(XSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type Y(YSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type Beta_0(Beta_0SEXP);
    Rcpp::traits::input_parameter< double >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type num_iterations(num_iterationsSEXP);
    rcpp_result_gen = Rcpp::wrap(CoordinateDescent(X, Y, Beta_0, lambda, num_iterations));
    return rcpp_result_gen;
END_RCPP
}
// CoordinateDescent_DG
Rcpp::NumericVector CoordinateDescent_DG(Rcpp::NumericMatrix X, Rcpp::NumericVector Y, Rcpp::NumericVector Beta_0, double lambda, double duality_gap_target);
RcppExport SEXP HDIM_CoordinateDescent_DG(SEXP XSEXP, SEXP YSEXP, SEXP Beta_0SEXP, SEXP lambdaSEXP, SEXP duality_gap_targetSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type X(XSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type Y(YSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type Beta_0(Beta_0SEXP);
    Rcpp::traits::input_parameter< double >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< double >::type duality_gap_target(duality_gap_targetSEXP);
    rcpp_result_gen = Rcpp::wrap(CoordinateDescent_DG(X, Y, Beta_0, lambda, duality_gap_target));
    return rcpp_result_gen;
END_RCPP
}
// ISTA
Rcpp::NumericVector ISTA(Rcpp::NumericMatrix X, Rcpp::NumericVector Y, Rcpp::NumericVector Beta_0, double lambda, unsigned int num_iterations, double L_0);
RcppExport SEXP HDIM_ISTA(SEXP XSEXP, SEXP YSEXP, SEXP Beta_0SEXP, SEXP lambdaSEXP, SEXP num_iterationsSEXP, SEXP L_0SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type X(XSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type Y(YSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type Beta_0(Beta_0SEXP);
    Rcpp::traits::input_parameter< double >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type num_iterations(num_iterationsSEXP);
    Rcpp::traits::input_parameter< double >::type L_0(L_0SEXP);
    rcpp_result_gen = Rcpp::wrap(ISTA(X, Y, Beta_0, lambda, num_iterations, L_0));
    return rcpp_result_gen;
END_RCPP
}
// ISTA_DG
Rcpp::NumericVector ISTA_DG(Rcpp::NumericMatrix X, Rcpp::NumericVector Y, Rcpp::NumericVector Beta_0, double lambda, double duality_gap_target, double L_0);
RcppExport SEXP HDIM_ISTA_DG(SEXP XSEXP, SEXP YSEXP, SEXP Beta_0SEXP, SEXP lambdaSEXP, SEXP duality_gap_targetSEXP, SEXP L_0SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type X(XSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type Y(YSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type Beta_0(Beta_0SEXP);
    Rcpp::traits::input_parameter< double >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< double >::type duality_gap_target(duality_gap_targetSEXP);
    Rcpp::traits::input_parameter< double >::type L_0(L_0SEXP);
    rcpp_result_gen = Rcpp::wrap(ISTA_DG(X, Y, Beta_0, lambda, duality_gap_target, L_0));
    return rcpp_result_gen;
END_RCPP
}
// FISTA
Rcpp::NumericVector FISTA(Rcpp::NumericMatrix X, Rcpp::NumericVector Y, Rcpp::NumericVector Beta_0, double lambda, unsigned int num_iterations, double L_0);
RcppExport SEXP HDIM_FISTA(SEXP XSEXP, SEXP YSEXP, SEXP Beta_0SEXP, SEXP lambdaSEXP, SEXP num_iterationsSEXP, SEXP L_0SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type X(XSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type Y(YSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type Beta_0(Beta_0SEXP);
    Rcpp::traits::input_parameter< double >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type num_iterations(num_iterationsSEXP);
    Rcpp::traits::input_parameter< double >::type L_0(L_0SEXP);
    rcpp_result_gen = Rcpp::wrap(FISTA(X, Y, Beta_0, lambda, num_iterations, L_0));
    return rcpp_result_gen;
END_RCPP
}
// FISTA_DG
Rcpp::NumericVector FISTA_DG(Rcpp::NumericMatrix X, Rcpp::NumericVector Y, Rcpp::NumericVector Beta_0, double lambda, double duality_gap_target, double L_0);
RcppExport SEXP HDIM_FISTA_DG(SEXP XSEXP, SEXP YSEXP, SEXP Beta_0SEXP, SEXP lambdaSEXP, SEXP duality_gap_targetSEXP, SEXP L_0SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type X(XSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type Y(YSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type Beta_0(Beta_0SEXP);
    Rcpp::traits::input_parameter< double >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< double >::type duality_gap_target(duality_gap_targetSEXP);
    Rcpp::traits::input_parameter< double >::type L_0(L_0SEXP);
    rcpp_result_gen = Rcpp::wrap(FISTA_DG(X, Y, Beta_0, lambda, duality_gap_target, L_0));
    return rcpp_result_gen;
END_RCPP
}
