template< typename T, uint n, uint p >
FOS< T , n, p >::FOS( Eigen::Matrix< T, n, p >& X, Eigen::Matrix< T, n, 1 >& Y ){

}

template< typename T, uint n, uint p >
T FOS< T , n, p >::StdDev( Eigen::Matrix< T, n, p >& mat ) {

    Eigen::RowVectorXf mean = mat.colwise().mean();
    return ((mat.rowwise() - mean).array().square().colwise().sum() / (mat.rows() - 1)).sqrt();
}


template< typename T, uint n, uint p >
void FOS< T , n, p >::Normalize( Eigen::Matrix< T, n, p >& mat ) {

    auto mean = mat.colwise().mean();
    auto std = ((mat.rowwise() - mean).array().square().colwise().sum() / (mat.rows() - 1)).sqrt();

    mat = (mat.rowwise() - mean).array().rowwise() / std.array();
}
