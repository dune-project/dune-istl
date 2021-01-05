// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_BLOCKKRYLOV_CHOLESKY_HH
#define DUNE_ISTL_BLOCKKRYLOV_CHOLESKY_HH

#include <vector>
#include <cmath>

#include <dune/common/fmatrix.hh>


namespace Dune {

  template<class T, int N>
  std::vector<size_t> cholesky_factorize(FieldMatrix<T, N>& mat){
    using std::sqrt, std::real;
    std::vector<size_t> dependend_columns = {};
    for(size_t i=0; i<N;++i){
      for(size_t j=0; j<=i; ++j){
        T sum = mat[i][j];
        for(size_t k=0;k<j;++k)
          sum -= mat[i][k]*conjugateComplex(mat[j][k]);
        if(i > j){
          mat[i][j] = sum/mat[j][j];
        }else{ // i == j
          if(real(sum) > 100.0*real(mat[i][j]*std::numeric_limits<T>::epsilon())){
            mat[i][i] = sqrt(sum);
          }else{
            dependend_columns.push_back(i);
            mat[i][i] = 0.0;
          }
        }
      }
      for(size_t j=i+1;j<N;++j){
        mat[i][j] = 0.0;
      }
    }
    return dependend_columns;
  }
}

#endif
