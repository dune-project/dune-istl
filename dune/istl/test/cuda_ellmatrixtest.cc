// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#include "config.h"
#include <iostream>
#include <cstdlib>
#include <dune/istl/vector/cuda.hh>
#include <dune/istl/ellmatrix/cuda.hh>

using namespace Dune;


template <typename DT_, typename A_>
int ell_test()
{
  int result(EXIT_SUCCESS);
  size_t size(5);
  size_t rows(size);
  size_t cols(size);
  size_t nonzeros(5);

  size_t * row = new size_t[nonzeros];
  size_t * col = new size_t[nonzeros];
  DT_ * val = new DT_[nonzeros];

  row[0] = 0;
  col[0] = 0;
  val[0] = 1;
  row[1] = 1;
  col[1] = 1;
  val[1] = 2;
  row[2] = 2;
  col[2] = 2;
  val[2] = 3;
  row[3] = 3;
  col[3] = 3;
  val[3] = 4;
  row[4] = 4;
  col[4] = 4;
  val[4] = 5;
  /*row[5] = 0;
  col[5] = 1;
  val[5] = 6;
  row[6] = 1;
  col[6] = 0;
  val[6] = 7;
  row[7] = 3;
  col[7] = 4;
  val[7] = 8;*/

  ISTL::ELLMatrix<DT_, A_> m1(val, row, col, nonzeros, rows, cols, 2, 1);
  delete[] row;
  delete[] col;
  delete[] val;
  m1.print();
  if (m1.size() != rows*cols)
    return EXIT_FAILURE;


  ISTL::Vector<DT_, A_> x(size, DT_(1));
  ISTL::Vector<DT_, A_> y(size, DT_(0));

  m1.mv(y, x);

  for (size_t i(0) ; i < size ; ++i)
    if (y[i] != i+1)
      return EXIT_FAILURE;

  m1.mmv(y, x);

  for (size_t i(0) ; i < size ; ++i)
    if (y[i] != 0)
      return EXIT_FAILURE;

  for (size_t i(0) ; i < size ; ++i)
    y(i, DT_(1));

  m1.umv(y, x);
  for (size_t i(0) ; i < size ; ++i)
    if (y[i] != i+2)
      return EXIT_FAILURE;

  for (size_t i(0) ; i < size ; ++i)
    y(i, DT_(0));

  m1.usmv(2, y, x);
  for (size_t i(0) ; i < size ; ++i)
    if (y[i] != 2*(i+1))
      return EXIT_FAILURE;

  return result;
}


int main()
{
  if (ell_test<float, Memory::CudaAllocator<float> >() == EXIT_FAILURE)
  {
    std::cout<<"ell test failed!"<<std::endl;
    return EXIT_FAILURE;
  }
  /*if (ell_test<double, Memory::CudaAllocator<double> >() == EXIT_FAILURE)
  {
    std::cout<<"ell test failed!"<<std::endl;
    return EXIT_FAILURE;
  }*/

  return EXIT_SUCCESS;
}
