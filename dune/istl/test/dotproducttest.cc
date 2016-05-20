// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#include "config.h"
#include <dune/istl/bvector.hh>
#include <dune/istl/vbvector.hh>
#include <dune/common/fvector.hh>
#include <dune/common/classname.hh>

// scalar ordering doesn't work for complex numbers
template <class RealBlockVector, class ComplexBlockVector>
int  DotProductTest(const size_t numBlocks,const size_t blockSizeOrCapacity) {
  typedef typename RealBlockVector::field_type rt;
  typedef typename ComplexBlockVector::field_type ct;
  const rt myEps((rt)1e-6);

  static_assert(std::is_same< typename Dune::FieldTraits<rt>::real_type, rt>::value,
                "DotProductTest requires real data type for first block vector!");

  const bool secondBlockIsComplex = !std::is_same< typename Dune::FieldTraits<ct>::real_type, ct>::value;

  const ct complexSign = secondBlockIsComplex ? -1. : 1.;
  // avoid constructor ct(0.,1.)
  const ct I = secondBlockIsComplex ? std::sqrt(ct(-1.)) : ct(1.); // imaginary unit

  typedef typename RealBlockVector::size_type size_type;

  // empty vectors
  RealBlockVector one(numBlocks,blockSizeOrCapacity);
  ComplexBlockVector iVec(numBlocks,blockSizeOrCapacity);

  const size_type blockSize = one[0].size();

  assert(numBlocks==one.N());
  assert(numBlocks==iVec.N());
  const size_type length = numBlocks * blockSize; // requires innter block size of VariableBlockVector to be 1!

  ct ctlength = ct(length);

  std::cout << __func__ << "\t \t ( " << Dune::className(one) << " and \n \t \t \t   " << Dune::className(iVec) << " )" << std::endl << std::endl;

  // initialize vectors with data
  for(size_type i=0; i < numBlocks; ++i) {
    for(size_type j=0; j < blockSize; ++j) {
      one[i][j] = 1.;
      iVec[i][j] = I;
    }
  }

  ct result = ct();

  // blockwise dot tests
  result = ct();
  for(size_type i=0; i < numBlocks; ++i) {
    result += dot(one[i],one[i]) + one[i].dot(one[i]);
  }

  assert(std::abs(result-ct(2)*ctlength)<= myEps);

  result = ct();
  for(size_type i=0; i < numBlocks; ++i) {
    result += dot(iVec[i],iVec[i])+ (iVec[i]).dot(iVec[i]);
  }

  assert(std::abs(result-ct(2)*ctlength)<= myEps);

  // blockwise dotT / operator * tests
  result = ct();
  for(size_type i=0; i < numBlocks; ++i) {
    result += dotT(one[i],one[i]) + one[i]*one[i];
  }

  assert(std::abs(result-ct(2)*ctlength)<= myEps);

  result = ct();
  for(size_type i=0; i < numBlocks; ++i) {
    result += dotT(iVec[i],iVec[i]) + iVec[i]*iVec[i];
  }

  assert(std::abs(result-complexSign*ct(2)*ctlength)<= myEps);

  // global operator * tests
  result = one*one +  dotT(one,one);

  assert(std::abs(result-ct(2)*ctlength)<= myEps);

  result = iVec*iVec + dotT(iVec,iVec);

  assert(std::abs(result-complexSign*ct(2)*ctlength)<= myEps);

  // global operator dot(,) tests
  result = one.dot(one)  +  dot(one,one);

  assert(std::abs(result-ct(2)*ctlength)<= myEps);
  result = iVec.dot(iVec) + dot(iVec,iVec);

  assert(std::abs(result-ct(2)*ctlength)<= myEps);

  // mixed global dotT tests
  result = iVec*one + one*iVec + dotT(one,iVec) + dotT(iVec,one);

  assert(std::abs(result-ct(4)*ctlength*I)<= myEps);

  // mixed global dot tests
  result = iVec.dot(one) + dot(iVec,one);

  assert(std::abs(result-ct(2)*complexSign*ctlength*I)<= myEps);
  result = one.dot(iVec) + dot(one,iVec);

  assert(std::abs(result-ct(2)*ctlength*I)<= myEps);

  return 0;
}


int main()
{
  int ret = 0;
  const size_t BlockSize = 5;
  const size_t numBlocks = 10;
  const size_t capacity = BlockSize * numBlocks * 2; // use capacity here, that we can use the a constructor taking two integers  for both BlockVector and VariableBlockVector

  ret += DotProductTest<Dune::BlockVector<Dune::FieldVector<int,BlockSize> >, Dune::BlockVector<Dune::FieldVector<int,BlockSize> > >  (numBlocks,capacity);
  ret += DotProductTest<Dune::VariableBlockVector<Dune::FieldVector<int,1> >, Dune::VariableBlockVector<Dune::FieldVector<int,1> > >  (numBlocks,1);

  ret += DotProductTest<Dune::BlockVector<Dune::FieldVector<float,BlockSize> >, Dune::BlockVector<Dune::FieldVector<std::complex<float>,BlockSize> > >  (numBlocks,capacity);
  ret += DotProductTest<Dune::VariableBlockVector<Dune::FieldVector<float,1> >, Dune::VariableBlockVector<Dune::FieldVector<std::complex<float>,1> > >  (numBlocks,BlockSize);

  ret += DotProductTest<Dune::BlockVector<Dune::FieldVector<float,BlockSize> >, Dune::BlockVector<Dune::FieldVector<float,BlockSize> > >  (numBlocks,capacity);
  ret += DotProductTest<Dune::VariableBlockVector<Dune::FieldVector<float,1> >, Dune::VariableBlockVector<Dune::FieldVector<float,1> > >  (numBlocks,1);

  ret += DotProductTest<Dune::BlockVector<Dune::FieldVector<double,BlockSize> >, Dune::BlockVector<Dune::FieldVector<std::complex<double>,BlockSize> > >  (numBlocks,capacity);
  ret += DotProductTest<Dune::VariableBlockVector<Dune::FieldVector<double,1> >, Dune::VariableBlockVector<Dune::FieldVector<std::complex<double>,1> > >  (numBlocks,BlockSize);

  ret += DotProductTest<Dune::BlockVector<Dune::FieldVector<double,BlockSize> >, Dune::BlockVector<Dune::FieldVector<double,BlockSize> > >  (numBlocks,capacity);
  ret += DotProductTest<Dune::VariableBlockVector<Dune::FieldVector<double,1> >, Dune::VariableBlockVector<Dune::FieldVector<double,1> > >  (numBlocks,BlockSize);

  return ret;
}
