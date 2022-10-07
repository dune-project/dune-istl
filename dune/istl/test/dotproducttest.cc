// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#include "config.h"
#include <dune/istl/bvector.hh>
#include <dune/istl/vbvector.hh>
#include <dune/common/fvector.hh>
#include <dune/common/classname.hh>
#include <dune/common/std/type_traits.hh>
#include <dune/common/test/testsuite.hh>
#include <dune/common/scalarvectorview.hh>

template<typename T>
struct Sign
{
  static T complexSign()
    { return T(1.0); }
  static T sqrtComplexSign()
    { return T(1.0); }
};

template<typename T>
struct Sign< std::complex<T> >
{
  static std::complex<T> complexSign()
    { return std::complex<T>(-1.0); }
  static std::complex<T> sqrtComplexSign()
    { return std::complex<T>(0.0, 1.0); }
};

// scalar ordering doesn't work for complex numbers
template <class RealBlockVector, class ComplexBlockVector>
Dune::TestSuite DotProductTest(const size_t numBlocks,const size_t blockSizeOrCapacity) {
  Dune::TestSuite t;

  typedef typename RealBlockVector::field_type rt;
  typedef typename ComplexBlockVector::field_type ct;
  const rt myEps((rt)1e-6);

  static_assert(std::is_same< typename Dune::FieldTraits<rt>::real_type, rt>::value,
                "DotProductTest requires real data type for first block vector!");

  const ct complexSign = Sign<ct>::complexSign();
  const ct I = Sign<ct>::sqrtComplexSign();

  typedef typename RealBlockVector::size_type size_type;

  // empty vectors
  RealBlockVector one(numBlocks,blockSizeOrCapacity);
  ComplexBlockVector iVec(numBlocks,blockSizeOrCapacity);

  using RealBlockType = typename RealBlockVector::block_type;

  const size_type blockSize = Dune::Impl::asVector(one[0]).size();

  t.require(numBlocks==one.N());
  t.require(numBlocks==iVec.N());
  const size_type length = numBlocks * blockSize; // requires inner block size of VariableBlockVector to be 1!

  ct ctlength = ct(length);

  std::cout << __func__ << "\t \t ( " << Dune::className(one) << " and \n \t \t \t   " << Dune::className(iVec) << " )" << std::endl << std::endl;

  // initialize vectors with data
  for(size_type i=0; i < numBlocks; ++i) {
    one[i] = 1.;
    iVec[i] = I;
  }

  ct result = ct();

  // blockwise dot tests
  if constexpr (!Dune::IsNumber<RealBlockType>{}){
    result = ct();
    for(size_type i=0; i < numBlocks; ++i) {
      result += dot(one[i],one[i]) + (one[i]).dot(one[i]);
    }

    t.check(std::abs(result-ct(2)*ctlength)<= myEps);

    result = ct();
    for(size_type i=0; i < numBlocks; ++i) {
      result += dot(iVec[i],iVec[i])+ (iVec[i]).dot(iVec[i]);
    }

    t.check(std::abs(result-ct(2)*ctlength)<= myEps);

    // blockwise dotT / operator * tests
    result = ct();
    for(size_type i=0; i < numBlocks; ++i) {
      result += dotT(one[i],one[i]) + one[i]*one[i];
    }

    t.check(std::abs(result-ct(2)*ctlength)<= myEps);

    result = ct();
    for(size_type i=0; i < numBlocks; ++i) {
      result += dotT(iVec[i],iVec[i]) + iVec[i]*iVec[i];
    }

    t.check(std::abs(result-complexSign*ct(2)*ctlength)<= myEps);
  }

  // global operator * tests
  result = one*one +  dotT(one,one);

  t.check(std::abs(result-ct(2)*ctlength)<= myEps);

  result = iVec*iVec + dotT(iVec,iVec);

  t.check(std::abs(result-complexSign*ct(2)*ctlength)<= myEps);

  // global operator dot(,) tests
  result = one.dot(one)  +  dot(one,one);

  t.check(std::abs(result-ct(2)*ctlength)<= myEps);
  result = iVec.dot(iVec) + dot(iVec,iVec);

  t.check(std::abs(result-ct(2)*ctlength)<= myEps);

  // mixed global dotT tests
  result = iVec*one + one*iVec + dotT(one,iVec) + dotT(iVec,one);

  t.check(std::abs(result-ct(4)*ctlength*I)<= myEps);

  // mixed global dot tests
  result = iVec.dot(one) + dot(iVec,one);

  t.check(std::abs(result-ct(2)*complexSign*ctlength*I)<= myEps);
  result = one.dot(iVec) + dot(one,iVec);

  t.check(std::abs(result-ct(2)*ctlength*I)<= myEps);

  return t;
}


int main()
{
  Dune::TestSuite t;
  const size_t BlockSize = 5;
  const size_t numBlocks = 10;
  const size_t capacity = BlockSize * numBlocks * 2; // use capacity here, that we can use the a constructor taking two integers  for both BlockVector and VariableBlockVector

  t.subTest(DotProductTest<Dune::BlockVector<Dune::FieldVector<int,BlockSize> >, Dune::BlockVector<Dune::FieldVector<int,BlockSize> > >  (numBlocks,capacity));
  t.subTest(DotProductTest<Dune::VariableBlockVector<Dune::FieldVector<int,1> >, Dune::VariableBlockVector<Dune::FieldVector<int,1> > >  (numBlocks,1));

  t.subTest(DotProductTest<Dune::BlockVector<Dune::FieldVector<float,BlockSize> >, Dune::BlockVector<Dune::FieldVector<std::complex<float>,BlockSize> > >  (numBlocks,capacity));
  t.subTest(DotProductTest<Dune::VariableBlockVector<Dune::FieldVector<float,1> >, Dune::VariableBlockVector<Dune::FieldVector<std::complex<float>,1> > >  (numBlocks,BlockSize));

  t.subTest(DotProductTest<Dune::BlockVector<Dune::FieldVector<float,BlockSize> >, Dune::BlockVector<Dune::FieldVector<float,BlockSize> > >  (numBlocks,capacity));
  t.subTest(DotProductTest<Dune::VariableBlockVector<Dune::FieldVector<float,1> >, Dune::VariableBlockVector<Dune::FieldVector<float,1> > >  (numBlocks,1));

  t.subTest(DotProductTest<Dune::BlockVector<Dune::FieldVector<double,BlockSize> >, Dune::BlockVector<Dune::FieldVector<std::complex<double>,BlockSize> > >  (numBlocks,capacity));
  t.subTest(DotProductTest<Dune::VariableBlockVector<Dune::FieldVector<double,1> >, Dune::VariableBlockVector<Dune::FieldVector<std::complex<double>,1> > >  (numBlocks,BlockSize));

  t.subTest(DotProductTest<Dune::BlockVector<Dune::FieldVector<double,BlockSize> >, Dune::BlockVector<Dune::FieldVector<double,BlockSize> > >  (numBlocks,capacity));
  t.subTest(DotProductTest<Dune::VariableBlockVector<Dune::FieldVector<double,1> >, Dune::VariableBlockVector<Dune::FieldVector<double,1> > >  (numBlocks,BlockSize));

  t.subTest(DotProductTest<Dune::BlockVector<double>, Dune::BlockVector<double> >  (numBlocks,capacity));

  return t.exit();
}
