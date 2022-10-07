// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#include "config.h"

#undef NDEBUG // make sure assert works

#include <dune/common/float_cmp.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/exceptions.hh>
#include <dune/istl/bcrsmatrix.hh>

typedef Dune::BCRSMatrix<Dune::FieldMatrix<double,1,1> > ScalarMatrix;

void buildMatrix(ScalarMatrix& m)
{
  m.entry(0,0) = 1.0; m.entry(0,1) = 1.0; m.entry(0,2) = 1.0;  m.entry(0,3) = 1.0;
  m.entry(1,0) = 1.0; m.entry(1,1) = 1.0; m.entry(1,2) = 1.0;
  m.entry(2,1) = 1.0; m.entry(2,2) = 1.0; m.entry(2,3) = 1.0;
  m.entry(3,2) = 1.0; m.entry(3,3) = 1.0; m.entry(3,4) = 1.0;
  m.entry(4,3) = 1.0; m.entry(4,4) = 1.0; m.entry(4,5) = 1.0;
  m.entry(5,4) = 1.0; m.entry(5,5) = 1.0; m.entry(5,6) = 1.0;
  m.entry(6,5) = 1.0; m.entry(6,6) = 1.0; m.entry(6,7) = 1.0;
  m.entry(7,6) = 1.0; m.entry(7,7) = 1.0; m.entry(7,8) = 1.0;
  m.entry(8,7) = 1.0; m.entry(8,8) = 1.0; m.entry(8,9) = 1.0;
  m.entry(9,8) = 1.0; m.entry(9,9) = 1.0;
  // add some more entries in random order
  m.entry(7,3) = 1.0;
  m.entry(6,0) = 1.0;
  m.entry(3,8) = 1.0;
}

template<typename M>
void setMatrix(M& m)
{
  m[0][0] = 1.0; m[0][1] = 1.0; m[0][2] = 1.0;  m[0][3] = 1.0;
  m[1][0] = 1.0; m[1][1] = 1.0; m[1][2] = 1.0;
  m[2][1] = 1.0; m[2][2] = 1.0; m[2][3] = 1.0;
  m[3][2] = 1.0; m[3][3] = 1.0; m[3][4] = 1.0;
  m[4][3] = 1.0; m[4][4] = 1.0; m[4][5] = 1.0;
  m[5][4] = 1.0; m[5][5] = 1.0; m[5][6] = 1.0;
  m[6][5] = 1.0; m[6][6] = 1.0; m[6][7] = 1.0;
  m[7][6] = 1.0; m[7][7] = 1.0; m[7][8] = 1.0;
  m[8][7] = 1.0; m[8][8] = 1.0; m[8][9] = 1.0;
  m[9][8] = 1.0; m[9][9] = 1.0;
  // add some more entries in random order
  m[7][3] = 1.0;
  m[6][0] = 1.0;
  m[3][8] = 1.0;
}

void testImplicitBuild()
{
  ScalarMatrix m(10,10,3,0.1,ScalarMatrix::implicit);
  buildMatrix(m);
  ScalarMatrix::CompressionStatistics stats = m.compress();
  assert(Dune::FloatCmp::eq(stats.avg,33./10.));
  assert(stats.maximum == 4);
  assert(stats.overflow_total == 4);
  setMatrix(m);
  ScalarMatrix m1(m);
}

void testImplicitBuildWithInsufficientOverflow()
{
  try {
    ScalarMatrix m(10,10,1,0,ScalarMatrix::implicit);
    // add diagonal entries + completely fill the first row with entries
    // with the current base buffer of 4 * avg, that should be enough to make
    // compress fail.
    for (int i = 0; i < 10; ++i)
      {
        m.entry(i,i) = 1.0;
        m.entry(0,i) = 1.0;
      }
    m.compress();
    assert(false && "compress() should have thrown an exception");
  } catch (const Dune::ImplicitModeCompressionBufferExhausted& e) {
    // test passed
  }
}

void testSetterInterface()
{
  ScalarMatrix m;
  m.setBuildMode(ScalarMatrix::implicit);
  m.setImplicitBuildModeParameters(3,0.1);
  m.setSize(10,10);
  buildMatrix(m);
  ScalarMatrix::CompressionStatistics stats = m.compress();
  assert(Dune::FloatCmp::eq(stats.avg,33.0/10.0));
  assert(stats.maximum == 4);
  assert(stats.overflow_total == 4);
}

void testDoubleSetSize()
{
  ScalarMatrix m;
  m.setBuildMode(ScalarMatrix::implicit);
  m.setImplicitBuildModeParameters(3,0.1);
  m.setSize(14,14);
  m.setSize(10,10);
  buildMatrix(m);
  ScalarMatrix::CompressionStatistics stats = m.compress();
  assert(Dune::FloatCmp::eq(stats.avg,33.0/10.0));
  assert(stats.maximum == 4);
  assert(stats.overflow_total == 4);
}

int testInvalidBuildModeConstructorCall()
{
  try {
    ScalarMatrix m(10,10,1,-1.0,ScalarMatrix::random);
    std::cerr<< "ERROR: Constructor should have thrown an exception!"<<std::endl;
    return 1;
  } catch (Dune::BCRSMatrixError& e) {
    // test passed
    return 0;
  }
}

int testNegativeOverflowConstructorCall()
{
  try {
    ScalarMatrix m(10,10,1,-1.0,ScalarMatrix::implicit);
    std::cerr<<"ERROR: Constructor should have thrown an exception!"<<std::endl;
    return 1;
  } catch (Dune::BCRSMatrixError& e) {
    // test passed
    return 0;
  }
}

int testInvalidSetImplicitBuildModeParameters()
{
  try {
    ScalarMatrix m;
    m.setBuildMode(ScalarMatrix::implicit);
    m.setImplicitBuildModeParameters(1,-1.0);
    std::cerr<<"ERROR: setImplicitBuildModeParameters() should have thrown an exception!"<<std::endl;
    return 1;
  } catch (Dune::BCRSMatrixError& e) {
    // test passed
    return 0;
  }
}

int testSetImplicitBuildModeParametersAfterSetSize()
{
  try {
    ScalarMatrix m;
    m.setBuildMode(ScalarMatrix::implicit);
    m.setImplicitBuildModeParameters(3,0.1);
    m.setSize(10,10);
    m.setImplicitBuildModeParameters(4,0.1);
    std::cerr<<"setImplicitBuildModeParameters() should have thrown an exception!"<<std::endl;
    return 1;
  } catch (Dune::InvalidStateException& e) {
    // test passed
    return 0;
  }
}

int testSetSizeWithNonzeroes()
{
  try {
    ScalarMatrix m;
    m.setBuildMode(ScalarMatrix::implicit);
    m.setImplicitBuildModeParameters(3,0.1);
    m.setSize(10,10,300);
    std::cerr<<"ERROR: setSize() should have thrown an exception!"<<std::endl;
    return 1;
  } catch (Dune::BCRSMatrixError& e) {
    // test passed
    return 0;
  }
}

void testCopyConstructionAndAssignment()
{
  ScalarMatrix m(10,10,3,0.1,ScalarMatrix::implicit);
  buildMatrix(m);
  m.compress();
  ScalarMatrix m2(m);
  m2 = 3.0;
  ScalarMatrix m3(m);
  m3 = m2;
  ScalarMatrix m4;
  m4 = m;
}

int testInvalidCopyConstruction()
{
  try {
    ScalarMatrix m(10,10,3,0.1,ScalarMatrix::implicit);
    buildMatrix(m);
    ScalarMatrix m2(m);
    std::cerr<<"ERROR: copy constructor should have thrown an exception!"<<std::endl;
    return 1;
  } catch (Dune::InvalidStateException& e) {
    // test passed
    return 0;
  }
}

int testInvalidCopyAssignment()
{
  ScalarMatrix m(10,10,3,0.1,ScalarMatrix::implicit);
  buildMatrix(m);
  int ret=0;
  // copy incomplete matrix into empty one
  try {
    ScalarMatrix m2;
    m2 = m;
    std::cerr<<"ERROR: operator=() should have thrown an exception!"<<std::endl;
    ++ret;
  } catch (Dune::InvalidStateException& e) {
    // test passed
  }
  // copy incomplete matrix into full one
  try {
    ScalarMatrix m2(10,10,3,0.1,ScalarMatrix::implicit);
    buildMatrix(m2);
    m2.compress();
    m2 = m;
    std::cerr<<"ERROR: operator=() should have thrown an exception!"<<std::endl;
    ++ret;
  } catch (Dune::InvalidStateException& e) {
    // test passed
  }
  // copy fully build matrix into half-built one
  m.compress();
  try {
    ScalarMatrix m2(10,10,3,0.1,ScalarMatrix::implicit);
    buildMatrix(m2);
    m2 = m;
    std::cerr<<"ERROR: operator=() should have thrown an exception!"<<std::endl;
    ++ret;
  } catch (Dune::InvalidStateException& e) {
    // test passed
  }
  return ret;
}

int testEntryConsistency()
{
  int ret=0;
  ScalarMatrix m(10,10,3,0.1,ScalarMatrix::implicit);
  if(!Dune::FloatCmp::eq(static_cast<const double&>(m.entry(0,3)),0.0))
    ret++;
  if(!Dune::FloatCmp::eq(static_cast<const double&>(m.entry(7,6)),0.0))
    ++ret;
  buildMatrix(m);
  if(!Dune::FloatCmp::eq(static_cast<const double&>(m.entry(0,3)),1.0))
    ++ret;
  if(!Dune::FloatCmp::eq(static_cast<const double&>(m.entry(7,6)),1.0))
    ++ret;
  m.entry(4,4) += 3.0;
  if(!Dune::FloatCmp::eq(static_cast<const double&>(m.entry(4,4)),4.0))
    ++ret;
  m.compress();
  if(!Dune::FloatCmp::eq(static_cast<const double&>(m[0][3]),1.0))
    ++ret;
  if(!Dune::FloatCmp::eq(static_cast<const double&>(m[7][6]),1.0))
    ++ret;
  if(!Dune::FloatCmp::eq(static_cast<const double&>(m[4][4]),4.0))
    ++ret;
  if(ret)
    std::cerr<<"ERROR: Entries are not consistent"<<std::endl;
  return ret;
}

int testEntryAfterCompress()
{
  try {
    ScalarMatrix m(10,10,3,0.1,ScalarMatrix::implicit);
    buildMatrix(m);
    m.compress();
    m.entry(3,3);
    std::cerr<<"ERROR: entry() should have thrown an exception!"<<std::endl;
    return 1;
  } catch (Dune::BCRSMatrixError& e) {
    // test passed
    return 0;
  }
}

int testBracketOperatorBeforeCompress()
{
  try {
    ScalarMatrix m(10,10,3,0.1,ScalarMatrix::implicit);
    buildMatrix(m);
    m[3][3];
    std::cerr<<"ERROR: operator[]() should have thrown an exception!"<<std::endl;
    return 1;
  } catch (Dune::BCRSMatrixError& e) {
    // test passed
    return 0;
  }
}

int testConstBracketOperatorBeforeCompress()
{
  try {
    ScalarMatrix m(10,10,3,0.1,ScalarMatrix::implicit);
    buildMatrix(m);
    const_cast<const ScalarMatrix&>(m)[3][3];
    std::cerr<<"ERROR: operator[]() should have thrown an exception!"<<std::endl;
    return 1;
  } catch (Dune::BCRSMatrixError& e) {
    // test passed
    return 0;
  }
}

void testImplicitMatrixBuilder()
{
  ScalarMatrix m(10,10,3,0.1,ScalarMatrix::implicit);
  Dune::ImplicitMatrixBuilder<ScalarMatrix> b(m);
  setMatrix(b);
  m.compress();
  setMatrix(m);
}

void testImplicitMatrixBuilderExtendedConstructor()
{
  ScalarMatrix m;
  Dune::ImplicitMatrixBuilder<ScalarMatrix> b(m,10,10,3,0.1);
  setMatrix(b);
  m.compress();
  setMatrix(m);
}

int main()
{
  int ret=0;
  try{
    testImplicitBuild();
    testImplicitBuildWithInsufficientOverflow();
    testSetterInterface();
    testDoubleSetSize();
    ret+=testInvalidBuildModeConstructorCall();
    ret+=testNegativeOverflowConstructorCall();
    ret+=testInvalidSetImplicitBuildModeParameters();
    ret+=testSetImplicitBuildModeParametersAfterSetSize();
    ret+=testSetSizeWithNonzeroes();
    testCopyConstructionAndAssignment();
    ret+=testInvalidCopyConstruction();
    ret+=testInvalidCopyAssignment();
    ret+=testEntryConsistency();
    ret+=testEntryAfterCompress();
    ret+=testBracketOperatorBeforeCompress();
    ret+=testConstBracketOperatorBeforeCompress();
    testImplicitMatrixBuilder();
    testImplicitMatrixBuilderExtendedConstructor();
  }catch(Dune::Exception& e) {
    std::cerr << e <<std::endl;
    return 1;
  }
  return ret;
}
