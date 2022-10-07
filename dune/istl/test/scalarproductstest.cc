// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#include "config.h"

#include <dune/common/fvector.hh>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/test/testsuite.hh>
#include <dune/common/scalarvectorview.hh>

#include <dune/istl/bvector.hh>
#include <dune/istl/owneroverlapcopy.hh>
#include <dune/istl/scalarproducts.hh>

using namespace Dune;

// scalar ordering doesn't work for complex numbers
template <class ScalarProduct, class BlockVector>
TestSuite scalarProductTest(const ScalarProduct& scalarProduct,
                            const size_t numBlocks)
{
  TestSuite t;

  static_assert(std::is_same<BlockVector, typename ScalarProduct::domain_type>::value,
                "ScalarProduct does not properly export its vector type!");

  typedef typename ScalarProduct::field_type field_type;
  typedef typename ScalarProduct::real_type  real_type;
  const real_type myEps((real_type)1e-6);

  static_assert(std::is_same<typename FieldTraits<field_type>::real_type, real_type>::value,
                "real_type does not match field_type");

  typedef typename BlockVector::size_type size_type;

  // empty vectors
  BlockVector one(numBlocks);

  const size_type blockSize = Impl::asVector(one[0]).size();

  t.require(numBlocks==one.N());
  const size_type length = numBlocks * blockSize; // requires inner block size of VariableBlockVector to be 1!

  std::cout << __func__ << "\t \t ( " << className(one) << " )" << std::endl << std::endl;

  // initialize test vector with data
  for(size_type i=0; i < numBlocks; ++i)
    one[i] = 1.;

  field_type result = field_type();

  // global operator * tests
  result = scalarProduct.dot(one,one);

  // The Euclidean norm of a vector with all entries set to '1' equals its number of entries
  t.check(std::abs(result-field_type(length))<= myEps);

  auto sp   = scalarProduct.dot(one,one);
  auto norm = scalarProduct.norm(one);

  t.check(std::abs(sp - norm*norm) <=myEps);

  return t;
}


int main(int argc, char** argv)
{
  MPIHelper::instance(argc, argv);

  TestSuite t;
  const size_t BlockSize = 5;
  const size_t numBlocks = 10;

  // Test the ScalarProduct class
  {
    using Vector = BlockVector<FieldVector<double,BlockSize> >;
    using ScalarProduct = ScalarProduct<Vector>;
    ScalarProduct scalarProduct;
    scalarProductTest<ScalarProduct, Vector>(scalarProduct,numBlocks);
  }

  {
    using Vector = BlockVector<float>;
    using ScalarProduct = ScalarProduct<Vector>;
    ScalarProduct scalarProduct;
    scalarProductTest<ScalarProduct, Vector>(scalarProduct,numBlocks);
  }

  {
    using Vector = BlockVector<FieldVector<std::complex<float>, 1> >;
    using ScalarProduct = ScalarProduct<Vector>;
    ScalarProduct scalarProduct;
    scalarProductTest<ScalarProduct, Vector>(scalarProduct,numBlocks);
  }

  // Test the SeqScalarProduct class
  {
    using Vector = BlockVector<FieldVector<double,BlockSize> >;
    using ScalarProduct = SeqScalarProduct<Vector>;
    ScalarProduct scalarProduct;
    scalarProductTest<ScalarProduct, Vector>(scalarProduct,numBlocks);
  }

  {
    using Vector = BlockVector<float>;
    using ScalarProduct = SeqScalarProduct<Vector>;
    ScalarProduct scalarProduct;
    scalarProductTest<ScalarProduct, Vector>(scalarProduct,numBlocks);
  }

  {
    using Vector = BlockVector<FieldVector<std::complex<float>, 1> >;
    using ScalarProduct = SeqScalarProduct<Vector>;
    ScalarProduct scalarProduct;
    scalarProductTest<ScalarProduct, Vector>(scalarProduct,numBlocks);
  }

#if HAVE_MPI
  // Test the ParallelScalarProduct class
  {
    using Vector = BlockVector<FieldVector<double,BlockSize> >;
    using Comm = OwnerOverlapCopyCommunication<std::size_t,std::size_t>;
    using ScalarProduct = ParallelScalarProduct<Vector, Comm>;
    auto communicator = std::make_shared<Comm>();
    ScalarProduct scalarProduct(communicator,SolverCategory::nonoverlapping);
    scalarProductTest<ScalarProduct, Vector>(scalarProduct,numBlocks);
  }

  {
    using Vector = BlockVector<float>;
    using Comm = OwnerOverlapCopyCommunication<std::size_t,std::size_t>;
    using ScalarProduct = ParallelScalarProduct<Vector, Comm>;
    auto communicator = std::make_shared<Comm>();
    ScalarProduct scalarProduct(communicator,SolverCategory::nonoverlapping);
    scalarProductTest<ScalarProduct, Vector>(scalarProduct,numBlocks);
  }

  {
    using Vector = BlockVector<FieldVector<std::complex<double>, 1> >;
    using Comm = OwnerOverlapCopyCommunication<std::size_t,std::size_t>;
    using ScalarProduct = ParallelScalarProduct<Vector, Comm>;
    Comm communicator; // test constructor taking a const reference to the communicator
    ScalarProduct scalarProduct(communicator,SolverCategory::nonoverlapping);
    scalarProductTest<ScalarProduct, Vector>(scalarProduct,numBlocks);
  }
#endif

  return t.exit();
}
