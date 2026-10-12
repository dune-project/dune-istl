// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#include "../bcrsmatrix.hh"
#include "../../bvector.hh"
#include "../../test/laplacian.hh"
#include "../../test/matrixtest.hh"

#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/common/float_cmp.hh>
#include <dune/common/timer.hh>
#include <dune/common/classname.hh>
#include <dune/common/exceptions.hh>
#include <dune/common/test/testsuite.hh>

#include <array>
#include <memory>


using namespace Dune;

namespace {

template<class Alloc = std::allocator<uint_least32_t>>
Dune::Experimental::SparseIndexRanges<uint_least32_t, Alloc> makeDiagonalPattern(std::size_t n, const Alloc& alloc = Alloc())
{
  Dune::Experimental::UnsequencedSparseIndexRangeBuilder<Alloc> builder(n, n, 1, alloc);
  for (std::size_t i = 0; i < n; ++i)
    builder.setSize(i, 1);
  for (std::size_t i = 0; i < n; ++i)
    builder.addIndex(i, i);
  return Dune::Experimental::SparseIndexRanges<uint_least32_t, Alloc>(std::move(builder));
}

int testInheritedBaseConstructors()
{
  Dune::TestSuite test("IBCRS inherited/base constructors");

  using Index = uint_least32_t;
  using Pattern = Dune::Experimental::SparseIndexRanges<Index>;
  using BaseMatrix = Dune::Experimental::Matrix<double, Pattern>;
  using Matrix = Dune::Experimental::BCRSMatrix<double, Index>;

  // Inherited allocator constructor from Matrix.
  Matrix allocatorConstructed(std::allocator<double>{});
  test.check(allocatorConstructed.N() == 0 && allocatorConstructed.M() == 0,
             "allocator constructor creates an empty matrix");
  test.check(allocatorConstructed.nonzeroes() == 0,
             "allocator constructor leaves matrix with zero entries");

  // Inherited constructor from shared pattern.
  auto sharedPattern = std::make_shared<Pattern>(makeDiagonalPattern(3));
  Matrix sharedPatternConstructed(sharedPattern);
  test.check(sharedPatternConstructed.N() == 3 && sharedPatternConstructed.M() == 3,
             "shared-pattern constructor preserves matrix dimensions");
  test.check(sharedPatternConstructed.nonzeroes() == 3,
             "shared-pattern constructor allocates matrix entries");

  // Inherited constructor from rvalue pattern.
  Matrix movedPatternConstructed(makeDiagonalPattern(2));
  test.check(movedPatternConstructed.N() == 2 && movedPatternConstructed.M() == 2,
             "rvalue-pattern constructor preserves matrix dimensions");
  test.check(movedPatternConstructed.nonzeroes() == 2,
             "rvalue-pattern constructor allocates matrix entries");

  // Constructors from Matrix base class instances.
  BaseMatrix baseCopySource(makeDiagonalPattern(2));
  baseCopySource[0][0] = 1.5;
  baseCopySource[1][1] = 2.5;
  Matrix fromBaseCopy(baseCopySource);
  test.check(fromBaseCopy.N() == 2 && fromBaseCopy.M() == 2,
             "constructor from base copy preserves dimensions");
  test.check(fromBaseCopy[0][0] == 1.5 && fromBaseCopy[1][1] == 2.5,
             "constructor from base copy preserves entries");

  BaseMatrix baseMoveSource(makeDiagonalPattern(2));
  baseMoveSource[0][0] = 3.5;
  baseMoveSource[1][1] = 4.5;
  Matrix fromBaseMove(std::move(baseMoveSource));
  test.check(fromBaseMove.N() == 2 && fromBaseMove.M() == 2,
             "constructor from base move preserves dimensions");
  test.check(fromBaseMove[0][0] == 3.5 && fromBaseMove[1][1] == 4.5,
             "constructor from base move preserves entries");

  return test.exit();
}

} // namespace

template <class Matrix, class Vector>
int testBCRSMatrix(int size)
{
  Matrix mat;
  setupLaplacian(mat, size);
  testVectorSpaceOperations(mat);
  testNorms(mat);
  testMatrixConstructibility<Matrix>();
  // Test the matrix vector products
  Vector domain(mat.M());
  domain = 0;
  Vector range(mat.N());

  testMatrixVectorProducts(mat,domain,range);

  return 0;
}

int main(int argc, char** argv)
{
  std::size_t size = 10;
  if (argc > 1)
    size = std::stoul(argv[1]);

  int ret = 0;
  ret += testInheritedBaseConstructors();
  int rep = 3;
  // Test scalar matrices and vectors
  for (int i = 0; i != rep; ++i)
    ret = testBCRSMatrix<Experimental::BCRSMatrix<double>, BlockVector<double> >(size);

  // Test block matrices and vectors with trivial blocks
  for (int i = 0; i != rep; ++i)
    ret = testBCRSMatrix<Experimental::BCRSMatrix<FieldMatrix<double,1,1> >, BlockVector<FieldVector<double,1> > >(size);

  for (int i = 0; i != rep; ++i)
    ret = testBCRSMatrix<Experimental::BCRSMatrix<double>, BlockVector<double> >(size);

  // Test scalar matrices with complex numbers
  ret = testBCRSMatrix<Experimental::BCRSMatrix<std::complex<double>>, BlockVector<std::complex<double>>>(10);

  // Dimension-sensitive check: non-square matrix catches swapped domain/range arguments.
  {
    using Matrix = Experimental::BCRSMatrix<double>;
    constexpr std::size_t rows = 4;
    constexpr std::size_t cols = 7;

    Matrix mat;
    mat.setBuildMode(Matrix::row_wise);
    mat.setSize(rows, cols, rows * 3);

    for (auto row = mat.createbegin(); row != mat.createend(); ++row)
    {
      const std::size_t i = row.index();
      const std::array<std::size_t, 3> pattern = {i % cols, (i + 2) % cols, (i + 4) % cols};
      for (const auto j : pattern)
        row.insert(j);
    }

    for (auto row = mat.begin(); row != mat.end(); ++row)
      for (auto col = row->begin(); col != row->end(); ++col)
        *col = 0.2 * static_cast<double>(1 + row.index() + 2 * col.index());

    BlockVector<double> domain(cols);
    BlockVector<double> range(rows);
    testMatrixVectorProducts(mat, domain, range);
  }

  // ////////////////////////////////////////////////////////////
  //   Test the BCRSMatrix class -- a sparse dynamic matrix
  // ////////////////////////////////////////////////////////////

  {
    Experimental::BCRSMatrix<double> bcrsMatrix(4,4, Experimental::BCRSMatrix<double>::random);

    bcrsMatrix.setrowsize(0,2);
    bcrsMatrix.setrowsize(1,3);
    bcrsMatrix.setrowsize(2,3);
    bcrsMatrix.setrowsize(3,2);

    bcrsMatrix.endrowsizes();

    bcrsMatrix.addindex(0, 0);
    bcrsMatrix.addindex(0, 1);

    bcrsMatrix.addindex(1, 0);
    bcrsMatrix.addindex(1, 1);
    bcrsMatrix.addindex(1, 2);

    bcrsMatrix.addindex(2, 1);
    bcrsMatrix.addindex(2, 2);
    bcrsMatrix.addindex(2, 3);

    bcrsMatrix.addindex(3, 2);
    bcrsMatrix.addindex(3, 3);

    bcrsMatrix.endindices();

    for (auto row = bcrsMatrix.begin(); row != bcrsMatrix.end(); ++row)
      for (auto col = row->begin(); col != row->end(); ++col)
        *col = 1.0 + static_cast<double>(row.index()) * static_cast<double>(col.index());

    BlockVector<double> x(4), y(4);
    testMatrix(bcrsMatrix, x, y);

    // Test whether matrix resizing works
    int resize = 3;
    bcrsMatrix.setBuildMode(Experimental::BCRSMatrix<double>::random);
    bcrsMatrix.setSize(resize, resize, resize);

    for (int i = 0; i < resize; i++)
      bcrsMatrix.setrowsize(i, 1);
    bcrsMatrix.endrowsizes();

    for (int i = 0; i < resize; i++)
      bcrsMatrix.addindex(i, i);
    bcrsMatrix.endindices();

    for (int i = 0; i < resize; i++)
      bcrsMatrix[i][i] = 1.0;

    x.resize(resize);
    y.resize(resize);
    testMatrix(bcrsMatrix, x, y);
  }

  // ////////////////////////////////////////////////////////////
  //   Test the BCRSMatrix class with FieldMatrix entries
  // ////////////////////////////////////////////////////////////

  {
    Experimental::BCRSMatrix<FieldMatrix<double,2,2>> bcrsMatrix2x2(4, 4, Experimental::BCRSMatrix<FieldMatrix<double,2,2>>::random);

    bcrsMatrix2x2.setrowsize(0,2);
    bcrsMatrix2x2.setrowsize(1,3);
    bcrsMatrix2x2.setrowsize(2,3);
    bcrsMatrix2x2.setrowsize(3,2);

    bcrsMatrix2x2.endrowsizes();

    bcrsMatrix2x2.addindex(0, 0);
    bcrsMatrix2x2.addindex(0, 1);

    bcrsMatrix2x2.addindex(1, 0);
    bcrsMatrix2x2.addindex(1, 1);
    bcrsMatrix2x2.addindex(1, 2);

    bcrsMatrix2x2.addindex(2, 1);
    bcrsMatrix2x2.addindex(2, 2);
    bcrsMatrix2x2.addindex(2, 3);

    bcrsMatrix2x2.addindex(3, 2);
    bcrsMatrix2x2.addindex(3, 3);

    bcrsMatrix2x2.endindices();

    for (auto row = bcrsMatrix2x2.begin(); row != bcrsMatrix2x2.end(); ++row)
      for (auto col = row->begin(); col != row->end(); ++col)
        *col = 1.0 + static_cast<double>(row.index()) * static_cast<double>(col.index());

    testSuperMatrix(bcrsMatrix2x2);

    // Test whether matrix resizing works
    int resize = 3;
    bcrsMatrix2x2.setBuildMode(Experimental::BCRSMatrix<FieldMatrix<double,2,2>>::random);
    bcrsMatrix2x2.setSize(resize, resize, resize);

    for (int i = 0; i < resize; i++)
      bcrsMatrix2x2.setrowsize(i, 1);
    bcrsMatrix2x2.endrowsizes();

    for (int i = 0; i < resize; i++)
      bcrsMatrix2x2.addindex(i, i);
    bcrsMatrix2x2.endindices();

    for (int i = 0; i < resize; i++)
      bcrsMatrix2x2[i][i] = 1.0;

    testSuperMatrix(bcrsMatrix2x2);
  }

  // ////////////////////////////////////////////////////////////
  //   Regression tests for BCRSMatrix state handling
  // ////////////////////////////////////////////////////////////

  {
    using Matrix = Experimental::BCRSMatrix<double>;

    Matrix mat;
    if (mat.buildMode() != Matrix::unknown) {
      std::cerr << "ERROR: Default-constructed matrix must report BuildMode::unknown" << std::endl;
      ++ret;
    }

    mat.setBuildMode(Matrix::row_wise);
    if (mat.buildMode() != Matrix::row_wise) {
      std::cerr << "ERROR: setBuildMode(row_wise) not reflected by buildMode()" << std::endl;
      ++ret;
    }

    mat.setBuildMode(Matrix::implicit);
    if (mat.buildMode() != Matrix::implicit) {
      std::cerr << "ERROR: setBuildMode(implicit) not reflected by buildMode()" << std::endl;
      ++ret;
    }

    mat.setBuildMode(Matrix::unknown);
    if (mat.buildMode() != Matrix::unknown) {
      std::cerr << "ERROR: setBuildMode(unknown) not reflected by buildMode()" << std::endl;
      ++ret;
    }

    mat.setBuildMode(Matrix::row_wise);
    mat.setSize(0, 0, 0);

    // Finalizing a build must keep matrix data and reset build mode to unknown.
    mat.setBuildMode(Matrix::implicit);
    mat.setSize(2, 2, 2);
    mat.setrowsize(0, 1);
    mat.setrowsize(1, 1);
    mat.endrowsizes();
    mat.addindex(0, 0);
    mat.addindex(1, 1);
    mat.endindices();
    mat[0][0] = 3.0;
    mat[1][1] = 4.0;

    if (mat.buildMode() != Matrix::unknown) {
      std::cerr << "ERROR: finalized matrix must report BuildMode::unknown" << std::endl;
      ++ret;
    }
    if (mat.N() != 2 || mat.M() != 2 || mat[0][0] != 3.0 || mat[1][1] != 4.0) {
      std::cerr << "ERROR: finalizing build must preserve constructed matrix" << std::endl;
      ++ret;
    }

    // Self-assignment must be a no-op and keep the matrix usable.
    mat = mat;
    if (mat.N() != 2 || mat.M() != 2 || mat[0][0] != 3.0 || mat[1][1] != 4.0 || mat.buildMode() != Matrix::unknown) {
      std::cerr << "ERROR: self-copy-assignment must preserve matrix state" << std::endl;
      ++ret;
    }
  }

  {
    using Matrix = Experimental::BCRSMatrix<double>;

    Matrix rowWise;
    rowWise.setBuildMode(Matrix::row_wise);
    rowWise.setSize(3, 3, 6);

    Matrix implicit;
    implicit.setBuildMode(Matrix::implicit);
    implicit.setSize(3, 3, 6);

    rowWise.swap(implicit);

    if (rowWise.buildMode() != Matrix::implicit) {
      std::cerr << "ERROR: swap() must swap active builder state (expected implicit)" << std::endl;
      ++ret;
    }

    if (implicit.buildMode() != Matrix::row_wise) {
      std::cerr << "ERROR: swap() must swap active builder state (expected row_wise)" << std::endl;
      ++ret;
    }

    try {
      rowWise.setrowsize(0, 2);
      rowWise.setrowsize(1, 1);
      rowWise.setrowsize(2, 1);
      rowWise.endrowsizes();
      rowWise.addindex(0, 0);
      rowWise.addindex(0, 1);
      rowWise.addindex(1, 1);
      rowWise.addindex(2, 2);
      rowWise.endindices();
    } catch (const Dune::Exception& e) {
      std::cerr << "ERROR: swapped implicit builder became unusable: " << e.what() << std::endl;
      ++ret;
    }

    try {
      for (auto row = implicit.createbegin(); row != implicit.createend(); ++row) {
        row.insert(row.index());
      }
    } catch (const Dune::Exception& e) {
      std::cerr << "ERROR: swapped row-wise builder became unusable: " << e.what() << std::endl;
      ++ret;
    }
  }

  return ret;
}
