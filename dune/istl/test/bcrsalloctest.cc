// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/common/float_cmp.hh>

#include <dune/istl/bvector.hh>
#include <dune/istl/test/laplacian.hh>
#include <dune/istl/test/matrixtest.hh>

#include <unordered_map>
#include <mutex>
#include <cstdlib>

using namespace Dune;

std::unordered_map<void*, std::size_t> allocBlocks;
std::mutex allocMutex;

template <class T>
struct CustomAllocator
{
  using value_type = T;

  CustomAllocator() = default;

  template <class U>
  CustomAllocator(const CustomAllocator<U>&) noexcept {}

  T *allocate(std::size_t size)
  {
    T *ptr = static_cast<T*>(std::malloc(size * sizeof(T)));
    if (ptr)
    {
      std::lock_guard<std::mutex> lock(allocMutex);
      allocBlocks[ptr] = size;
    }
    return ptr;
  }

  void deallocate(T *ptr, std::size_t size)
  {
    {
      std::lock_guard<std::mutex> lock(allocMutex);
      allocBlocks.erase(ptr);
    }
    std::free(ptr);
  }

  constexpr bool operator==(const CustomAllocator &) const noexcept
  {
    return true;
  }
};

template <class T>
struct IsCustomAllocator : std::false_type {};

template <class T>
struct IsCustomAllocator<CustomAllocator<T>> : std::true_type {};

template <class A, class T>
void testValueAllocatedRange(const T &value) {
  const void* valuePtr = &value;
  std::lock_guard<std::mutex> lock(allocMutex);
  bool found = false;
  for (const auto& [blockPtr, blockSize] : allocBlocks) {
    const char* blockStart = static_cast<const char*>(blockPtr);
    const char* blockEnd = blockStart + blockSize * sizeof(T);
    if (valuePtr >= blockStart && valuePtr < blockEnd) {
      found = true;
      break;
    }
  }
  if (IsCustomAllocator<A>::value && !found) {
    throw std::runtime_error("Value address not in allocated region");
  } else if (!IsCustomAllocator<A>::value && found) {
    throw std::runtime_error("Value address found in custom allocated region for non-custom allocator");
  }
}

template<class Matrix>
void testMatrixAllocatedRange(const Matrix &mat) {
  for (auto const &row : mat)
    for (auto const &entry : row)
      testValueAllocatedRange<typename Matrix::allocator_type>(entry);
}

template<class Vector>
void testVectorAllocatedRange(const Vector &vec) {
  for (auto const &entry : vec)
    testValueAllocatedRange<typename Vector::allocator_type>(entry);
}

template <class Matrix, class Vector>
int testBCRSMatrix(int size)
{
  // Set up a test matrix
  Matrix mat;
  setupLaplacian(mat, size);

  testMatrixAllocatedRange(mat);

  // Test vector space operations
  testVectorSpaceOperations(mat);

  // Test the matrix norms
  testNorms(mat);

  // Test whether matrix class has the required constructors
  testMatrixConstructibility<Matrix>();

  // Test the matrix vector products
  Vector domain(mat.M());
  domain = 0;
  Vector range(mat.N());

  testMatrixVectorProducts(mat, domain, range);

  testVectorAllocatedRange(domain);
  testVectorAllocatedRange(range);

  return 0;
}

struct DummyBlock {};

int main(int argc, char **argv)
{
  // Test scalar matrices and vectors with default allocator
  int ret = testBCRSMatrix<BCRSMatrix<double>, BlockVector<double>>(100);

  // Test scalar matrices and vectors with custom allocator
  ret = testBCRSMatrix<BCRSMatrix<double, CustomAllocator<double>>, BlockVector<double, CustomAllocator<double>>>(100);

  // Test scalar matrices and vectors with allocator for another type
  ret = testBCRSMatrix<BCRSMatrix<double, std::allocator<DummyBlock>>, BlockVector<double, std::allocator<DummyBlock>>>(100);
  ret = testBCRSMatrix<BCRSMatrix<double, CustomAllocator<DummyBlock>>, BlockVector<double, CustomAllocator<DummyBlock>>>(100);

  // Test block matrices and vectors with trivial blocks
  using FVec = FieldVector<double, 1>;
  using FMat = FieldMatrix<double, 1, 1>;
  ret = testBCRSMatrix<BCRSMatrix<FMat>, BlockVector<FVec>>(100);

  // Test block matrices and vectors with trivial blocks and custom allocator
  ret = testBCRSMatrix<BCRSMatrix<FMat, CustomAllocator<FMat>>, BlockVector<FVec, CustomAllocator<FVec>>>(100);

  // Test block matrices and vectors with trivial blocks and allocator for another type
  ret = testBCRSMatrix<BCRSMatrix<FMat, std::allocator<DummyBlock>>, BlockVector<FVec, std::allocator<DummyBlock>>>(100);
  ret = testBCRSMatrix<BCRSMatrix<FMat, CustomAllocator<DummyBlock>>, BlockVector<FVec, CustomAllocator<DummyBlock>>>(100);

  return ret;
}
