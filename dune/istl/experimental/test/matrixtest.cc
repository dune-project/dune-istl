// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception

#include "../matrix.hh"
#include "../../bvector.hh"
#include "../../test/matrixtest.hh"

#include <dune/common/exceptions.hh>
#include <dune/common/test/testsuite.hh>
#include <dune/common/ftraits.hh>

#include <memory>
#include <utility>

namespace {

template<class T>
struct StatefulAllocator {
  using value_type = T;
  using propagate_on_container_copy_assignment = std::false_type;
  using propagate_on_container_move_assignment = std::false_type;
  using propagate_on_container_swap = std::false_type;
  using is_always_equal = std::false_type;

  StatefulAllocator() = default;
  explicit StatefulAllocator(int id_) : id(id_) {}

  template<class U>
  StatefulAllocator(const StatefulAllocator<U>& other) noexcept : id(other.id) {}

  T* allocate(std::size_t n)
  {
    return std::allocator<T>{}.allocate(n);
  }

  void deallocate(T* p, std::size_t n)
  {
    std::allocator<T>{}.deallocate(p, n);
  }

  template<class U>
  bool operator==(const StatefulAllocator<U>& other) const
  {
    return id == other.id;
  }

  template<class U>
  bool operator!=(const StatefulAllocator<U>& other) const
  {
    return !(*this == other);
  }

  template<class>
  friend struct StatefulAllocator;

  int id = 0;
};

struct AllocAwareBlock {
  using allocator_type = StatefulAllocator<AllocAwareBlock>;

  AllocAwareBlock() = default;

  explicit AllocAwareBlock(int value_) : value(value_) {}

  AllocAwareBlock(std::allocator_arg_t, const allocator_type& alloc)
    : allocatorId(alloc.id)
  {}

  AllocAwareBlock(std::allocator_arg_t, const allocator_type& alloc, const AllocAwareBlock& other)
    : allocatorId(alloc.id)
    , value(other.value)
  {}

  AllocAwareBlock(std::allocator_arg_t, const allocator_type& alloc, AllocAwareBlock&& other)
    : allocatorId(alloc.id)
    , value(other.value)
  {}

  int allocatorId = -1;
  int value = 0;
};

} // namespace

namespace Dune {

template<>
struct FieldTraits<AllocAwareBlock>
{
  using field_type = int;
  using real_type = int;
};

} // namespace Dune

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

void testMoveConstructors(Dune::TestSuite& test)
{
  using Pattern = Dune::Experimental::SparseIndexRanges<uint_least32_t>;
  using Matrix = Dune::Experimental::Matrix<double, Pattern, StatefulAllocator<double>>;

  Matrix matrix(makeDiagonalPattern(2), StatefulAllocator<double>(1));
  matrix[0][0] = 1.5;
  matrix[1][1] = 2.5;

  Matrix moved(std::move(matrix));
  test.check(moved.N() == 2 && moved.M() == 2, "move constructor preserves matrix dimensions");
  test.check(moved[0][0] == 1.5 && moved[1][1] == 2.5,
             "move constructor preserves matrix entries");

  Matrix movedWithAllocator(std::move(moved), StatefulAllocator<double>(2));
  test.check(movedWithAllocator.N() == 2 && movedWithAllocator.M() == 2,
             "move constructor with allocator preserves matrix dimensions");
  test.check(movedWithAllocator[0][0] == 1.5 && movedWithAllocator[1][1] == 2.5,
             "move constructor with allocator preserves matrix entries");
  test.check(movedWithAllocator.get_allocator() == StatefulAllocator<double>(2),
             "move constructor with allocator uses target allocator");
}

void testSwapRejectsUnequalAllocators(Dune::TestSuite& test)
{
  using Pattern = Dune::Experimental::SparseIndexRanges<uint_least32_t>;
  using Matrix = Dune::Experimental::Matrix<double, Pattern, StatefulAllocator<double>>;

  Matrix lhs(makeDiagonalPattern(1), StatefulAllocator<double>(1));
  Matrix rhs(makeDiagonalPattern(1), StatefulAllocator<double>(2));
  lhs[0][0] = 3.0;
  rhs[0][0] = 4.0;

  bool thrown = false;
  try {
    lhs.swap(rhs);
  } catch (const Dune::Exception&) {
    thrown = true;
  }

  test.check(thrown, "swap throws for unequal allocators when propagate_on_container_swap is false");
  test.check(lhs[0][0] == 3.0 && rhs[0][0] == 4.0,
             "failed swap leaves both matrices unchanged");
}

void testAllocatorAwareBlocks(Dune::TestSuite& test)
{
  using Pattern = Dune::Experimental::SparseIndexRanges<>;
  using Matrix = Dune::Experimental::Matrix<AllocAwareBlock, Pattern, StatefulAllocator<AllocAwareBlock>>;

  Matrix matrix(makeDiagonalPattern(2), StatefulAllocator<AllocAwareBlock>(7));
  test.check(matrix[0][0].allocatorId == 7 && matrix[1][1].allocatorId == 7,
             "allocator-aware block construction receives the matrix allocator");
}

void testMatrixInterface(Dune::TestSuite&)
{
  using Pattern = Dune::Experimental::SparseIndexRanges<uint_least32_t>;
  using Matrix = Dune::Experimental::Matrix<double, Pattern, StatefulAllocator<double>>;

  Matrix matrix(makeDiagonalPattern(3), StatefulAllocator<double>(3));
  matrix[0][0] = 1.0;
  matrix[1][1] = 2.0;
  matrix[2][2] = 3.0;

  Dune::BlockVector<double> x(3);
  Dune::BlockVector<double> y(3);
  x = 0.0;
  y = 0.0;

  Dune::testMatrix(matrix, x, y);
}

} // namespace

int main()
{
  Dune::TestSuite test("matrix base");

  testMatrixInterface(test);
  testMoveConstructors(test);
  testSwapRejectsUnequalAllocators(test);
  testAllocatorAwareBlocks(test);

  return test.exit();
}
