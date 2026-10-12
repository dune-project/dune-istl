// SPDX-FileCopyrightText: Copyright (c) DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception

#include "../vectorview.hh"

#include <dune/common/test/testsuite.hh>
#include <dune/common/rangeutilities.hh>

#include <array>
#include <cmath>
#include <span>
#include <ranges>
#include <exception>

#include <dune/common/fvector.hh>

namespace {

template<class B, class IS>
class TestVectorView : public Dune::Experimental::VectorView<B, IS>
{
  using Base = Dune::Experimental::VectorView<B, IS>;

public:
  using Base::Base;
  using Base::operator=;
};

struct DenseIndexRange : public Dune::IntegralRange<int> {
  using IntegralRange::IntegralRange;
};

struct SparseIndexRange : public std::vector<int> {
    SparseIndexRange(std::size_t N, std::initializer_list<int> indices) : std::vector<int>(indices), N(N) {}
    int N;
};

} // namespace
namespace Dune {

template<class B, class IS>
struct FieldTraits<::TestVectorView<B, IS>>
{
  using field_type = typename FieldTraits<typename ::TestVectorView<B, IS>::block_type>::field_type;
  using real_type = typename FieldTraits<typename ::TestVectorView<B, IS>::block_type>::real_type;
};

template<>
struct Experimental::OrderedIndexRangeTraits<SparseIndexRange>
{
  using index_range_type = SparseIndexRange;
  using index_type = int;
  using size_type = std::size_t;

  static IntegralRange<index_type> range(const SparseIndexRange& index_set)
  {
    return { 0, index_set.N };
  }

  static size_type offset(const SparseIndexRange& index_set, const index_type& value)
  {
    const auto lb = std::lower_bound(index_set.begin(), index_set.end(), value);
    return static_cast<size_type>(std::distance(index_set.begin(), lb));
  }
};

} // namespace Dune

template<>
constexpr bool std::ranges::enable_borrowed_range<SparseIndexRange> = true;

template<>
constexpr bool std::ranges::enable_borrowed_range<DenseIndexRange> = true;


void testDenseAndSparseOps(Dune::TestSuite& test)
{
  std::array<double, 8> xData{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
  std::array<double, 3> yData{ 10.0, 20.0, 30.0 };

  const DenseIndexRange denseIndices(0, xData.size());
  const SparseIndexRange sparseIndicesData{ xData.size(), {1, 3, 7} };

  using DenseView = TestVectorView<double*, DenseIndexRange>;
  using SparseView = TestVectorView<double*, SparseIndexRange>;

  DenseView x(xData.data(), denseIndices );
  SparseView y(yData.data(), sparseIndicesData);

  test.check(x.size() == denseIndices.size(), "dense view size equals number of dense indices");
  test.check(x.N() == 8, "sparse logical span N covers [0, 8)");
  test.check(y.size() == sparseIndicesData.size(), "sparse view size equals number of sparse indices");
  test.check(y.N() == 8, "sparse logical span N covers [0, 8)");

  x = 1.5;
  for (const auto& value : xData)
    test.check(value == 1.5, "scalar assignment writes all dense entries");

  x += y;
  test.check(xData[0] == 1.5, "operator+= leaves non-overlapping index unchanged");
  test.check(xData[1] == 11.5, "operator+= updates first sparse overlap");
  test.check(xData[3] == 21.5, "operator+= updates second sparse overlap");
  test.check(xData[7] == 31.5, "operator+= updates third sparse overlap");

  x -= y;
  test.check(xData[1] == 1.5, "operator-= restores first sparse overlap");
  test.check(xData[3] == 1.5, "operator-= restores second sparse overlap");
  test.check(xData[7] == 1.5, "operator-= restores third sparse overlap");

  x.axpy(2.0, y);
  test.check(xData[1] == 21.5, "axpy updates first sparse overlap with scale");
  test.check(xData[3] == 41.5, "axpy updates second sparse overlap with scale");
  test.check(xData[7] == 61.5, "axpy updates third sparse overlap with scale");

  const auto tdot = x * y;
  const auto dot = x.dot(y);
  const double expected = xData[1] * yData[0] + xData[3] * yData[1] + xData[7] * yData[2];
  test.check(tdot == expected, "operator* accumulates over index intersection");
  test.check(dot == expected, "dot accumulates over index intersection for real field");
}

void testNorms(Dune::TestSuite& test)
{
  std::array<double, 3> values{ -2.0, 3.0, -6.0 };

  using SparseView = TestVectorView<double*, SparseIndexRange>;
  SparseView v(values.data(), SparseIndexRange{ 8, { 1, 3, 7 } });

  test.check(v.one_norm() == 11.0, "one_norm sums absolute values");
  test.check(v.one_norm_real() == 11.0, "one_norm_real matches one_norm for real values");
  test.check(v.two_norm2() == 49.0, "two_norm2 sums squared values");
  test.check(v.two_norm() == 7.0, "two_norm is sqrt(two_norm2)");
  test.check(v.infinity_norm() == 6.0, "infinity_norm is maximum absolute value");
  test.check(v.infinity_norm_real() == 6.0, "infinity_norm_real matches infinity_norm for real values");

  v *= 0.5;
  test.check(values[0] == -1.0 && values[1] == 1.5 && values[2] == -3.0,
             "operator*= scales all stored entries");

  v /= 0.5;
  test.check(values[0] == -2.0 && values[1] == 3.0 && values[2] == -6.0,
             "operator/= restores all stored entries");
}

// ---------------------------------------------------------------------------
// Block element type: FieldVector<double, 2>
// Each stored entry is a 2-component vector; all norms and arithmetic must
// recurse into the block components.
// ---------------------------------------------------------------------------
void testFieldVectorBlocks(Dune::TestSuite& test)
{
  using Block = Dune::FieldVector<double, 2>;
  using DenseView = TestVectorView<Block*, DenseIndexRange>;
  using SparseView = TestVectorView<Block*, SparseIndexRange>;

  // xData: 3 blocks at dense indices {0, 1, 2}
  std::array<Block, 3> xData{ Block{1.0, 2.0}, Block{3.0, 4.0}, Block{5.0, 6.0} };
  // yData: 2 blocks at sparse indices {1, 2}
  std::array<Block, 2> yData{ Block{10.0, 20.0}, Block{30.0, 40.0} };

  DenseView x(xData.data(), DenseIndexRange{ 0, 3 });
  SparseView y(yData.data(), SparseIndexRange{ 3, { 1, 2 } });

  // size/dim
  test.check(x.size() == 3, "fv block: dense size");
  test.check(y.size() == 2, "fv block: sparse size");

  // scalar assignment fills all components of all blocks
  x = 1.5;
  for (const auto& b : xData)
    test.check(b[0] == 1.5 && b[1] == 1.5,
               "fv block: scalar assignment fills all block components");

  // norms over block components
  // x = [[1.5, 1.5], [1.5, 1.5], [1.5, 1.5]]
  test.check(x.one_norm() == 9.0,
             "fv block: one_norm sums |component| across all blocks");
  test.check(x.two_norm2() == 6 * 1.5 * 1.5,
             "fv block: two_norm2 sums squared components across all blocks");
  test.check(x.two_norm() == std::sqrt(6 * 1.5 * 1.5),
             "fv block: two_norm is sqrt(two_norm2)");
  test.check(x.infinity_norm() == 1.5,
             "fv block: infinity_norm is max over all block components");

  // operator+=: only indices in y are updated
  x += y;
  // x[0] = [1.5, 1.5] (unchanged)
  // x[1] += y[0] = [1.5+10, 1.5+20] = [11.5, 21.5]
  // x[2] += y[1] = [1.5+30, 1.5+40] = [31.5, 41.5]
  test.check(xData[0][0] == 1.5 && xData[0][1] == 1.5,
             "fv block: operator+= leaves non-overlapping block unchanged");
  test.check(xData[1][0] == 11.5 && xData[1][1] == 21.5,
             "fv block: operator+= adds to first overlapping block component-wise");
  test.check(xData[2][0] == 31.5 && xData[2][1] == 41.5,
             "fv block: operator+= adds to second overlapping block component-wise");

  x -= y;
  test.check(xData[1][0] == 1.5 && xData[1][1] == 1.5,
             "fv block: operator-= restores first block");
  test.check(xData[2][0] == 1.5 && xData[2][1] == 1.5,
             "fv block: operator-= restores second block");

  x.axpy(2.0, y);
  test.check(xData[1][0] == 21.5 && xData[1][1] == 41.5,
             "fv block: axpy updates first block component-wise");
  test.check(xData[2][0] == 61.5 && xData[2][1] == 81.5,
             "fv block: axpy updates second block component-wise");

  // operator*: inner product over index intersection, summing block dot products
  x = 0.0;
  xData[1] = Block{ 2.0, 3.0 };
  xData[2] = Block{ 4.0, 5.0 };
  // expected = x[1]*y[0] + x[2]*y[1]
  //          = (2*10 + 3*20) + (4*30 + 5*40) = 80 + 320 = 400
  const double expected = 2.0*10.0 + 3.0*20.0 + 4.0*30.0 + 5.0*40.0;
  test.check(x * y == expected,
             "fv block: operator* accumulates block inner products over intersection");
  test.check(x.dot(y) == expected,
             "fv block: dot accumulates block inner products over intersection");
}

// ---------------------------------------------------------------------------
// Nested VectorView blocks: outer view stores inner VectorViews as entries.
// This mirrors a matrix row whose columns are themselves indexed sub-views.
// ---------------------------------------------------------------------------
void testNestedVectorViewBlocks(Dune::TestSuite& test)
{
  // Lay out a flat 4-element backing store for all inner view data.
  // inner[0] covers backing[0..1], inner[1] covers backing[2..3].
  std::array<double, 4> backing{ 1.0, 2.0, 3.0, 4.0 };

  const std::array<int, 2> innerIdx0{ 0, 1 };
  const std::array<int, 2> innerIdx1{ 0, 1 };

  using InnerView = TestVectorView<double*, DenseIndexRange>;

  // Build two inner views pointing into different halves of backing.
  std::array<InnerView, 2> rows{
    InnerView(backing.data(),     DenseIndexRange{ 0, 2 }),
    InnerView(backing.data() + 2, DenseIndexRange{ 0, 2 })
  };

  // Outer view: iterates over the two InnerView objects.
  const std::array<int, 2> outerIdx{ 0, 1 };
  using OuterIS = std::span<const int>;
  using OuterView = TestVectorView<InnerView*, OuterIS>;

  OuterView outer(rows.data(), OuterIS{ outerIdx });

  test.check(outer.size() == 2, "nested: outer size is number of inner views");

  // Read values through outer iterator
  auto it = outer.begin();
  test.check((*it)[0] == 1.0 && (*it)[1] == 2.0,
             "nested: first inner view elements accessible through outer iterator");
  ++it;
  test.check((*it)[0] == 3.0 && (*it)[1] == 4.0,
             "nested: second inner view elements accessible through outer iterator");

  // norms recurse into inner blocks
  // one_norm = |1|+|2|+|3|+|4| = 10
  test.check(outer.one_norm() == 10.0,
             "nested: one_norm recurses into inner view components");
  // two_norm2 = 1+4+9+16 = 30
  test.check(outer.two_norm2() == 30.0,
             "nested: two_norm2 recurses into inner view components");
  test.check(outer.infinity_norm() == 4.0,
             "nested: infinity_norm is max over all inner block components");

  // scalar assignment propagates into all inner blocks
  outer = 2.0;
  for (const auto& val : backing)
    test.check(val == 2.0, "nested: scalar assignment fills all inner block components");
}

void testSubsetViolationsThrow(Dune::TestSuite& test)
{
  // program must be compiled with DUNE_ISTL_WITH_CHECKING=1
  std::array<double, 3> xData{ 1.0, 2.0, 3.0 };
  std::array<double, 2> yData{ 10.0, 20.0 };

  const SparseIndexRange xIdx{ 5, {0, 2, 4} };
  const SparseIndexRange yBadIdx{ 5, {1, 4} }; // 1 is not in x index set

  using View = TestVectorView<double*, SparseIndexRange>;

  const auto expectThrow = [&test](const auto& operation,
                                   const char* message) {
    bool thrown = false;
    try {
      operation();
    } catch (const Dune::Exception&) {
      thrown = true;
    } catch (...) {
      thrown = true;
    }
    test.check(thrown, message);
  };

  {
    View x(xData.data(), xIdx);
    View y(yData.data(), yBadIdx);
    expectThrow([&]() { x += y; }, "operator+= throws on non-subset index set");
  }

  {
    View x(xData.data(), xIdx);
    View y(yData.data(), yBadIdx);
    expectThrow([&]() { x -= y; }, "operator-= throws on non-subset index set");
  }

  {
    View x(xData.data(), xIdx);
    View y(yData.data(), yBadIdx);
    expectThrow([&]() { x.axpy(2.0, y); }, "axpy throws on non-subset index set");
  }
}

int main()
{
  Dune::TestSuite test("vector view");
  testDenseAndSparseOps(test);
  testNorms(test);
  testFieldVectorBlocks(test);
  testNestedVectorViewBlocks(test);
  testSubsetViolationsThrow(test);
  return test.exit();
}
