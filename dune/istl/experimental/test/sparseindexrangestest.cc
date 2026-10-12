// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception

#include "../sparseindexranges.hh"

#include <dune/common/exceptions.hh>
#include <dune/common/test/testsuite.hh>

#include <vector>
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <set>

namespace {

template<class IndexRanges>
constexpr bool checkCanonicalPattern(const IndexRanges& cis)
{
  bool checks = true;
  checks &= (cis.size() == 3);
  checks &= (cis.range().size() == 6);
  checks &= (cis.count() == 3);
  checks &= (cis.offset(0) == 0);
  checks &= (cis.offset(1) == 2);
  checks &= (cis.offset(2) == 3);
  checks &= (cis.offset(3) == 3);

  std::size_t nnz = 0;
  for (const auto& row : cis)
    for (const auto col : row)
      nnz += static_cast<std::size_t>(col < cis.range().size());

  checks &= (nnz == cis.count());

  const std::array<std::size_t, 2> expectedRow0{1, 2};
  std::size_t pos = 0;
  for (const auto idx : cis[0])
  {
    checks &= (pos < expectedRow0.size());
    if (pos < expectedRow0.size())
      checks &= (idx == expectedRow0[pos]);
    ++pos;
  }
  checks &= (pos == expectedRow0.size());

  pos = 0;
  for (const auto idx : cis[1])
  {
    checks &= (pos == 0);
    checks &= (idx == 0);
    ++pos;
  }
  checks &= (pos == 1);
  checks &= (cis[2].begin() == cis[2].end());

  const auto lb0eq1 = cis[0].lower_bound(1);
  const auto lb0eq2 = cis[0].lower_bound(2);
  const auto lb0gt2 = cis[0].lower_bound(3);
  checks &= (lb0eq1 != cis[0].end() && *lb0eq1 == 1);
  checks &= (lb0eq2 != cis[0].end() && *lb0eq2 == 2);
  checks &= (lb0gt2 == cis[0].end());

  std::size_t sum = 0;
  std::for_each(cis[0].begin(), cis[0].end(), [&sum](auto v) { sum += static_cast<std::size_t>(v); });
  checks &= (sum == 3);

  return checks;
}

template<class IndexRanges>
constexpr bool constTest() {
  using namespace Dune::Experimental;
  bool checks = true;

  UnsequencedSparseIndexRangeBuilder isBuilder(3, 6);
  isBuilder.setSize(0, 3);
  isBuilder.setSize(0, 3);
  isBuilder.setSize(1, 2);
  isBuilder.setSize(2, 1);

  isBuilder.addIndex(0, 1);
  isBuilder.addIndex(0, 2);
  isBuilder.addIndex(1, 0);
  isBuilder.addIndex(1, 0);

  auto cis = IndexRanges(std::move(isBuilder));
  checks &= checkCanonicalPattern(cis);

  IndexRanges ccis = std::move(cis);
  checks &= checkCanonicalPattern(ccis);

  // different construction path, but same pattern
  UnsequencedSparseIndexRangeBuilder cccisBuilder(3, 6);
  cccisBuilder.setSize(2, 0);
  cccisBuilder.setSize(1, 1);
  cccisBuilder.setSize(0, 2);

  cccisBuilder.addIndex(0, 2);
  cccisBuilder.addIndex(1, 0);
  cccisBuilder.addIndex(0, 1);

  auto cccis = IndexRanges(std::move(cccisBuilder));
  checks &= checkCanonicalPattern(cccis);
  checks &= (ccis == cccis);

  SequencedSparseIndexRangeBuilder seqBuilder(3, 6, 1);
  for (auto& rowBuilder : seqBuilder) {
    checks &= !rowBuilder.contains(1);
    rowBuilder.insert(1);
    rowBuilder.insert(1);
    checks &= rowBuilder.contains(1);
    checks &= !rowBuilder.contains(2);
    rowBuilder.insert(2);
    checks &= rowBuilder.contains(2);
  }
  auto seqIs = IndexRanges(std::move(seqBuilder));
  checks &= (seqIs.size() == 3);
  checks &= (seqIs.count() == 6);
  checks &= (seqIs[0].size() == 2);
  checks &= (seqIs[1].size() == 2);
  checks &= (seqIs[2].size() == 2);
  checks &= (seqIs[1].lower_bound(1) != seqIs[1].end());
  checks &= (seqIs[1].lower_bound(2) != seqIs[1].end());

  UnsequencedSparseIndexRangeBuilder unseqBuilder(3, 6, 1);
  unseqBuilder.addIndex(0, 1);
  unseqBuilder.addIndex(0, 1);
  unseqBuilder.addIndex(1, 0);
  unseqBuilder.addIndex(1, 0);
  unseqBuilder.addIndex(0, 2);

  auto unseqIs = IndexRanges(std::move(unseqBuilder));
  checks &= checkCanonicalPattern(unseqIs);
  checks &= (unseqIs == cccis);

  return checks;
}

template<class IndexRanges>
constexpr bool constexprPreSizedUnsequencedSparseIndexRangeBuilderTest()
{
  using namespace Dune::Experimental;
  UnsequencedSparseIndexRangeBuilder builder(3, 6);
  builder.setSize(0, 3);
  builder.setSize(1, 2);
  builder.setSize(2, 1);

  builder.addIndex(0, 1);
  builder.addIndex(0, 2);
  builder.addIndex(1, 0);
  builder.addIndex(1, 0);
  auto cis = IndexRanges(std::move(builder));
  return checkCanonicalPattern(cis);
}

template<class IndexRanges>
constexpr bool constexprSequencedSparseIndexRangeBuilderTest()
{
  using namespace Dune::Experimental;
  SequencedSparseIndexRangeBuilder builder(3, 6, 1);
  for (auto& rowBuilder : builder)
  {
    rowBuilder.insert(2);
    rowBuilder.insert(1);
    rowBuilder.insert(1);
  }

  auto cis = IndexRanges(std::move(builder));
  bool checks = true;
  checks &= (cis.size() == 3);
  checks &= (cis.count() == 6);
  checks &= (cis[0].size() == 2 && cis[1].size() == 2 && cis[2].size() == 2);
  checks &= (cis.offset(0) == 0 && cis.offset(1) == 2 && cis.offset(2) == 4 && cis.offset(3) == 6);

  for (std::size_t i = 0; i < cis.size(); ++i)
  {
    const auto lb1 = cis[i].lower_bound(1);
    const auto lb2 = cis[i].lower_bound(2);
    checks &= (lb1 != cis[i].end() && *lb1 == 1);
    checks &= (lb2 != cis[i].end() && *lb2 == 2);
  }

  return checks;
}

template<class IndexRanges>
constexpr bool constexprSequencedOverrideIndicesTest()
{
  using namespace Dune::Experimental;
  SequencedSparseIndexRangeBuilder builder(2, 8, 4);
  auto row = builder.begin();

  row.insert(1);
  row.insert(5);

  constexpr std::array<std::size_t, 3> firstOverride{4, 4, 2};
  row.overrideIndices(firstOverride.begin(), firstOverride.end());

  bool checks = true;
  checks &= (row.size() == 2);
  checks &= row.contains(2);
  checks &= row.contains(4);
  checks &= !row.contains(1);
  checks &= !row.contains(5);

  constexpr std::array<std::size_t, 1> secondOverride{3};
  row.overrideIndices(secondOverride.begin(), secondOverride.end());

  checks &= (row.size() == 1);
  checks &= row.contains(3);
  checks &= !row.contains(2);
  checks &= !row.contains(4);

  ++row;
  row.insert(7);
  ++row;

  const auto cis = IndexRanges(std::move(builder));
  checks &= (cis.size() == 2);
  checks &= (cis.count() == 2);
  checks &= (cis[0].size() == 1 && cis[1].size() == 1);
  checks &= (cis[0].begin() != cis[0].end() && *cis[0].begin() == 3);
  checks &= (cis[1].begin() != cis[1].end() && *cis[1].begin() == 7);

  return checks;
}

template<class IndexRanges>
constexpr bool constexprUnsequencedSparseIndexRangeBuilderTest()
{
  using namespace Dune::Experimental;
  UnsequencedSparseIndexRangeBuilder builder(3, 6, 1);
  builder.addIndex(0, 2);
  builder.addIndex(1, 0);
  builder.addIndex(0, 1);
  builder.addIndex(1, 0);
  builder.addIndex(0, 2);
  auto cis = IndexRanges(std::move(builder));
  return checkCanonicalPattern(cis);
}

template<class IndexRanges>
constexpr bool constexprMoveAndEqualityTest()
{
  using namespace Dune::Experimental;
  UnsequencedSparseIndexRangeBuilder builder(3, 6);
  builder.setSize(0, 2);
  builder.setSize(1, 1);
  builder.setSize(2, 0);

  builder.addIndex(0, 1);
  builder.addIndex(0, 2);
  builder.addIndex(1, 0);
  auto lhs = IndexRanges(std::move(builder));

  UnsequencedSparseIndexRangeBuilder otherBuilder(3, 6, 1);
  otherBuilder.addIndex(0, 2);
  otherBuilder.addIndex(0, 1);
  otherBuilder.addIndex(1, 0);
  auto rhs = IndexRanges(std::move(otherBuilder));

  IndexRanges moved{std::move(lhs)};
  return checkCanonicalPattern(moved) && (moved == rhs);
}

template<class IndexRanges>
constexpr bool constexprIndexRangeTraitsTest()
{
  using namespace Dune::Experimental;
  UnsequencedSparseIndexRangeBuilder builder(3, 6);
  builder.setSize(0, 2);
  builder.setSize(1, 1);
  builder.setSize(2, 0);

  builder.addIndex(0, 1);
  builder.addIndex(0, 2);
  builder.addIndex(1, 0);

  const auto cis = IndexRanges(std::move(builder));
  using View = decltype(cis[0]);
  using Traits = OrderedIndexRangeTraits<View>;

  bool checks = true;
  checks &= (Traits::range(cis[0]).size() == 6);
  checks &= (Traits::range(cis[1]).size() == 6);
  checks &= (Traits::range(cis[2]).size() == 6);

  checks &= (Traits::offset(cis[0], 0) == 0);
  checks &= (Traits::offset(cis[0], 1) == 0);
  checks &= (Traits::offset(cis[0], 2) == 1);
  checks &= (Traits::offset(cis[0], 3) == 2);

  checks &= (Traits::offset(cis[1], 0) == 0);
  checks &= (Traits::offset(cis[1], 1) == 1);

  checks &= (Traits::offset(cis[2], 0) == 0);

  return checks;
}

template<class IndexRanges>
void runBuilderBehaviorTests(Dune::TestSuite& test, const std::string& prefix)
{
  using namespace Dune::Experimental;
  SequencedSparseIndexRangeBuilder seqBuilder(3, 6, 1);
  std::size_t currentRow = 0;
  for (auto& rowBuilder : seqBuilder)
  {
    test.check(rowBuilder.index() == currentRow, "sequenced builder exposes current row index");
    test.check(rowBuilder.size() == 0, "sequenced builder starts row with zero entries");

    rowBuilder.insert(2);
    rowBuilder.insert(1);
    rowBuilder.insert(1);

    test.check(rowBuilder.contains(1), "contains() finds inserted index");
    test.check(rowBuilder.contains(2), "contains() finds second inserted index");
    test.check(!rowBuilder.contains(3), "contains() rejects missing index");
    test.check(rowBuilder.size() == 2, "duplicate insertions are ignored");
    ++currentRow;
  }

  auto seqIs = IndexRanges(std::move(seqBuilder));
  test.check(seqIs.count() == 6, prefix + ": sequenced builder stores unique row entries");

  UnsequencedSparseIndexRangeBuilder unseqBuilder(3, 6, 1);
  unseqBuilder.addIndex(1, 0);
  unseqBuilder.addIndex(0, 2);
  unseqBuilder.addIndex(2, 4);
  unseqBuilder.addIndex(0, 1);
  unseqBuilder.addIndex(2, 4);
  unseqBuilder.addIndex(2, 3);
  auto unseqIs = IndexRanges(std::move(unseqBuilder));

  test.check(unseqIs[0].size() == 2, prefix + ": unsequenced builder row 0 size");
  test.check(unseqIs[1].size() == 1, prefix + ": unsequenced builder row 1 size");
  test.check(unseqIs[2].size() == 2, prefix + ": unsequenced builder row 2 size");

  const auto row2lb = unseqIs[2].lower_bound(4);
  test.check(row2lb != unseqIs[2].end() && *row2lb == 4, prefix + ": lower_bound() finds value in unsequenced row");

  std::size_t row2sum = 0;
  std::for_each(unseqIs[2].begin(), unseqIs[2].end(), [&row2sum](auto v) { row2sum += static_cast<std::size_t>(v); });
  test.check(row2sum == 7, prefix + ": for_each() visits all row entries");
}

template<class IndexRanges>
void runRandomCompressionCase(Dune::TestSuite& test,
                              const std::string& caseName,
                              std::size_t rows,
                              std::size_t cols,
                              std::size_t samples,
                              std::size_t requiredMaxIndex,
                              std::uint32_t seed)
{
  // create a random pattern with the given average number of entries per row
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<std::size_t> col_dist(0, static_cast<std::size_t>(std::min(requiredMaxIndex-1, cols-1)));
  std::uniform_int_distribution<std::size_t> row_dist(0, static_cast<std::size_t>(rows-1));

  std::vector<std::array<std::size_t, 2>> vecPattern;
  std::vector<std::set<std::size_t>> setPattern(rows);

  for (std::size_t i = 0; i != samples; ++i) {
    auto [row, col] =  vecPattern.emplace_back(std::array<std::size_t, 2>{row_dist(rng), col_dist(rng)});
    setPattern[row].insert(col);
  }

  using namespace Dune::Experimental;
  UnsequencedSparseIndexRangeBuilder unseqBuilder(rows, cols);
  UnsequencedSparseIndexRangeBuilder sizedUnseqBuilder(rows, cols);
  SequencedSparseIndexRangeBuilder seqBuilder(rows, cols);

  for (std::size_t row = 0; row != rows; ++row)
    sizedUnseqBuilder.setSize(row, setPattern[row].size());

  for (auto [row, col] : vecPattern) {
    sizedUnseqBuilder.addIndex(row, col);
    unseqBuilder.addIndex(row, col);
  }

  auto createBegin = seqBuilder.begin();
  for (std::size_t row = 0; row != rows; ++row) {
    for (const auto col : setPattern[row])
      createBegin.insert(col);
    ++createBegin;
  }

  auto unseqIS = IndexRanges(std::move(unseqBuilder));
  auto sizedUnseqIS = IndexRanges(std::move(sizedUnseqBuilder));
  auto seqIS = IndexRanges(std::move(seqBuilder));

  std::size_t count = std::accumulate(setPattern.begin(), setPattern.end(), std::size_t(0), [](std::size_t sum, const auto& row) { return sum + row.size(); });
  auto checkIS = [&](const auto& is) {
    test.check(is.size() == rows, caseName + ": size is correct");
    test.check(is.range().size() == cols, caseName + ": range is correct");
    test.check(is.count() == count, caseName + ": count is correct");
    std::size_t offset = 0;
    for (std::size_t row = 0; row < rows; ++row)
    {
      test.check(std::find(is[row].begin(), is[row].end(), requiredMaxIndex) == std::end(is[row]), caseName + ": preserves required max index");
      test.check(std::is_sorted(is[row].begin(), is[row].end()), caseName + ": row indices are sorted");
      test.check(is.offset(row) == offset, caseName + ": offset(" + std::to_string(row) + ") is correct");
      offset += is[row].size();
      for (const auto col : is[row])
        test.check(setPattern[row].contains(col), caseName + ": row contains expected column index");
      for (const auto col : setPattern[row])
        test.check(is[row].contains(col), caseName + ": expected column index is present in row");
    }
  };

  checkIS(unseqIS);
  checkIS(sizedUnseqIS);
  checkIS(seqIS);
}

} // namespace

int main()
{
  using namespace Dune::Experimental;
  Dune::TestSuite test("SparseIndexRanges");
  // GCC 10 constexpr support is incomplete, so we exclude tests that are know to fail.
#if !(defined(__GNUC__) && !defined(__clang__) && __GNUC__ <= 10)
  static_assert(constexprUnsequencedSparseIndexRangeBuilderTest<SparseIndexRanges<>>(), "SparseIndexRanges unsequenced builder must work in constant evaluation");
  static_assert(constexprSequencedSparseIndexRangeBuilderTest<SparseIndexRanges<>>(), "SparseIndexRanges sequenced builder must work in constant evaluation");
  static_assert(constexprSequencedSparseIndexRangeBuilderTest<CompressedSparseIndexRanges<>>(), "CompressedSparseIndexRanges sequenced builder must work in constant evaluation");
  static_assert(constexprSequencedOverrideIndicesTest<SparseIndexRanges<>>(), "SparseIndexRanges sequenced overrideIndices must work in constant evaluation");
  static_assert(constexprSequencedOverrideIndicesTest<CompressedSparseIndexRanges<>>(), "CompressedSparseIndexRanges sequenced overrideIndices must work in constant evaluation");
  static_assert(constexprPreSizedUnsequencedSparseIndexRangeBuilderTest<CompressedSparseIndexRanges<>>(), "CompressedSparseIndexRanges unsequenced pre-sized builder must work in constant evaluation");
  static_assert(constexprUnsequencedSparseIndexRangeBuilderTest<CompressedSparseIndexRanges<>>(), "CompressedSparseIndexRanges unsequenced builder must work in constant evaluation");
  static_assert(constexprMoveAndEqualityTest<CompressedSparseIndexRanges<>>(), "CompressedSparseIndexRanges move and equality must work in constant evaluation");
  static_assert(constexprIndexRangeTraitsTest<CompressedSparseIndexRanges<>>(), "CompressedSparseIndexRanges row traits must work in constant evaluation");
  static_assert(constTest<SparseIndexRanges<>>(), "SparseIndexRanges combined constexpr scenario must pass");
  static_assert(constTest<CompressedSparseIndexRanges<>>(), "CompressedSparseIndexRanges combined constexpr scenario must pass");
#endif
  static_assert(constexprPreSizedUnsequencedSparseIndexRangeBuilderTest<SparseIndexRanges<>>(), "SparseIndexRanges unsequenced pre-sized builder must work in constant evaluation");
  static_assert(constexprMoveAndEqualityTest<SparseIndexRanges<>>(), "SparseIndexRanges move and equality must work in constant evaluation");
  static_assert(constexprIndexRangeTraitsTest<SparseIndexRanges<>>(), "SparseIndexRanges row traits must work in constant evaluation");

  test.check(constexprPreSizedUnsequencedSparseIndexRangeBuilderTest<SparseIndexRanges<>>(), "SparseIndexRanges unsequenced pre-sized builder constexpr test");
  test.check(constexprPreSizedUnsequencedSparseIndexRangeBuilderTest<CompressedSparseIndexRanges<>>(), "CompressedSparseIndexRanges unsequenced pre-sized builder constexpr test");
  test.check(constexprUnsequencedSparseIndexRangeBuilderTest<SparseIndexRanges<>>(), "SparseIndexRanges unsequenced builder constexpr test");
  test.check(constexprUnsequencedSparseIndexRangeBuilderTest<CompressedSparseIndexRanges<>>(), "CompressedSparseIndexRanges unsequenced builder constexpr test");
  test.check(constexprSequencedSparseIndexRangeBuilderTest<SparseIndexRanges<>>(), "SparseIndexRanges sequenced builder constexpr test");
  test.check(constexprSequencedSparseIndexRangeBuilderTest<CompressedSparseIndexRanges<>>(), "CompressedSparseIndexRanges sequenced builder constexpr test");
  test.check(constexprSequencedOverrideIndicesTest<SparseIndexRanges<>>(), "SparseIndexRanges sequenced overrideIndices constexpr test");
  test.check(constexprSequencedOverrideIndicesTest<CompressedSparseIndexRanges<>>(), "CompressedSparseIndexRanges sequenced overrideIndices constexpr test");
  test.check(constexprMoveAndEqualityTest<SparseIndexRanges<>>(), "SparseIndexRanges move/equality constexpr test");
  test.check(constexprMoveAndEqualityTest<CompressedSparseIndexRanges<>>(), "CompressedSparseIndexRanges move/equality constexpr test");
  test.check(constexprIndexRangeTraitsTest<SparseIndexRanges<>>(), "SparseIndexRanges row traits constexpr test");
  test.check(constexprIndexRangeTraitsTest<CompressedSparseIndexRanges<>>(), "CompressedSparseIndexRanges row traits constexpr test");
  test.check(constTest<SparseIndexRanges<>>(), "SparseIndexRanges combined constexpr test");
  test.check(constTest<CompressedSparseIndexRanges<>>(), "CompressedSparseIndexRanges combined constexpr test");

  {
    UnsequencedSparseIndexRangeBuilder isBuilder(3, 6);

    isBuilder.setSize(0, 3);
    isBuilder.setSize(1, 2);
    isBuilder.setSize(2, 1);

    isBuilder.addIndex(0, 1);
    isBuilder.addIndex(0, 2);
    isBuilder.addIndex(1, 0);
    isBuilder.addIndex(1, 0);

    auto is = CompressedSparseIndexRanges<>(std::move(isBuilder));

    test.check(is.size() == 3, "size is stored correctly");
    test.check(is.range().size() == 6, "cols are stored correctly");
    test.check(is.count() == 3, "compress removes unused slots");
    test.check(is.offset(0) == 0, "offset(0) after compression");
    test.check(is.offset(1) == 2, "offset(1) after compression");
    test.check(is.offset(2) == 3, "offset(2) after compression");
    test.check(is.offset(3) == 3, "offset(3) after compression");

    const std::vector<std::size_t> row0(is[0].begin(), is[0].end());
    const std::vector<std::size_t> row1(is[1].begin(), is[1].end());
    const std::vector<std::size_t> row2(is[2].begin(), is[2].end());

    test.check(row0 == std::vector<std::size_t>({1, 2}), "row 0 stores sorted indices");
    test.check(row1 == std::vector<std::size_t>({0}), "row 1 stores unique index");
    test.check(row2.empty(), "row 2 is empty");
  }

  {
    UnsequencedSparseIndexRangeBuilder isBuilder(3, 6);

    isBuilder.setSize(0, 3);
    isBuilder.setSize(1, 2);
    isBuilder.setSize(2, 1);

    isBuilder.addIndex(0, 1);
    isBuilder.addIndex(0, 2);
    isBuilder.addIndex(1, 0);
    isBuilder.addIndex(1, 0);

    auto is = SparseIndexRanges<>(std::move(isBuilder));

    test.check(is.size() == 3, "sparse size is stored correctly");
    test.check(is.range().size() == 6, "sparse cols are stored correctly");
    test.check(is.count() == 3, "sparse compression removes unused slots");
    test.check(is.offset(0) == 0, "sparse offset(0) after compression");
    test.check(is.offset(1) == 2, "sparse offset(1) after compression");
    test.check(is.offset(2) == 3, "sparse offset(2) after compression");
    test.check(is.offset(3) == 3, "sparse offset(3) after compression");

    const std::vector<std::size_t> row0(is[0].begin(), is[0].end());
    const std::vector<std::size_t> row1(is[1].begin(), is[1].end());
    const std::vector<std::size_t> row2(is[2].begin(), is[2].end());

    test.check(row0 == std::vector<std::size_t>({1, 2}), "sparse row 0 stores sorted indices");
    test.check(row1 == std::vector<std::size_t>({0}), "sparse row 1 stores unique index");
    test.check(row2.empty(), "sparse row 2 is empty");
    }

    runBuilderBehaviorTests<SparseIndexRanges<>>(test, "SparseIndexRanges");
    runBuilderBehaviorTests<CompressedSparseIndexRanges<>>(test, "CompressedSparseIndexRanges");

    runRandomCompressionCase<SparseIndexRanges<std::uint_least8_t>>(
      test,
      "SparseIndexRanges random compression uint8",
      40,
      10,
      50,
      250,
      0xA1B2C3D4u);

    runRandomCompressionCase<CompressedSparseIndexRanges<>>(
      test,
      "CompressedSparseIndexRanges random compression uint8",
      40,
      10,
      50,
      250,
      0xA1B2C3D4u);

    runRandomCompressionCase<SparseIndexRanges<std::uint_least16_t>>(
      test,
      "SparseIndexRanges random compression uint16",
      40,
      10,
      100,
      static_cast<std::size_t>(std::numeric_limits<std::uint_least8_t>::max()) + 300,
      0xB2C3D4E5u);

    runRandomCompressionCase<CompressedSparseIndexRanges<>>(
      test,
      "CompressedSparseIndexRanges random compression uint16",
      40,
      10,
      100,
      static_cast<std::size_t>(std::numeric_limits<std::uint_least8_t>::max()) + 300,
      0xB2C3D4E5u);

    runRandomCompressionCase<SparseIndexRanges<std::uint_least32_t>>(
      test,
      "SparseIndexRanges random compression uint32",
      40,
      10,
      1000,
      static_cast<std::size_t>(std::numeric_limits<std::uint_least16_t>::max()) + 50000,
      0xC3D4E5F6u);

    runRandomCompressionCase<CompressedSparseIndexRanges<>>(
      test,
      "CompressedSparseIndexRanges random compression uint32",
      40,
      10,
      1000,
      static_cast<std::size_t>(std::numeric_limits<std::uint_least16_t>::max()) + 50000,
      0xC3D4E5F6u);

    runRandomCompressionCase<SparseIndexRanges<std::uint_least64_t>>(
      test,
      "SparseIndexRanges random compression uint64",
      40,
      10,
      1000,
      static_cast<std::size_t>(std::numeric_limits<std::uint_least32_t>::max()) + 1000ULL,
      0xD4E5F607u);

    runRandomCompressionCase<CompressedSparseIndexRanges<>>(
      test,
      "CompressedSparseIndexRanges random compression uint64",
      40,
      10,
      1000,
      static_cast<std::size_t>(std::numeric_limits<std::uint_least32_t>::max()) + 1000ULL,
      0xD4E5F607u);

  return test.exit();
}
