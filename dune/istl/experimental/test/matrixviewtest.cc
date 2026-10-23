// SPDX-FileCopyrightText: Copyright (c) DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception

#include "../matrixview.hh"
#include "../../bvector.hh"
#include "../../test/matrixtest.hh"

#include <dune/common/test/testsuite.hh>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <initializer_list>
#include <ranges>
#include <span>
#include <vector>

namespace {

struct SparseIndexRange : public std::vector<int> {
  SparseIndexRange() = default;
  SparseIndexRange(std::size_t nCols_, std::initializer_list<int> indices)
    : std::vector<int>(indices)
    , nCols(static_cast<int>(nCols_))
  {}

  Dune::IntegralRange<int> range() const {
    return { 0, nCols };
  }
  int nCols = 0;
};

struct TestIndexRanges : public std::vector<SparseIndexRange> {
  using size_type = std::size_t;

  TestIndexRanges() = default;
  TestIndexRanges(std::size_t nCols, std::initializer_list<SparseIndexRange> rows)
    : std::vector<SparseIndexRange>(rows)
    , nCols_(nCols)
  {
    offsets_.reserve(this->size() + 1);
    offsets_.push_back(0);
    for (const auto& row : *this)
      offsets_.push_back(offsets_.back() + row.size());
  }

  size_type offset(size_type row) const
  {
    return offsets_[row];
  }

  size_type count() const
  {
    return offsets_.empty() ? 0 : offsets_.back();
  }

  Dune::IntegralRange<int> range() const
  {
    return { 0, static_cast<int>(nCols_) };
  }

private:
  size_type nCols_ = 0;
  std::vector<size_type> offsets_;
};


// Pattern of a fully stored nRows x nCols matrix, with every row an IntegralRange.
struct SimpleDenseIndexRanges {
  using size_type = std::size_t;

  SimpleDenseIndexRanges(std::size_t nRows, std::size_t nCols)
    : nRows_(nRows), nCols_(nCols)
    , rows_(nRows, Dune::IntegralRange<int>{ 0, static_cast<int>(nCols) })
  {}

  size_type offset(size_type row) const
  {
    return row * nCols_;
  }

  size_type count() const
  {
    return nRows_ * nCols_;
  }

  Dune::IntegralRange<int> range() const
  {
    return { 0, static_cast<int>(nCols_) };
  }

  size_type size() const
  {
    return nRows_;
  }

  // Every row is the same full range of columns.
  Dune::IntegralRange<int> operator[](size_type) const
  {
    return { 0, static_cast<int>(nCols_) };
  }

  // Random access iterator over rows; dereferencing always yields the full column range.
  auto begin() const
  {
    return rows_.begin();
  }

  auto end() const
  {
    return rows_.end();
  }

private:
  size_type nRows_;
  size_type nCols_;
  std::vector<Dune::IntegralRange<int>> rows_;
};


Dune::BlockVector<std::complex<double>> makeVector(std::initializer_list<std::complex<double>> values)
{
  Dune::BlockVector<std::complex<double>> v(values.size());
  std::size_t i = 0;
  for (const auto& x : values)
    v[i++] = x;
  return v;
}

Dune::BlockVector<double> makeVector(std::initializer_list<double> values)
{
  Dune::BlockVector<double> v(values.size());
  std::size_t i = 0;
  for (const auto& x : values)
    v[i++] = x;
  return v;
}

Dune::BlockVector<double> makeFilledVector(std::size_t n, double value)
{
  Dune::BlockVector<double> v(n);
  for (std::size_t i = 0; i < n; ++i)
    v[i] = value;
  return v;
}

} // namespace

namespace Dune {

template<>
struct Experimental::OrderedIndexRangeTraits<SparseIndexRange>
{
  using index_range_type = SparseIndexRange;
  using index_type = int;
  using size_type = std::size_t;

  static IntegralRange<index_type> range(const SparseIndexRange& index_set)
  {
    return { 0, index_set.nCols };
  }

  static size_type offset(const SparseIndexRange& index_set, const index_type& value)
  {
    const auto lb = std::lower_bound(index_set.begin(), index_set.end(), value);
    return static_cast<size_type>(std::distance(index_set.begin(), lb));
  }
};

} // namespace Dune

namespace std::ranges {

template<>
constexpr bool enable_borrowed_range<SparseIndexRange> = true;

} // namespace std::ranges

namespace {

void testBasicAccessAndSize(Dune::TestSuite& test)
{
  std::array<double, 3> data{ 10.0, 20.0, 30.0 };
  const TestIndexRanges isets{ 3, {
    SparseIndexRange{ 3, { 0, 2 } },
    SparseIndexRange{ 3, { 1 } }
  } };

  Dune::Experimental::MatrixView<double*, TestIndexRanges> m(data.data(), &isets);

  test.check(m.N() == 2, "MatrixView N returns row count");
  test.check(m.M() == 3, "MatrixView M returns logical column count");
  test.check(m.nonzeroes() == 3, "MatrixView nonzeroes returns stored block count");

  test.check(m[0][0] == 10.0, "row 0 col 0 maps to first stored entry");
  test.check(m[0][2] == 20.0, "row 0 col 2 maps to second stored entry");
  test.check(m[1][1] == 30.0, "row 1 col 1 maps to third stored entry");

  test.check(m.exists(0, 0), "exists returns true for present row/column");
  test.check(!m.exists(0, 1), "exists returns false for absent row/column");

  auto it = m.begin();
  test.check((*it)[0] == 10.0, "begin dereferences first row");
  ++it;
  test.check((*it)[1] == 30.0, "iterator increment reaches second row");
}

void testArithmeticAndNorms(Dune::TestSuite& test)
{
  std::array<double, 3> dataA{ 1.0, 2.0, 3.0 };
  std::array<double, 3> dataB{ 4.0, 5.0, 6.0 };

  const TestIndexRanges isets{ 3, {
    SparseIndexRange{ 3, { 0, 2 } },
    SparseIndexRange{ 3, { 1 } }
  } };

  Dune::Experimental::MatrixView<double*, TestIndexRanges> a(dataA.data(), &isets);
  Dune::Experimental::MatrixView<double*, TestIndexRanges> b(dataB.data(), &isets);

  a += b;
  test.check(dataA[0] == 5.0 && dataA[1] == 7.0 && dataA[2] == 9.0,
             "operator+= accumulates corresponding stored entries");

  a -= b;
  test.check(dataA[0] == 1.0 && dataA[1] == 2.0 && dataA[2] == 3.0,
             "operator-= restores original stored entries");

  a.axpy(2.0, b);
  test.check(dataA[0] == 9.0 && dataA[1] == 12.0 && dataA[2] == 15.0,
             "axpy updates stored entries with scaled addend");

  // reset for norm checks
  dataA = { 1.0, 2.0, 3.0 };
  test.check(a.frobenius_norm2() == 14.0, "frobenius_norm2 sums squared entries");
  test.check(a.frobenius_norm() == std::sqrt(14.0), "frobenius_norm is sqrt(frobenius_norm2)");
  test.check(a.infinity_norm() == 3.0, "infinity_norm is max row sum for scalar blocks");
  test.check(a.infinity_norm_real() == 3.0, "infinity_norm_real matches infinity_norm for real values");

  a *= 2.0;
  test.check(dataA[0] == 2.0 && dataA[1] == 4.0 && dataA[2] == 6.0,
             "operator*= scales all stored entries");

  a /= 2.0;
  test.check(dataA[0] == 1.0 && dataA[1] == 2.0 && dataA[2] == 3.0,
             "operator/= restores all stored entries");
}

void testMvFamily(Dune::TestSuite& test)
{
  // Matrix structure:
  // row 0: col 0 -> 2, col 2 -> 3
  // row 1: col 1 -> 4
  std::array<double, 3> data{ 2.0, 3.0, 4.0 };
  const TestIndexRanges isets{ 3, {
    SparseIndexRange{ 3, { 0, 2 } },
    SparseIndexRange{ 3, { 1 } }
  } };

  Dune::Experimental::MatrixView<double*, TestIndexRanges> a(data.data(), &isets);

  auto x = makeVector({ 10.0, 20.0, 30.0 });
  auto y = makeFilledVector(2, 0.0);

  a.mv(x, y);
  test.check(y[0] == 110.0 && y[1] == 80.0,
             "mv computes y = A*x for sparse row layout");

  auto y2 = makeVector({ 1.0, 1.0 });
  a.umv(x, y2);
  test.check(y2[0] == 111.0 && y2[1] == 81.0,
             "umv accumulates A*x into y");

  auto y3 = makeVector({ 120.0, 90.0 });
  a.mmv(x, y3);
  test.check(y3[0] == 10.0 && y3[1] == 10.0,
             "mmv subtracts A*x from y");

  auto y4 = makeVector({ 0.0, 0.0 });
  a.usmv(0.5, x, y4);
  test.check(y4[0] == 55.0 && y4[1] == 40.0,
             "usmv accumulates alpha*A*x into y");
}

void testMismatchThrows(Dune::TestSuite& test)
{
  std::array<double, 2> dataSmall{ 1.0, 2.0 };
  std::array<double, 2> dataOther{ 3.0, 4.0 };

  const TestIndexRanges aIs{ 3, {
    SparseIndexRange{ 3, { 0 } },
    SparseIndexRange{ 3, { 1 } }
  } };

  const TestIndexRanges bIs{ 4, {
    SparseIndexRange{ 4, { 0 } },
    SparseIndexRange{ 4, { 2 } }
  } };

  Dune::Experimental::MatrixView<double*, TestIndexRanges> a(dataSmall.data(), &aIs);
  Dune::Experimental::MatrixView<double*, TestIndexRanges> b(dataOther.data(), &bIs);

  const auto expectThrow = [&test](const auto& operation, const char* message) {
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

  expectThrow([&]() { a += b; }, "operator+= throws for mismatching matrix sizes");
  expectThrow([&]() { a -= b; }, "operator-= throws for mismatching matrix sizes");
  expectThrow([&]() { a.axpy(1.0, b); }, "axpy throws for mismatching matrix sizes");

  auto xBad = makeFilledVector(2, 1.0);
  auto yOk = makeFilledVector(2, 0.0);
  expectThrow([&]() { a.mv(xBad, yOk); }, "mv throws for wrong x dimension");
}

void testArithmethicSubSets(Dune::TestSuite& test)
{
  std::array<double, 3> dataA{ 1.0, 2.0, 3.0 };
  std::array<double, 3> dataB{ 4.0, 6.0 };

  // Two distinct TestIndexRanges objects with subset patterns
  const TestIndexRanges isetsA{ 3, {
    SparseIndexRange{ 3, { 0, 2 } },
    SparseIndexRange{ 3, { 1 } }
  } };
  const TestIndexRanges isetsB{ 3, {
    SparseIndexRange{ 3, { 0 } },
    SparseIndexRange{ 3, { 1 } }
  } };

  Dune::Experimental::MatrixView<double*, TestIndexRanges> a(dataA.data(), &isetsA);
  Dune::Experimental::MatrixView<double*, TestIndexRanges> b(dataB.data(), &isetsB);

  a += b;
  test.check(dataA[0] == 5.0 && dataA[1] == 2.0 && dataA[2] == 9.0,
             "operator+= works for equivalent matrices backed by distinct pattern objects");

  a -= b;
  test.check(dataA[0] == 1.0 && dataA[1] == 2.0 && dataA[2] == 3.0,
             "operator-= restores values for equivalent matrices backed by distinct pattern objects");
}

void testDuneMatrixInterface(Dune::TestSuite& test)
{
  using Matrix = Dune::Experimental::MatrixView<double*, TestIndexRanges>;

  std::array<double, 3> data{ 2.0, 3.0, 4.0 };
  const TestIndexRanges isets{ 3, {
    SparseIndexRange{ 3, { 0, 2 } },
    SparseIndexRange{ 3, { 1 } }
  } };

  Matrix a(data.data(), &isets);
  auto domain = makeVector({ 10.0, 20.0, 30.0 }); // size M = 3
  auto range = makeVector({ 0.0, 0.0 });          // size N = 2

  bool ok = true;
  try {
    Dune::testMatrixView(a, domain, range);
  } catch (...) {
    ok = false;
  }

  test.check(ok, "DUNE generic matrix test accepts MatrixView");
}

void testDuneMatrixInterfaceComplex(Dune::TestSuite& test)
{
  using Matrix = Dune::Experimental::MatrixView<std::complex<double>*, TestIndexRanges>;

  std::array<std::complex<double>, 3> data{{ {1.0, 1.0}, {2.0, -1.0}, {3.0, 0.5} }};
  const TestIndexRanges isets{ 3, {
    SparseIndexRange{ 3, { 0, 2 } },
    SparseIndexRange{ 3, { 1 } }
  } };

  Matrix a(data.data(), &isets);
  auto domain = makeVector({ std::complex<double>(10.0, 1.0), std::complex<double>(20.0, -1.0), std::complex<double>(30.0, 0.5) });
  auto range = makeVector({ std::complex<double>(0.0, 0.0), std::complex<double>(0.0, 0.0) });

  bool ok = true;
  try {
    Dune::testMatrixView(a, domain, range);
  } catch (...) {
    ok = false;
  }

  test.check(ok, "DUNE generic matrix test accepts MatrixView with complex values");
}

void testDenseRanges(Dune::TestSuite& test)
{
  using Matrix = Dune::Experimental::MatrixView<double*, SimpleDenseIndexRanges>;

  // Matrix structure: [1 2 3; 4 5 6]
  std::array<double, 6> data{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
  const SimpleDenseIndexRanges isets(2, 3);

  Matrix a(data.data(), &isets);

  test.check(a.N() == 2, "dense MatrixView N returns row count");
  test.check(a.M() == 3, "dense MatrixView M returns column count");
  test.check(a.nonzeroes() == 6, "dense MatrixView nonzeroes counts all entries");

  test.check(a[0][0] == 1.0 && a[0][2] == 3.0 && a[1][0] == 4.0 && a[1][2] == 6.0,
             "dense MatrixView maps row/column to row-major storage");
  test.check(a.exists(0, 0) && a.exists(1, 2), "exists returns true for in-range entries");

  auto x = makeVector({ 1.0, 10.0, 100.0 });
  auto y = makeFilledVector(2, 0.0);
  a.mv(x, y);
  test.check(y[0] == 321.0 && y[1] == 654.0, "mv computes y = A*x for dense rows");

  a.umv(x, y);
  test.check(y[0] == 642.0 && y[1] == 1308.0, "umv accumulates A*x for dense rows");

  a.mmv(x, y);
  test.check(y[0] == 321.0 && y[1] == 654.0, "mmv subtracts A*x for dense rows");

  std::array<double, 6> other{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
  Matrix b(other.data(), &isets);
  a += b;
  test.check(a[0][0] == 2.0 && a[1][2] == 7.0, "operator+= works for dense rows");
  a -= b;
  test.check(a[0][0] == 1.0 && a[1][2] == 6.0, "operator-= works for dense rows");

  test.check(a.frobenius_norm2() == 91.0, "frobenius_norm2 sums squared dense entries");
  test.check(a.infinity_norm() == 6.0, "infinity_norm is max row sum for dense rows");

  auto domain = makeVector({ 10.0, 20.0, 30.0 });
  auto range = makeVector({ 0.0, 0.0 });
  bool ok = true;
  try {
    Dune::testMatrixView(a, domain, range);
  } catch (...) {
    ok = false;
  }
  test.check(ok, "DUNE generic matrix test accepts MatrixView with dense ranges");
}

} // namespace

int main()
{
  Dune::TestSuite test("matrix view");
  testBasicAccessAndSize(test);
  testArithmeticAndNorms(test);
  testMvFamily(test);
  testMismatchThrows(test);
  testArithmethicSubSets(test);
  testDuneMatrixInterface(test);
  testDuneMatrixInterfaceComplex(test);
  testDenseRanges(test);
  return test.exit();
}
