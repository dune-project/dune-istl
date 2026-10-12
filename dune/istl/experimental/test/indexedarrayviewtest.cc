// SPDX-FileCopyrightText: Copyright (c) DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception

#include "../indexedarrayview.hh"

#include <dune/common/test/testsuite.hh>
#include <dune/common/rangeutilities.hh>

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <ranges>
#include <span>
#include <string>
#include <type_traits>

struct DenseIndexRange : public Dune::IntegralRange<int> {
  using IntegralRange::IntegralRange;
};

template<>
constexpr bool std::ranges::enable_borrowed_range<DenseIndexRange> = true;

void testIndexedViewDense(Dune::TestSuite& test)
{
  {
    std::array<double, 8> data{ 0.5, 1.5, 2.5, 3.5, -1.0, -2.0, -3.0, -4.0 };
    auto denseIndices = DenseIndexRange(0, data.size());

    using ArrayView = Dune::Experimental::IndexedArrayView<double*, DenseIndexRange>;
    auto view = ArrayView{data.data(), denseIndices};
    const auto& cview = view;

    test.check(view.size() == data.size(), "dense size matches index count");
    test.check(std::distance(view.begin(), view.end()) == static_cast<std::ptrdiff_t>(view.size()),
               "dense iterator distance equals size");
    test.check(view[0] == 0.5, "dense index 0 maps to first element");
    test.check(view[1] == 1.5, "dense index 1 maps to second element");
    test.check(view[3] == 3.5, "dense index 3 maps to fourth element");
    test.check(cview[0] == 0.5, "dense const operator[] reads first element");

    view[3] = 13.5;
    test.check(data[3] == 13.5, "dense mutable operator[] writes through to backing data");
    test.check(cview[3] == 13.5, "dense const view observes mutation");

    const auto itDense = view.find(2);
    test.check(itDense != view.end(), "dense find(existing) succeeds");
    if (itDense != view.end()) {
      test.check(itDense.index() == 2, "dense find(existing) iterator index is correct");
      test.check(*itDense == data[2], "dense find(existing) points to mapped value");
    }

    const auto itDenseFirst = view.find(0);
    const auto itDenseLast = view.find(static_cast<int>(data.size() - 1));
    test.check(itDenseFirst == view.begin(), "dense find(first) returns begin");
    test.check(itDenseLast != view.end(), "dense find(last) succeeds");
    if (itDenseLast != view.end())
      test.check(itDenseLast.index() == static_cast<int>(data.size() - 1),
                 "dense find(last) returns last index");

    test.check(view.find(99) == view.end(), "dense find(missing) returns end");

    std::size_t k = 0;
    for (auto it = view.begin(); it != view.end(); ++it, ++k)
      test.check(it.index() == denseIndices[k], "dense iterator exposes sorted indices");

    k = 0;
    for (auto it = cview.begin(); it != cview.end(); ++it, ++k)
      test.check(*it == data[k], "dense const iterator yields backing values");
  }
}

void testIndexedViewSparse(Dune::TestSuite& test)
{
  {
    std::array<double, 8> data{ 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0 };
    const std::array<int, 3> sparseIndices{ 1, 3, 7 };

    auto view = Dune::Experimental::IndexedArrayView{data.data(), std::span{sparseIndices}};
    const auto& cview = view;

    test.check(view.size() == sparseIndices.size(), "sparse size matches index count");
    test.check(std::distance(view.begin(), view.end()) == static_cast<std::ptrdiff_t>(view.size()),
               "sparse iterator distance equals size");
    test.check(view[1] == 10.0, "sparse index 1 maps to first stored element");
    test.check(view[3] == 20.0, "sparse index 3 maps to second stored element");
    test.check(view[7] == 30.0, "sparse index 7 maps to third stored element");
    test.check(cview[7] == 30.0, "sparse const operator[] reads mapped element");

    view[3] = -22.0;
    test.check(data[1] == -22.0, "sparse mutable operator[] writes mapped backing element");
    test.check(cview[3] == -22.0, "sparse const view observes mutation");

    test.check(view.find(2) == view.end(), "sparse find(missing gap) returns end");
    test.check(view.find(0) == view.end(), "sparse find(missing below first) returns end");
    test.check(view.find(8) == view.end(), "sparse find(missing above last) returns end");

    const auto itSparse = view.find(7);
    test.check(itSparse != view.end(), "sparse find(existing) succeeds");
    if (itSparse != view.end()) {
      test.check(itSparse.index() == 7, "sparse find(existing) iterator index is correct");
      test.check(*itSparse == data[2], "sparse find(existing) points to mapped value");
    }

    const auto itSparseFirst = view.find(1);
    test.check(itSparseFirst == view.begin(), "sparse find(first present) returns begin");

    std::size_t k = 0;
    for (auto it = view.begin(); it != view.end(); ++it, ++k)
      test.check(it.index() == sparseIndices[k], "sparse iterator follows sparse index order");

    k = 0;
    for (auto it = cview.begin(); it != cview.end(); ++it, ++k)
      test.check(it.index() == sparseIndices[k], "sparse const iterator follows sparse index order");
  }
}

int main()
{
  Dune::TestSuite test("indexed view dense+sparse");
  testIndexedViewDense(test);
  testIndexedViewSparse(test);
  return test.exit();
}
