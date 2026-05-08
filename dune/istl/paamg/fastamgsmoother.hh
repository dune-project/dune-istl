// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_FASTAMGSMOOTHER_HH
#define DUNE_ISTL_FASTAMGSMOOTHER_HH

#include <cstddef>
#include <dune/common/typetraits.hh>

namespace Dune
{
  namespace Amg
  {

    template<std::size_t level>
    struct GaussSeidelPresmoothDefect {

      template<typename M, typename X, typename Y>
      static void apply(const M& A, X& x, Y& d,
                        const Y& b)
      {
        typedef typename M::ConstRowIterator RowIterator;
        typedef typename M::ConstColIterator ColIterator;

        typename Y::iterator dIter=d.begin();
        typename Y::const_iterator bIter=b.begin();
        typename X::iterator xIter=x.begin();

        for(RowIterator row=A.begin(), end=A.end(); row != end;
            ++row, ++dIter, ++xIter, ++bIter)
        {
          ColIterator col=(*row).begin();
          *dIter = *bIter;

          for (; col.index()<row.index(); ++col)
          {
            if constexpr (Dune::IsNumber<std::decay_t<decltype(*col)>>::value)
              *dIter -= (*col)*x[col.index()];
            else
              (*col).mmv(x[col.index()],*dIter);     // rhs -= sum_{j<i} a_ij * xnew_j
          }
          assert(row.index()==col.index());
          ColIterator diag=col;              // upper diagonal matrix not needed as x was 0 before.

          // Not recursive yet. Just solve with the diagonal
          if constexpr (Dune::IsNumber<std::decay_t<decltype(*diag)>>::value)
            *xIter = (*dIter)/(*diag);
          else
            diag->solve(*xIter,*dIter);

          *dIter=0;   //as r=v

          // Update residual for the symmetric case
          for(col=(*row).begin(); col.index()<row.index(); ++col)
          {
            if constexpr (Dune::IsNumber<std::decay_t<decltype(*col)>>::value)
              d[col.index()] -= (*col)*(*xIter);
            else
              col->mmv(*xIter, d[col.index()]);     //d_j-=A_ij x_i
          }
        }
      }
    };

    template<std::size_t level>
    struct GaussSeidelPostsmoothDefect {

      template<typename M, typename X, typename Y>
      static void apply(const M& A, X& x, Y& d,
                        const Y& b)
      {
        using ReverseRowIterator = std::reverse_iterator<typename M::ConstRowIterator>;
        using ReverseColIterator = std::reverse_iterator<typename M::ConstColIterator>;
        typedef typename Y::block_type YBlock;

        std::reverse_iterator<typename Y::iterator> dIter{d.end()};
        std::reverse_iterator<typename X::iterator> xIter{x.end()};
        std::reverse_iterator<typename Y::const_iterator> bIter{b.end()};

        auto rindex = [](auto it) { return std::prev(it.base()).index(); };
        for(ReverseRowIterator row=ReverseRowIterator{A.end()}, end=ReverseRowIterator{A.begin()}; row != end;
            ++row, ++dIter, ++xIter, ++bIter)
        {
          ReverseColIterator rbeginCol{(*row).begin()};
          ReverseColIterator col{(*row).end()};
          *dIter = *bIter;

          for (; rindex(col)>rindex(row); ++col)
          {
            if constexpr (Dune::IsNumber<std::decay_t<decltype(*col)>>::value)
              *dIter -= (*col)*x[rindex(col)];
            else
              (*col).mmv(x[rindex(col)],*dIter);     // rhs -= sum_{j<i} a_ij * xnew_j
          }
          assert(rindex(row)==rindex(col));
          ReverseColIterator diag=col;
          YBlock v=*dIter;
          // upper diagonal matrix
          for (++col; col!=rbeginCol; ++col)
          {
            if constexpr (Dune::IsNumber<std::decay_t<decltype(*col)>>::value)
              v -= (*col)*x[rindex(col)];
            else
              (*col).mmv(x[rindex(col)],v);     // v -= sum_{j<i} a_ij * xold_j
          }

          // Not recursive yet. Just solve with the diagonal
          if constexpr (Dune::IsNumber<std::decay_t<decltype(*diag)>>::value)
            *xIter = v/(*diag);
          else
            diag->solve(*xIter,v);

          *dIter-=v;

          // Update residual for the symmetric case
          // Skip residual computation as it is not needed.
          //for(col=(*row).begin();rindex(col)<rindex(row); ++col)
          //col.mmv(*xIter, d[rindex(col)]); //d_j-=A_ij x_i
        }
      }
    };
  } // end namespace Amg
} // end namespace Dune
#endif
