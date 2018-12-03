// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_BTDMATRIX_HH
#define DUNE_ISTL_BTDMATRIX_HH

#include <dune/common/fmatrix.hh>
#include <dune/istl/bcrsmatrix.hh>

/** \file
    \author Oliver Sander
    \brief Implementation of the BTDMatrix class
 */

namespace Dune {
  /**
   * @addtogroup ISTL_SPMV
   * @{
   */
  /** \brief A block-tridiagonal matrix

     \todo It would be safer and more efficient to have a real implementation of
     a block-tridiagonal matrix and not just subclassing from BCRSMatrix.  But that's
     quite a lot of work for that little advantage.*/
  template <class B, class A=std::allocator<B> >
  class BTDMatrix : public BCRSMatrix<B,A>
  {
  public:

    //===== type definitions and constants

    //! export the type representing the field
    using field_type = typename Imp::BlockTraits<B>::field_type;

    //! export the type representing the components
    typedef B block_type;

    //! export the allocator type
    typedef A allocator_type;

    //! implement row_type with compressed vector
    //typedef BCRSMatrix<B,A>::row_type row_type;

    //! The type for the index access and the size
    typedef typename A::size_type size_type;

    //! increment block level counter
    static constexpr unsigned int blocklevel = Imp::BlockTraits<B>::blockLevel()+1;

    /** \brief Default constructor */
    BTDMatrix() : BCRSMatrix<B,A>() {}

    explicit BTDMatrix(size_type size)
      : BCRSMatrix<B,A>(size, size, BCRSMatrix<B,A>::random)
    {
      // Set number of entries for each row
      // All rows get three entries, except for the first and the last one
      for (size_t i=0; i<size; i++)
        this->BCRSMatrix<B,A>::setrowsize(i, 3 - (i==0) - (i==(size-1)));

      this->BCRSMatrix<B,A>::endrowsizes();

      // The actual entries for each row
      for (size_t i=0; i<size; i++) {
        if (i>0)
          this->BCRSMatrix<B,A>::addindex(i, i-1);
        this->BCRSMatrix<B,A>::addindex(i, i  );
        if (i<size-1)
          this->BCRSMatrix<B,A>::addindex(i, i+1);
      }

      this->BCRSMatrix<B,A>::endindices();
    }

    /** \brief Resize the matrix.  Invalidates the content! */
    void setSize(size_type size)
    {
      auto nonZeros = (size==0) ? 0 : size + 2*(size-1);
      this->BCRSMatrix<B,A>::setSize(size,   // rows
                                     size,   // columns
                                     nonZeros);

      // Set number of entries for each row
      // All rows get three entries, except for the first and the last one
      for (size_t i=0; i<size; i++)
        this->BCRSMatrix<B,A>::setrowsize(i, 3 - (i==0) - (i==(size-1)));

      this->BCRSMatrix<B,A>::endrowsizes();

      // The actual entries for each row
      for (size_t i=0; i<size; i++) {
        if (i>0)
          this->BCRSMatrix<B,A>::addindex(i, i-1);
        this->BCRSMatrix<B,A>::addindex(i, i  );
        if (i<size-1)
          this->BCRSMatrix<B,A>::addindex(i, i+1);
      }

      this->BCRSMatrix<B,A>::endindices();
    }

    //! assignment
    BTDMatrix& operator= (const BTDMatrix& other) {
      this->BCRSMatrix<B,A>::operator=(other);
      return *this;
    }

    //! assignment from scalar
    BTDMatrix& operator= (const field_type& k) {
      this->BCRSMatrix<B,A>::operator=(k);
      return *this;
    }

    /** \brief Use the Thomas algorithm to solve the system Ax=b in O(n) time
     *
     * \exception ISTLError if the matrix is singular
     *
     */
    template <class V>
    void solve (V& x, const V& rhs) const {

      // special handling for 1x1 matrices.  The generic algorithm doesn't work for them
      if (this->N()==1) {
        Hybrid::ifElse(IsNumber<B>(),
          [&](auto id) {
            x[0] = id(rhs[0]) / id((*this)[0][0]);
          },
          [&](auto id) {
            id(*this)[0][0].solve(x[0],rhs[0]);
          });
        return;
      }

      // Make copies of the rhs and the right matrix band
      V d = rhs;
      std::vector<block_type> c(this->N()-1);
      for (size_t i=0; i<this->N()-1; i++)
        c[i] = (*this)[i][i+1];

      /* Modify the coefficients. */
      block_type a_00_inv = (*this)[0][0];
      Hybrid::ifElse(IsNumber<B>(),
        [&](auto id) {
          a_00_inv = 1.0 / id(a_00_inv);
        },
        [&](auto id) {
          id(a_00_inv).invert();
        });

      //c[0] /= (*this)[0][0];	/* Division by zero risk. */
      block_type c_0_tmp = c[0];
      Hybrid::ifElse(IsNumber<B>(), /* Division by zero risk. */
        [&](auto id) {
          c[0] = a_00_inv * id(c_0_tmp);
        },
        [&](auto id) {
          FMatrixHelp::multMatrix(id(a_00_inv), id(c_0_tmp), id(c[0]));
        });

      // d = a^{-1} d        /* Division by zero would imply a singular matrix. */
      Hybrid::ifElse(IsNumber<B>(),
        [&](auto id) {
          d[0] *= id(a_00_inv);
        },
        [&](auto id) {
          auto d_0_tmp = d[0];
          id(a_00_inv).mv(d_0_tmp,d[0]);
        });

      for (unsigned int i = 1; i < this->N(); i++) {

        // id = ( a_ii - c_{i-1} a_{i, i-1} ) ^{-1}
        block_type tmp;
        Hybrid::ifElse(IsNumber<B>(),
          [&](auto metaId) {
            tmp = metaId((*this)[i][i-1]) * metaId(c[i-1]);
          },
          [&](auto metaId) {
            FMatrixHelp::multMatrix(metaId((*this)[i][i-1]), metaId(c[i-1]), metaId(tmp));
          });

        block_type id = (*this)[i][i];
        id -= tmp;
        Hybrid::ifElse(IsNumber<B>(), /* Division by zero risk. */
          [&](auto metaId) {
            id = 1.0 / metaId(id);
          },
          [&](auto metaId) {
            metaId(id).invert();
          });

        if (i<c.size()) {
          // c[i] *= id
          Hybrid::ifElse(IsNumber<B>(),            /* Last value calculated is redundant. */
            [&](auto metaId) {
              c[i] *= metaId(id);
            },
            [&](auto metaId) {
              tmp = c[i];
              FMatrixHelp::multMatrix(metaId(id), metaId(tmp), metaId(c[i]));
            });
        }

        // d[i] = (d[i] - d[i-1] * (*this)[i][i-1]) * id;
        Hybrid::ifElse(IsNumber<B>(),
          [&](auto metaId) {
            d[i] -= (*this)[i][i-1] * metaId(d[i-1]);
            d[i] *= metaId(id);
          },
          [&](auto metaId) {
            metaId((*this)[i][i-1]).mmv(d[i-1], d[i]);
            auto tmpVec = d[i];
            metaId(id).mv(tmpVec, d[i]);
          });
      }

      /* Now back substitute. */
      x[this->N() - 1] = d[this->N() - 1];
      Hybrid::ifElse(IsNumber<B>(),
        [&](auto metaId) {
          for (int i = this->N() - 2; i >= 0; i--)
            x[i] = d[i] - c[i] * metaId(x[i+1]);
        },
        [&](auto metaId) {
          for (int i = this->N() - 2; i >= 0; i--) {
            //x[i] = d[i] - c[i] * x[i + 1];
            x[i] = d[i];
            metaId(c[i]).mmv(x[i+1], x[i]);
          }
        });

    }

  private:

    // ////////////////////////////////////////////////////////////////////////////
    //   The following methods from the base class should now actually be called
    // ////////////////////////////////////////////////////////////////////////////

    // createbegin and createend should be in there, too, but I can't get it to compile
    //     BCRSMatrix<B,A>::CreateIterator createbegin () {}
    //     BCRSMatrix<B,A>::CreateIterator createend () {}
    void setrowsize (size_type i, size_type s) {}
    void incrementrowsize (size_type i) {}
    void endrowsizes () {}
    void addindex (size_type row, size_type col) {}
    void endindices () {}
  };
  /** @}*/

}  // end namespace Dune

#endif
