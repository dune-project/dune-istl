// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ISTL_BVECTOR_HH
#define DUNE_ISTL_BVECTOR_HH

#include <algorithm>
#include <cmath>
#include <complex>
#include <initializer_list>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include <dune/common/dotproduct.hh>
#include <dune/common/ftraits.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/common/promotiontraits.hh>
#include <dune/common/typetraits.hh>
#include <dune/common/scalarvectorview.hh>

#include <dune/istl/blocklevel.hh>

#include "basearray.hh"
#include "istlexception.hh"

/*! \file

   \brief  This file implements a vector space as a tensor product of
   a given vector space. The number of components can be given at
   run-time.
 */

namespace Dune {

/** \brief Everything in this namespace is internal to dune-istl, and may change without warning */
namespace Imp {

  /** \brief Define some derived types transparently for number types and dune-istl vector types
   *
   * This is the actual implementation.  Calling code should use BlockTraits instead.
   * \tparam isNumber Whether B is a number type (true) or a dune-istl matrix or vector type (false)
   */
  template <class B, bool isNumber>
  class BlockTraitsImp;

  template <class B>
  class BlockTraitsImp<B,true>
  {
  public:
    using field_type = B;
  };

  template <class B>
  class BlockTraitsImp<B,false>
  {
  public:
    using field_type = typename B::field_type;
  };

  /** \brief Define some derived types transparently for number types and dune-istl matrix/vector types
   */
  template <class B>
  using BlockTraits = BlockTraitsImp<B,IsNumber<B>::value>;

  /**
      \brief An unmanaged vector of blocks.

      block_vector_unmanaged extends the base_array_unmanaged by
      vector operations such as addition and scalar multiplication.
          No memory management is added.

          Error checking: no error checking is provided normally.
          Setting the compile time switch DUNE_ISTL_WITH_CHECKING
          enables error checking.

   \internal This class is an implementation detail, and should not be used outside of dune-istl.
   */
  template<class B, class A=std::allocator<B> >
  class block_vector_unmanaged : public base_array_unmanaged<B,A>
  {
  public:

    //===== type definitions and constants
    using field_type = typename Imp::BlockTraits<B>::field_type;

    //! export the type representing the components
    typedef B block_type;

    //! export the allocator type
    typedef A allocator_type;

    //! The size type for the index access
    typedef typename A::size_type size_type;

    //! make iterators available as types
    typedef typename base_array_unmanaged<B,A>::iterator Iterator;

    //! make iterators available as types
    typedef typename base_array_unmanaged<B,A>::const_iterator ConstIterator;

    //! for STL compatibility
    typedef B value_type;

    //! Type used for references
    typedef B& reference;

    //! Type used for const references
    typedef const B& const_reference;

    //===== assignment from scalar
    //! Assignment from a scalar

    block_vector_unmanaged& operator= (const field_type& k)
    {
      for (size_type i=0; i<this->n; i++)
        (*this)[i] = k;
      return *this;
    }

    //===== vector space arithmetic
    //! vector space addition
    block_vector_unmanaged& operator+= (const block_vector_unmanaged& y)
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (this->n!=y.N()) DUNE_THROW(ISTLError,"vector size mismatch");
#endif
      for (size_type i=0; i<this->n; ++i) (*this)[i] += y[i];
      return *this;
    }

    //! vector space subtraction
    block_vector_unmanaged& operator-= (const block_vector_unmanaged& y)
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (this->n!=y.N()) DUNE_THROW(ISTLError,"vector size mismatch");
#endif
      for (size_type i=0; i<this->n; ++i) (*this)[i] -= y[i];
      return *this;
    }

    //! vector space multiplication with scalar
    block_vector_unmanaged& operator*= (const field_type& k)
    {
      for (size_type i=0; i<this->n; ++i) (*this)[i] *= k;
      return *this;
    }

    //! vector space division by scalar
    block_vector_unmanaged& operator/= (const field_type& k)
    {
      for (size_type i=0; i<this->n; ++i) (*this)[i] /= k;
      return *this;
    }

    //! vector space axpy operation
    block_vector_unmanaged& axpy (const field_type& a, const block_vector_unmanaged& y)
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (this->n!=y.N()) DUNE_THROW(ISTLError,"vector size mismatch");
#endif
      for (size_type i=0; i<this->n; ++i)
        Impl::asVector((*this)[i]).axpy(a,Impl::asVector(y[i]));

      return *this;
    }


    /**
     * \brief indefinite vector dot product \f$\left (x^T \cdot y \right)\f$ which corresponds to Petsc's VecTDot
     *
     * http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/VecTDot.html
     * @param y other (compatible)  vector
     * @return
     */
    template<class OtherB, class OtherA>
    auto operator* (const block_vector_unmanaged<OtherB,OtherA>& y) const
    {
      typedef typename PromotionTraits<field_type,typename BlockTraits<OtherB>::field_type>::PromotedType PromotedType;
      PromotedType sum(0);
#ifdef DUNE_ISTL_WITH_CHECKING
      if (this->n!=y.N()) DUNE_THROW(ISTLError,"vector size mismatch");
#endif
      for (size_type i=0; i<this->n; ++i) {
        sum += PromotedType(((*this)[i])*y[i]);
      }
      return sum;
    }

    /**
     * @brief vector dot product \f$\left (x^H \cdot y \right)\f$ which corresponds to Petsc's VecDot
     *
     * http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/VecDot.html
     * @param y other (compatible) vector
     * @return
     */
    template<class OtherB, class OtherA>
    auto dot(const block_vector_unmanaged<OtherB,OtherA>& y) const
    {
      typedef typename PromotionTraits<field_type,typename BlockTraits<OtherB>::field_type>::PromotedType PromotedType;
      PromotedType sum(0);
#ifdef DUNE_ISTL_WITH_CHECKING
      if (this->n!=y.N()) DUNE_THROW(ISTLError,"vector size mismatch");
#endif

      for (size_type i=0; i<this->n; ++i)
        sum += Impl::asVector((*this)[i]).dot(Impl::asVector(y[i]));

      return sum;
    }

    //===== norms

    //! one norm (sum over absolute values of entries)
    typename FieldTraits<field_type>::real_type one_norm () const
    {
      typename FieldTraits<field_type>::real_type sum=0;
      for (size_type i=0; i<this->n; ++i)
        sum += Impl::asVector((*this)[i]).one_norm();
      return sum;
    }

    //! simplified one norm (uses Manhattan norm for complex values)
    typename FieldTraits<field_type>::real_type one_norm_real () const
    {
      typename FieldTraits<field_type>::real_type sum=0;
      for (size_type i=0; i<this->n; ++i)
        sum += Impl::asVector((*this)[i]).one_norm_real();
      return sum;
    }

    //! two norm sqrt(sum over squared values of entries)
    typename FieldTraits<field_type>::real_type two_norm () const
    {
      using std::sqrt;
      return sqrt(two_norm2());
    }

    //! Square of the two-norm (the sum over the squared values of the entries)
    typename FieldTraits<field_type>::real_type two_norm2 () const
    {
      typename FieldTraits<field_type>::real_type sum=0;
      for (size_type i=0; i<this->n; ++i)
        sum += Impl::asVector((*this)[i]).two_norm2();
      return sum;
    }

    //! infinity norm (maximum of absolute values of entries)
    template <typename ft = field_type,
              typename std::enable_if<!HasNaN<ft>::value, int>::type = 0>
    typename FieldTraits<ft>::real_type infinity_norm() const {
      using real_type = typename FieldTraits<ft>::real_type;
      using std::max;

      real_type norm = 0;
      for (auto const &xi : *this) {
        real_type const a = Impl::asVector(xi).infinity_norm();
        norm = max(a, norm);
      }
      return norm;
    }

    //! simplified infinity norm (uses Manhattan norm for complex values)
    template <typename ft = field_type,
              typename std::enable_if<!HasNaN<ft>::value, int>::type = 0>
    typename FieldTraits<ft>::real_type infinity_norm_real() const {
      using real_type = typename FieldTraits<ft>::real_type;
      using std::max;

      real_type norm = 0;
      for (auto const &xi : *this) {
        real_type const a = Impl::asVector(xi).infinity_norm_real();
        norm = max(a, norm);
      }
      return norm;
    }

    //! infinity norm (maximum of absolute values of entries)
    template <typename ft = field_type,
              typename std::enable_if<HasNaN<ft>::value, int>::type = 0>
    typename FieldTraits<ft>::real_type infinity_norm() const {
      using real_type = typename FieldTraits<ft>::real_type;
      using std::max;
      using std::abs;

      real_type norm = 0;
      real_type isNaN = 1;

      for (auto const &xi : *this) {
        real_type const a = Impl::asVector(xi).infinity_norm();
        norm = max(a, norm);
        isNaN += a;
      }
      return norm * (isNaN / isNaN);
    }

    //! simplified infinity norm (uses Manhattan norm for complex values)
    template <typename ft = field_type,
              typename std::enable_if<HasNaN<ft>::value, int>::type = 0>
    typename FieldTraits<ft>::real_type infinity_norm_real() const {
      using real_type = typename FieldTraits<ft>::real_type;
      using std::max;

      real_type norm = 0;
      real_type isNaN = 1;

      for (auto const &xi : *this) {
        real_type const a = Impl::asVector(xi).infinity_norm_real();
        norm = max(a, norm);
        isNaN += a;
      }

      return norm * (isNaN / isNaN);
    }

    //===== sizes

    //! number of blocks in the vector (are of size 1 here)
    size_type N () const
    {
      return this->n;
    }

    //! dimension of the vector space
    size_type dim () const
    {
      size_type d=0;

      for (size_type i=0; i<this->n; i++)
        d += Impl::asVector((*this)[i]).dim();

      return d;
    }

  protected:
    //! make constructor protected, so only derived classes can be instantiated
    block_vector_unmanaged () : base_array_unmanaged<B,A>()
    {       }
  };

  //! simple scope guard, execute the provided functor on scope exit
  /**
   * The guard may not be copied or moved.  This avoids executing the cleanup
   * function twice.  The cleanup function should not throw, as it may be
   * called during stack unwinding.
   */
  template<class F>
  class ScopeGuard {
    F cleanupFunc_;
  public:
    ScopeGuard(F cleanupFunc) : cleanupFunc_(std::move(cleanupFunc)) {}
    ScopeGuard(const ScopeGuard &) = delete;
    ScopeGuard(ScopeGuard &&) = delete;
    ScopeGuard &operator=(ScopeGuard) = delete;
    ~ScopeGuard() { cleanupFunc_(); }
  };

  //! create a scope guard
  /**
   * Use like
   * ```c++
   * {
   *   const auto &guard = makeScopeGuard([&]{ doSomething(); });
   *   doSomethingThatMightThrow();
   * }
   * ```
   */
  template<class F>
  ScopeGuard<F> makeScopeGuard(F cleanupFunc)
  {
    return { std::move(cleanupFunc) };
  }

} // end namespace Imp
  /**
     @addtogroup ISTL_SPMV
     @{
   */
  /**
      \brief A vector of blocks with memory management.

      BlockVector adds memory management with ordinary
      copy semantics to the block_vector_unmanaged template.

          Error checking: no error checking is provided normally.
          Setting the compile time switch DUNE_ISTL_WITH_CHECKING
          enables error checking.
   */
  template<class B, class A=std::allocator<B> >
  class BlockVector : public Imp::block_vector_unmanaged<B,A>
  {
  public:

    //===== type definitions and constants

    //! export the type representing the field
    using field_type = typename Imp::BlockTraits<B>::field_type;

    //! export the type representing the components
    typedef B block_type;

    //! export the allocator type
    typedef A allocator_type;

    //! The type for the index access
    typedef typename A::size_type size_type;

    //! increment block level counter
    [[deprecated("Use free function blockLevel(). Will be removed after 2.8.")]]
    static constexpr unsigned int blocklevel = blockLevel<B>()+1;

    //! make iterators available as types
    typedef typename Imp::block_vector_unmanaged<B,A>::Iterator Iterator;

    //! make iterators available as types
    typedef typename Imp::block_vector_unmanaged<B,A>::ConstIterator ConstIterator;

    //===== constructors and such

    //! makes empty vector
    BlockVector ()
    {
      syncBaseArray();
    }

    //! make vector with _n components
    explicit BlockVector (size_type _n) : storage_(_n)
    {
      syncBaseArray();
    }

    /** \brief Construct from a std::initializer_list */
    BlockVector (std::initializer_list<B> const &l) : storage_(l)
    {
      syncBaseArray();
    }


    /** \brief Make vector with _n components but preallocating capacity components

       If _n > capacity then space for _n entries is allocated.
       \note This constructor is somewhat dangerous.  People may be tempted to
       write something like
       \code
       BlockVector<FieldVector<double,1> > my_vector(100,0);
       \endcode
       expecting to obtain a vector of 100 doubles initialized with zero.
       However, the code calls this constructor which tacitly does something else!
     */
    template<typename S>
    BlockVector (size_type _n, S _capacity)
    {
      static_assert(std::numeric_limits<S>::is_integer,
        "capacity must be an unsigned integral type (be aware, that this constructor does not set the default value!)" );
      if((size_type)_capacity > _n)
        storage_.reserve(_capacity);
      storage_.resize(_n);
      syncBaseArray();
    }


    /**
     * @brief Reserve space.
     *
     * Allocate storage for up to `capacity` blocks.  Resizing won't cause
     * reallocation until the size exceeds the `capacity`
     *
     * @param capacity The maximum number of elements the vector
     *        needs to hold.
     */
    void reserve(size_type capacity)
    {
      [[maybe_unused]] const auto &guard =
        Imp::makeScopeGuard([this]{ syncBaseArray(); });
      storage_.reserve(capacity);
    }

    /**
     * @brief Get the capacity of the vector.
     *
     * I. e. the maximum number of elements the vector can hold.
     * @return The capacity of the vector.
     */
    size_type capacity() const
    {
      return storage_.capacity();
    }

    /**
     * @brief Resize the vector.
     *
     * Resize the vector to the given number of blocks.  Blocks below the
     * given size are copied (moved if possible).  Old blocks above the given
     * size are destructed, new blocks above the given size are
     * value-initialized.
     *
     * @param size The new number of blocks of the vector.
     */
    void resize(size_type size)
    {
      [[maybe_unused]] const auto &guard =
        Imp::makeScopeGuard([this]{ syncBaseArray(); });
      storage_.resize(size);
    }

    //! copy constructor
    BlockVector(const BlockVector &a)
      noexcept(noexcept(std::declval<BlockVector>().storage_ = a.storage_))
    {
      storage_ = a.storage_;
      syncBaseArray();
    }

    //! move constructor
    BlockVector(BlockVector &&a)
      noexcept(noexcept(std::declval<BlockVector>().swap(a)))
    {
      swap(a);
    }

    //! assignment
    BlockVector& operator= (const BlockVector& a)
      noexcept(noexcept(std::declval<BlockVector>().storage_ = a.storage_))
    {
      [[maybe_unused]] const auto &guard =
        Imp::makeScopeGuard([this]{ syncBaseArray(); });
      storage_ = a.storage_;
      return *this;
    }

    //! move assignment
    BlockVector& operator= (BlockVector&& a)
      noexcept(noexcept(std::declval<BlockVector>().swap(a)))
    {
      swap(a);
      return *this;
    }

    //! swap operation
    void swap(BlockVector &other)
      noexcept(noexcept(
            std::declval<BlockVector&>().storage_.swap(other.storage_)))
    {
      [[maybe_unused]] const auto &guard = Imp::makeScopeGuard([&]{
          syncBaseArray();
          other.syncBaseArray();
        });
      storage_.swap(other.storage_);
    }

    //! assign from scalar
    BlockVector& operator= (const field_type& k)
    {
      // forward to operator= in base class
      (static_cast<Imp::block_vector_unmanaged<B,A>&>(*this)) = k;
      return *this;
    }

  private:
    void syncBaseArray() noexcept
    {
      this->p = storage_.data();
      this->n = storage_.size();
    }

    std::vector<B, A> storage_;
  };

  /** @} */

  /** @addtogroup DenseMatVec
      @{
   */
  template<class B, class A>
  struct FieldTraits< BlockVector<B, A> >
  {
    typedef typename FieldTraits<B>::field_type field_type;
    typedef typename FieldTraits<B>::real_type real_type;
  };
  /**
      @}
   */

  //! Send BlockVector to an output stream
  template<class K, class A>
  std::ostream& operator<< (std::ostream& s, const BlockVector<K, A>& v)
  {
    typedef typename  BlockVector<K, A>::size_type size_type;

    for (size_type i=0; i<v.size(); i++)
      s << v[i] << std::endl;

    return s;
  }

/** \brief Everything in this namespace is internal to dune-istl, and may change without warning */
namespace Imp {

  /** BlockVectorWindow adds window manipulation functions
          to the block_vector_unmanaged template.

          This class has no memory management. It assumes that the storage
          for the entries of the vector is maintained outside of this class.

          But you can copy objects of this class and of the base class
      with reference semantics.

          Assignment copies the data, if the format is incompatible with
      the argument an exception is thrown in debug mode.

          Error checking: no error checking is provided normally.
          Setting the compile time switch DUNE_ISTL_WITH_CHECKING
          enables error checking.

   \internal This class is an implementation detail, and should not be used outside of dune-istl.
   */
#ifndef DOXYGEN
  template<class B, class A>
#else
  template<class B, class A=std::allocator<B> >
#endif
  class BlockVectorWindow : public Imp::block_vector_unmanaged<B,A>
  {
  public:

    //===== type definitions and constants

    //! export the type representing the field
    using field_type = typename Imp::BlockTraits<B>::field_type;

    //! export the type representing the components
    typedef B block_type;

    //! export the allocator type
    typedef A allocator_type;

    //! The type for the index access
    typedef typename A::size_type size_type;

    //! increment block level counter
    [[deprecated("Use free function blockLevel(). Will be removed after 2.8.")]]
    static constexpr unsigned int blocklevel = blockLevel<B>()+1;

    //! make iterators available as types
    typedef typename Imp::block_vector_unmanaged<B,A>::Iterator Iterator;

    //! make iterators available as types
    typedef typename Imp::block_vector_unmanaged<B,A>::ConstIterator ConstIterator;


    //===== constructors and such
    //! makes empty array
    BlockVectorWindow () : Imp::block_vector_unmanaged<B,A>()
    {       }

    //! make array from given pointer and size
    BlockVectorWindow (B* _p, size_type _n)
    {
      this->n = _n;
      this->p = _p;
    }

    //! copy constructor, this has reference semantics!
    BlockVectorWindow (const BlockVectorWindow& a)
    {
      this->n = a.n;
      this->p = a.p;
    }

    //! assignment
    BlockVectorWindow& operator= (const BlockVectorWindow& a)
    {
      // check correct size
#ifdef DUNE_ISTL_WITH_CHECKING
      if (this->n!=a.N()) DUNE_THROW(ISTLError,"vector size mismatch");
#endif

      if (&a!=this)     // check if this and a are different objects
      {
        // copy data
        for (size_type i=0; i<this->n; i++) this->p[i]=a.p[i];
      }
      return *this;
    }

    //! assign from scalar
    BlockVectorWindow& operator= (const field_type& k)
    {
      (static_cast<Imp::block_vector_unmanaged<B,A>&>(*this)) = k;
      return *this;
    }

    //! copy into an independent BlockVector object
    operator BlockVector<B, A>() const {
      auto bv = BlockVector<B, A>(this->n);

      std::copy(this->begin(), this->end(), bv.begin());

      return bv;
    }

    //===== window manipulation methods

    //! set size and pointer
    void set (size_type _n, B* _p)
    {
      this->n = _n;
      this->p = _p;
    }

    //! set size only
    void setsize (size_type _n)
    {
      this->n = _n;
    }

    //! set pointer only
    void setptr (B* _p)
    {
      this->p = _p;
    }

    //! get pointer
    B* getptr ()
    {
      return this->p;
    }

    //! get size
    size_type getsize () const
    {
      return this->n;
    }
  };



  /** compressed_block_vector_unmanaged extends the compressed base_array_unmanaged by
      vector operations such as addition and scalar multiplication.
          No memory management is added.

          Error checking: no error checking is provided normally.
          Setting the compile time switch DUNE_ISTL_WITH_CHECKING
          enables error checking.

   \internal This class is an implementation detail, and should not be used outside of dune-istl.
   */
  template<class B, class A=std::allocator<B> >
  class compressed_block_vector_unmanaged : public compressed_base_array_unmanaged<B,A>
  {
  public:

    //===== type definitions and constants

    //! export the type representing the field
    using field_type = typename Imp::BlockTraits<B>::field_type;

    //! export the type representing the components
    typedef B block_type;

    //! export the allocator type
    typedef A allocator_type;

    //! make iterators available as types
    typedef typename compressed_base_array_unmanaged<B,A>::iterator Iterator;

    //! make iterators available as types
    typedef typename compressed_base_array_unmanaged<B,A>::const_iterator ConstIterator;

    //! The type for the index access
    typedef typename A::size_type size_type;

    //===== assignment from scalar

    compressed_block_vector_unmanaged& operator= (const field_type& k)
    {
      for (size_type i=0; i<this->n; i++)
        (this->p)[i] = k;
      return *this;
    }


    //===== vector space arithmetic

    //! vector space addition
    template<class V>
    compressed_block_vector_unmanaged& operator+= (const V& y)
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (!includesindexset(y)) DUNE_THROW(ISTLError,"index set mismatch");
#endif
      for (size_type i=0; i<y.n; ++i) this->operator[](y.j[i]) += y.p[i];
      return *this;
    }

    //! vector space subtraction
    template<class V>
    compressed_block_vector_unmanaged& operator-= (const V& y)
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (!includesindexset(y)) DUNE_THROW(ISTLError,"index set mismatch");
#endif
      for (size_type i=0; i<y.n; ++i) this->operator[](y.j[i]) -= y.p[i];
      return *this;
    }

    //! vector space axpy operation
    template<class V>
    compressed_block_vector_unmanaged& axpy (const field_type& a, const V& y)
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (!includesindexset(y)) DUNE_THROW(ISTLError,"index set mismatch");
#endif
      for (size_type i=0; i<y.n; ++i)
        Impl::asVector((*this)[y.j[i]]).axpy(a,Impl::asVector(y.p[i]));
      return *this;
    }

    //! vector space multiplication with scalar
    compressed_block_vector_unmanaged& operator*= (const field_type& k)
    {
      for (size_type i=0; i<this->n; ++i) (this->p)[i] *= k;
      return *this;
    }

    //! vector space division by scalar
    compressed_block_vector_unmanaged& operator/= (const field_type& k)
    {
      for (size_type i=0; i<this->n; ++i) (this->p)[i] /= k;
      return *this;
    }


    //===== Euclidean scalar product

    //! scalar product
    field_type operator* (const compressed_block_vector_unmanaged& y) const
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (!includesindexset(y) || !y.includesindexset(*this) )
        DUNE_THROW(ISTLError,"index set mismatch");
#endif
      field_type sum=0;
      for (size_type i=0; i<this->n; ++i)
        sum += (this->p)[i] * y[(this->j)[i]];
      return sum;
    }


    //===== norms

    //! one norm (sum over absolute values of entries)
    typename FieldTraits<field_type>::real_type one_norm () const
    {
      typename FieldTraits<field_type>::real_type sum=0;
      for (size_type i=0; i<this->n; ++i) sum += (this->p)[i].one_norm();
      return sum;
    }

    //! simplified one norm (uses Manhattan norm for complex values)
    typename FieldTraits<field_type>::real_type one_norm_real () const
    {
      typename FieldTraits<field_type>::real_type sum=0;
      for (size_type i=0; i<this->n; ++i) sum += (this->p)[i].one_norm_real();
      return sum;
    }

    //! two norm sqrt(sum over squared values of entries)
    typename FieldTraits<field_type>::real_type two_norm () const
    {
      using std::sqrt;
      typename FieldTraits<field_type>::real_type sum=0;
      for (size_type i=0; i<this->n; ++i) sum += (this->p)[i].two_norm2();
      return sqrt(sum);
    }

    //! Square of the two-norm (the sum over the squared values of the entries)
    typename FieldTraits<field_type>::real_type two_norm2 () const
    {
      typename FieldTraits<field_type>::real_type sum=0;
      for (size_type i=0; i<this->n; ++i) sum += (this->p)[i].two_norm2();
      return sum;
    }

    //! infinity norm (maximum of absolute values of entries)
    template <typename ft = field_type,
              typename std::enable_if<!HasNaN<ft>::value, int>::type = 0>
    typename FieldTraits<ft>::real_type infinity_norm() const {
      using real_type = typename FieldTraits<ft>::real_type;
      using std::max;

      real_type norm = 0;
      for (auto const &x : *this) {
        real_type const a = x.infinity_norm();
        norm = max(a, norm);
      }
      return norm;
    }

    //! simplified infinity norm (uses Manhattan norm for complex values)
    template <typename ft = field_type,
              typename std::enable_if<!HasNaN<ft>::value, int>::type = 0>
    typename FieldTraits<ft>::real_type infinity_norm_real() const {
      using real_type = typename FieldTraits<ft>::real_type;
      using std::max;

      real_type norm = 0;
      for (auto const &x : *this) {
        real_type const a = x.infinity_norm_real();
        norm = max(a, norm);
      }
      return norm;
    }

    //! infinity norm (maximum of absolute values of entries)
    template <typename ft = field_type,
              typename std::enable_if<HasNaN<ft>::value, int>::type = 0>
    typename FieldTraits<ft>::real_type infinity_norm() const {
      using real_type = typename FieldTraits<ft>::real_type;
      using std::max;

      real_type norm = 0;
      real_type isNaN = 1;
      for (auto const &x : *this) {
        real_type const a = x.infinity_norm();
        norm = max(a, norm);
        isNaN += a;
      }
      return norm * (isNaN / isNaN);
    }

    //! simplified infinity norm (uses Manhattan norm for complex values)
    template <typename ft = field_type,
              typename std::enable_if<HasNaN<ft>::value, int>::type = 0>
    typename FieldTraits<ft>::real_type infinity_norm_real() const {
      using real_type = typename FieldTraits<ft>::real_type;
      using std::max;

      real_type norm = 0;
      real_type isNaN = 1;
      for (auto const &x : *this) {
        real_type const a = x.infinity_norm_real();
        norm = max(a, norm);
        isNaN += a;
      }
      return norm * (isNaN / isNaN);
    }

    //===== sizes

    //! number of blocks in the vector (are of size 1 here)
    size_type N () const
    {
      return this->n;
    }

    //! dimension of the vector space
    size_type dim () const
    {
      size_type d=0;
      for (size_type i=0; i<this->n; i++)
        d += (this->p)[i].dim();
      return d;
    }

  protected:
    //! make constructor protected, so only derived classes can be instantiated
    compressed_block_vector_unmanaged () : compressed_base_array_unmanaged<B,A>()
    {       }

    //! return true if index sets coincide
    template<class V>
    bool includesindexset (const V& y)
    {
      typename V::ConstIterator e=this->end();
      for (size_type i=0; i<y.n; i++)
        if (this->find(y.j[i])==e)
          return false;
      return true;
    }
  };


  /** CompressedBlockVectorWindow adds window manipulation functions
          to the compressed_block_vector_unmanaged template.

          This class has no memory management. It assumes that the storage
          for the entries of the vector and its index set is maintained outside of this class.

          But you can copy objects of this class and of the base class
      with reference semantics.

          Assignment copies the data, if the format is incopmpatible with
      the argument an exception is thrown in debug mode.

          Error checking: no error checking is provided normally.
          Setting the compile time switch DUNE_ISTL_WITH_CHECKING
          enables error checking.

   \internal This class is an implementation detail, and should not be used outside of dune-istl.
   */
  template<class B, class A=std::allocator<B> >
  class CompressedBlockVectorWindow : public compressed_block_vector_unmanaged<B,A>
  {
  public:

    //===== type definitions and constants

    //! export the type representing the field
    using field_type = typename Imp::BlockTraits<B>::field_type;

    //! export the type representing the components
    typedef B block_type;

    //! export the allocator type
    typedef A allocator_type;

    //! The type for the index access
    typedef typename A::size_type size_type;

    //! increment block level counter
    [[deprecated("Use free function blockLevel(). Will be removed after 2.8.")]]
    static constexpr unsigned int blocklevel = blockLevel<B>()+1;

    //! make iterators available as types
    typedef typename compressed_block_vector_unmanaged<B,A>::Iterator Iterator;

    //! make iterators available as types
    typedef typename compressed_block_vector_unmanaged<B,A>::ConstIterator ConstIterator;


    //===== constructors and such
    //! makes empty array
    CompressedBlockVectorWindow () : compressed_block_vector_unmanaged<B,A>()
    {       }

    //! make array from given pointers and size
    CompressedBlockVectorWindow (B* _p, size_type* _j, size_type _n)
    {
      this->n = _n;
      this->p = _p;
      this->j = _j;
    }

    //! copy constructor, this has reference semantics!
    CompressedBlockVectorWindow (const CompressedBlockVectorWindow& a)
    {
      this->n = a.n;
      this->p = a.p;
      this->j = a.j;
    }

    //! assignment
    CompressedBlockVectorWindow& operator= (const CompressedBlockVectorWindow& a)
    {
      // check correct size
#ifdef DUNE_ISTL_WITH_CHECKING
      if (this->n!=a.N()) DUNE_THROW(ISTLError,"vector size mismatch");
#endif

      if (&a!=this)     // check if this and a are different objects
      {
        // copy data
        for (size_type i=0; i<this->n; i++) this->p[i]=a.p[i];
        for (size_type i=0; i<this->n; i++) this->j[i]=a.j[i];
      }
      return *this;
    }

    //! assign from scalar
    CompressedBlockVectorWindow& operator= (const field_type& k)
    {
      (static_cast<compressed_block_vector_unmanaged<B,A>&>(*this)) = k;
      return *this;
    }


    //===== window manipulation methods

    //! set size and pointer
    void set (size_type _n, B* _p, size_type* _j)
    {
      this->n = _n;
      this->p = _p;
      this->j = _j;
    }

    //! set size only
    void setsize (size_type _n)
    {
      this->n = _n;
    }

    //! set pointer only
    void setptr (B* _p)
    {
      this->p = _p;
    }

    //! set pointer only
    void setindexptr (size_type* _j)
    {
      this->j = _j;
    }

    //! get pointer
    B* getptr ()
    {
      return this->p;
    }

    //! get pointer
    size_type* getindexptr ()
    {
      return this->j;
    }

    //! get pointer
    const B* getptr () const
    {
      return this->p;
    }

    //! get pointer
    const size_type* getindexptr () const
    {
      return this->j;
    }
    //! get size
    size_type getsize () const
    {
      return this->n;
    }
  };

} // end namespace 'Imp'


  //! Specialization for the proxies of `BlockVectorWindow`
  template<typename B, typename A>
  struct AutonomousValueType<Imp::BlockVectorWindow<B,A>>
  {
    using type = BlockVector<B, A>;
  };


} // end namespace 'Dune'

#endif
