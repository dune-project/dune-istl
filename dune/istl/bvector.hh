// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_ISTL_BVECTOR_HH
#define DUNE_ISTL_BVECTOR_HH

#include <cmath>
#include <complex>
#include <memory>
#include <limits>

#include <dune/common/promotiontraits.hh>
#include <dune/common/dotproduct.hh>
#include <dune/common/ftraits.hh>

#include "istlexception.hh"
#include "basearray.hh"

/*! \file

   \brief  This file implements a vector space as a tensor product of
   a given vector space. The number of components can be given at
   run-time.
 */

namespace Dune {

  template<class B, class A=std::allocator<B> >
  class BlockVectorWindow;

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

    //! export the type representing the field
    typedef typename B::field_type field_type;

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
      for (size_type i=0; i<this->n; ++i) (*this)[i].axpy(a,y[i]);
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
    typename PromotionTraits<field_type,typename OtherB::field_type>::PromotedType operator* (const block_vector_unmanaged<OtherB,OtherA>& y) const
    {
      typedef typename PromotionTraits<field_type,typename OtherB::field_type>::PromotedType PromotedType;
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
    typename PromotionTraits<field_type,typename OtherB::field_type>::PromotedType dot(const block_vector_unmanaged<OtherB,OtherA>& y) const
    {
      typedef typename PromotionTraits<field_type,typename OtherB::field_type>::PromotedType PromotedType;
      PromotedType sum(0);
#ifdef DUNE_ISTL_WITH_CHECKING
      if (this->n!=y.N()) DUNE_THROW(ISTLError,"vector size mismatch");
#endif
      for (size_type i=0; i<this->n; ++i) sum += ((*this)[i]).dot(y[i]);
      return sum;
    }

    //===== norms

    //! one norm (sum over absolute values of entries)
    typename FieldTraits<field_type>::real_type one_norm () const
    {
      typename FieldTraits<field_type>::real_type sum=0;
      for (size_type i=0; i<this->n; ++i) sum += (*this)[i].one_norm();
      return sum;
    }

    //! simplified one norm (uses Manhattan norm for complex values)
    typename FieldTraits<field_type>::real_type one_norm_real () const
    {
      typename FieldTraits<field_type>::real_type sum=0;
      for (size_type i=0; i<this->n; ++i) sum += (*this)[i].one_norm_real();
      return sum;
    }

    //! two norm sqrt(sum over squared values of entries)
    typename FieldTraits<field_type>::real_type two_norm () const
    {
      typename FieldTraits<field_type>::real_type sum=0;
      for (size_type i=0; i<this->n; ++i) sum += (*this)[i].two_norm2();
      return sqrt(sum);
    }

    //! Square of the two-norm (the sum over the squared values of the entries)
    typename FieldTraits<field_type>::real_type two_norm2 () const
    {
      typename FieldTraits<field_type>::real_type sum=0;
      for (size_type i=0; i<this->n; ++i) sum += (*this)[i].two_norm2();
      return sum;
    }

    //! infinity norm (maximum of absolute values of entries)
    template <typename ft = field_type,
              typename std::enable_if<!has_nan<ft>::value, int>::type = 0>
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
              typename std::enable_if<!has_nan<ft>::value, int>::type = 0>
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
              typename std::enable_if<has_nan<ft>::value, int>::type = 0>
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
      isNaN /= isNaN;
      return norm * isNaN;
    }

    //! simplified infinity norm (uses Manhattan norm for complex values)
    template <typename ft = field_type,
              typename std::enable_if<has_nan<ft>::value, int>::type = 0>
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
      isNaN /= isNaN;
      return norm * isNaN;
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
        d += (*this)[i].dim();
      return d;
    }

  protected:
    //! make constructor protected, so only derived classes can be instantiated
    block_vector_unmanaged () : base_array_unmanaged<B,A>()
    {       }
  };

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
  class BlockVector : public block_vector_unmanaged<B,A>
  {
  public:

    //===== type definitions and constants

    //! export the type representing the field
    typedef typename B::field_type field_type;

    //! export the type representing the components
    typedef B block_type;

    //! export the allocator type
    typedef A allocator_type;

    //! The type for the index access
    typedef typename A::size_type size_type;

    //! increment block level counter
    enum {
      //! The number of blocklevel we contain.
      blocklevel = B::blocklevel+1
    };

    //! make iterators available as types
    typedef typename block_vector_unmanaged<B,A>::Iterator Iterator;

    //! make iterators available as types
    typedef typename block_vector_unmanaged<B,A>::ConstIterator ConstIterator;

    //===== constructors and such

    //! makes empty vector
    BlockVector () : block_vector_unmanaged<B,A>(),
                     capacity_(0)
    {}

    //! make vector with _n components
    explicit BlockVector (size_type _n)
    {
      this->n = _n;
      capacity_ = _n;
      if (capacity_>0) {
        this->p = this->allocator_.allocate(capacity_);
        // actually construct the objects
        new(this->p)B[capacity_];
      } else
      {
        this->p = 0;
        this->n = 0;
        capacity_ = 0;
      }
    }

    /** \brief Construct from a std::initializer_list */
    BlockVector (std::initializer_list<B> const &l)
    {
      this->n = l.size();
      capacity_ = l.size();
      if (capacity_>0) {
        this->p = this->allocator_.allocate(capacity_);
        // actually construct the objects
        new(this->p)B[capacity_];

        std::copy_n(l.begin(), l.size(), this->p);
      } else
      {
        this->p = 0;
        this->n = 0;
        capacity_ = 0;
      }
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
      size_type capacity = _capacity;
      this->n = _n;
      if(this->n > capacity)
        capacity_ = _n;
      else
        capacity_ = capacity;

      if (capacity_>0) {
        this->p = this->allocator_.allocate(capacity_);
        new (this->p)B[capacity_];
      } else
      {
        this->p = 0;
        this->n = 0;
        capacity_ = 0;
      }
    }


    /**
     * @brief Reserve space.
     *
     * After calling this method the vector can hold up to
     * capacity values. If the specified capacity is smaller
     * than the current capacity and bigger than the current size
     * space will be freed.
     *
     * If the template parameter copyOldValues is true the values will
     * be copied. If it is false the old values are lost.
     *
     * @param capacity The maximum number of elements the vector
     * needs to hold.
     * @param copyOldValues If false no object will be copied and the data might be
     * lost. Default value is true.
     */
    void reserve(size_type capacity, bool copyOldValues=true)
    {
      if(capacity >= block_vector_unmanaged<B,A>::N() && capacity != capacity_) {
        // save the old data
        B* pold = this->p;

        if(capacity>0) {
          // create new array with capacity
          this->p = this->allocator_.allocate(capacity);
          new (this->p)B[capacity];

          if(copyOldValues) {
            // copy the old values
            B* to = this->p;
            B* from = pold;

            for(size_type i=0; i < block_vector_unmanaged<B,A>::N(); ++i, ++from, ++to)
              *to = *from;
          }
          if(capacity_ > 0) {
            // Destruct old objects and free memory
            int i=capacity_;
            while (i)
              pold[--i].~B();
            this->allocator_.deallocate(pold,capacity_);
          }
        }else{
          if(capacity_ > 0)
            // free old data
            this->p = 0;
          capacity_ = 0;
        }

        capacity_ = capacity;
      }
    }

    /**
     * @brief Get the capacity of the vector.
     *
     * I. e. the maximum number of elements the vector can hold.
     * @return The capacity of the vector.
     */
    size_type capacity() const
    {
      return capacity_;
    }

    /**
     * @brief Resize the vector.
     *
     * After calling this method BlockVector::N() will return size
     * If the capacity of the vector is smaller than the specified
     * size then reserve(size) will be called.
     *
     * If the template parameter copyOldValues is true the values
     * will be copied if the capacity changes.  If it is false
     * the old values are lost.
     * @param size The new size of the vector.
     * @param copyOldValues If false no object will be copied and the data might be
     * lost.
     */
    void resize(size_type size, bool copyOldValues=true)
    {
      if(size > block_vector_unmanaged<B,A>::N())
        if(capacity_ < size)
          this->reserve(size, copyOldValues);
      this->n = size;
    }




    //! copy constructor
    BlockVector (const BlockVector& a) :
      block_vector_unmanaged<B,A>(a)
    {
      // allocate memory with same size as a
      this->n = a.n;
      capacity_ = a.capacity_;

      if (capacity_>0) {
        this->p = this->allocator_.allocate(capacity_);
        new (this->p)B[capacity_];
      } else
      {
        this->n = 0;
        this->p = 0;
      }

      // and copy elements
      for (size_type i=0; i<this->n; i++) this->p[i]=a.p[i];
    }

    //! free dynamic memory
    ~BlockVector ()
    {
      if (capacity_>0) {
        int i=capacity_;
        while (i)
          this->p[--i].~B();
        this->allocator_.deallocate(this->p,capacity_);
      }
    }

    //! assignment
    BlockVector& operator= (const BlockVector& a)
    {
      if (&a!=this)     // check if this and a are different objects
      {
        // adjust size of vector
        if (capacity_!=a.capacity_)           // check if size is different
        {
          if (capacity_>0) {
            int i=capacity_;
            while (i)
              this->p[--i].~B();
            this->allocator_.deallocate(this->p,capacity_);                     // free old memory
          }
          capacity_ = a.capacity_;
          if (capacity_>0) {
            this->p = this->allocator_.allocate(capacity_);
            new (this->p)B[capacity_];
          } else
          {
            this->p = 0;
            capacity_ = 0;
          }
        }
        this->n = a.n;
        // copy data
        for (size_type i=0; i<this->n; i++)
          this->p[i]=a.p[i];
      }
      return *this;
    }

    //! assign from scalar
    BlockVector& operator= (const field_type& k)
    {
      // forward to operator= in base class
      (static_cast<block_vector_unmanaged<B,A>&>(*this)) = k;
      return *this;
    }

    //! Assignment from BlockVectorWindow
    template<class OtherAlloc>
    BlockVector& operator= (const BlockVectorWindow<B,OtherAlloc>& other)
    {
      resize(other.size());
      for(std::size_t i=0; i<other.size(); ++i)
        (*this)[i] = other[i];
      return *this;
    }

  protected:
    size_type capacity_;

    A allocator_;

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
  class BlockVectorWindow : public block_vector_unmanaged<B,A>
  {
  public:

    //===== type definitions and constants

    //! export the type representing the field
    typedef typename B::field_type field_type;

    //! export the type representing the components
    typedef B block_type;

    //! export the allocator type
    typedef A allocator_type;

    //! The type for the index access
    typedef typename A::size_type size_type;

    //! increment block level counter
    enum {
      //! The number of blocklevels we contain
      blocklevel = B::blocklevel+1
    };

    //! make iterators available as types
    typedef typename block_vector_unmanaged<B,A>::Iterator Iterator;

    //! make iterators available as types
    typedef typename block_vector_unmanaged<B,A>::ConstIterator ConstIterator;


    //===== constructors and such
    //! makes empty array
    BlockVectorWindow () : block_vector_unmanaged<B,A>()
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
      (static_cast<block_vector_unmanaged<B,A>&>(*this)) = k;
      return *this;
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
    size_type getsize ()
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
    typedef typename B::field_type field_type;

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
      for (size_type i=0; i<y.n; ++i) (this->operator[](y.j[i])).axpy(a,y.p[i]);
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
              typename std::enable_if<!has_nan<ft>::value, int>::type = 0>
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
              typename std::enable_if<!has_nan<ft>::value, int>::type = 0>
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
              typename std::enable_if<has_nan<ft>::value, int>::type = 0>
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
      isNaN /= isNaN;
      return norm * isNaN;
    }

    //! simplified infinity norm (uses Manhattan norm for complex values)
    template <typename ft = field_type,
              typename std::enable_if<has_nan<ft>::value, int>::type = 0>
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
      isNaN /= isNaN;
      return norm * isNaN;
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
    typedef typename B::field_type field_type;

    //! export the type representing the components
    typedef B block_type;

    //! export the allocator type
    typedef A allocator_type;

    //! The type for the index access
    typedef typename A::size_type size_type;

    //! increment block level counter
    enum {
      //! The number of block level this vector contains.
      blocklevel = B::blocklevel+1
    };

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

} // end namespace

#endif
