// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_VBVECTOR_HH
#define DUNE_ISTL_VBVECTOR_HH

#include <cmath>
#include <complex>
#include <iostream>
#include <iterator>
#include <memory>

#include <dune/common/iteratorfacades.hh>
#include "istlexception.hh"
#include "bvector.hh"

#include <dune/istl/blocklevel.hh>

/** \file
 * \brief ???
 */

namespace Dune {

  /**
              @addtogroup ISTL_SPMV
              @{
   */

  /**
      \brief A Vector of blocks with different blocksizes.

          implements a vector consisting of a number of blocks (to
          be given at run-time) which themselves consist of a number
          of blocks (also given at run-time) of the given type B.

          VariableBlockVector is a container of containers!

   */
  template<class B, class A=std::allocator<B> >
  class VariableBlockVector : public Imp::block_vector_unmanaged<B,A>
                              // this derivation gives us all the blas level 1 and norms
                              // on the large array. However, access operators have to be
                              // overwritten.
  {
    // just a shorthand
    typedef Imp::BlockVectorWindow<B,A> window_type;

  public:

    //===== type definitions and constants

    //! export the type representing the field
    using field_type = typename Imp::BlockTraits<B>::field_type;

    //! export the allocator type
    typedef A allocator_type;

    /** \brief Export type used for references to container entries
     *
     * \note This is not B&, but an internal proxy class!
     */
    typedef window_type& reference;

    /** \brief Export type used for const references to container entries
     *
     * \note This is not B&, but an internal proxy class!
     */
    typedef const window_type& const_reference;

    //! The size type for the index access
    typedef typename A::size_type size_type;

    /** \brief Type of the elements of the outer vector, i.e., dynamic vectors of B
     *
     * Note that this is *not* the type referred to by the iterators and random access operators,
     * which return proxy objects.
     */
    typedef BlockVector<B,A> value_type;

    /** \brief Same as value_type, here for historical reasons
     */
    typedef BlockVector<B,A> block_type;

    /** increment block level counter, yes, it is two levels because
            VariableBlockVector is a container of containers
     */
    [[deprecated("Use free function blockLevel(). Will be removed after 2.8.")]]
    static constexpr auto blocklevel = blockLevel<B>()+2;

    //===== constructors and such

    /** constructor without arguments makes empty vector,
            object cannot be used yet
     */
    VariableBlockVector () : Imp::block_vector_unmanaged<B,A>()
    {
      // nothing is known ...
      nblocks = 0;
      block = nullptr;
      initialized = false;
    }

    /** make vector with given number of blocks, but size of each block is not yet known,
            object cannot be used yet
     */
    explicit VariableBlockVector (size_type _nblocks) : Imp::block_vector_unmanaged<B,A>()
    {
      // we can allocate the windows now
      nblocks = _nblocks;
      if (nblocks>0)
      {
        block = windowAllocator_.allocate(nblocks);
        new (block) window_type[nblocks];
      }
      else
      {
        nblocks = 0;
        block = nullptr;
      }

      // Note: memory in base class still not allocated
      // the vector not usable
      initialized = false;
    }

    /** make vector with given number of blocks each having a constant size,
            object is fully usable then.

            \param _nblocks Number of blocks
            \param m Number of elements in each block
     */
    VariableBlockVector (size_type _nblocks, size_type m) : Imp::block_vector_unmanaged<B,A>()
    {
      // and we can allocate the big array in the base class
      this->n = _nblocks*m;
      if (this->n>0)
      {
        this->p = allocator_.allocate(this->n);
        new (this->p)B[this->n];
      }
      else
      {
        this->n = 0;
        this->p = nullptr;
      }

      // we can allocate the windows now
      nblocks = _nblocks;
      if (nblocks>0)
      {
        // allocate and construct the windows
        block = windowAllocator_.allocate(nblocks);
        new (block) window_type[nblocks];

        // set the windows into the big array
        for (size_type i=0; i<nblocks; ++i)
          block[i].set(m,this->p+(i*m));
      }
      else
      {
        nblocks = 0;
        block = nullptr;
      }

      // and the vector is usable
      initialized = true;
    }

    //! copy constructor, has copy semantics
    VariableBlockVector (const VariableBlockVector& a)
    {
      // allocate the big array in the base class
      this->n = a.n;
      if (this->n>0)
      {
        // allocate and construct objects
        this->p = allocator_.allocate(this->n);
        new (this->p)B[this->n];

        // copy data
        for (size_type i=0; i<this->n; i++) this->p[i]=a.p[i];
      }
      else
      {
        this->n = 0;
        this->p = nullptr;
      }

      // we can allocate the windows now
      nblocks = a.nblocks;
      if (nblocks>0)
      {
        // alloc
        block = windowAllocator_.allocate(nblocks);
        new (block) window_type[nblocks];

        // and we must set the windows
        block[0].set(a.block[0].getsize(),this->p);           // first block
        for (size_type i=1; i<nblocks; ++i)                         // and the rest
          block[i].set(a.block[i].getsize(),block[i-1].getptr()+block[i-1].getsize());
      }
      else
      {
        nblocks = 0;
        block = nullptr;
      }

      // and we have a usable vector
      initialized = true;
    }

    //! free dynamic memory
    ~VariableBlockVector ()
    {
      if (this->n>0) {
        size_type i=this->n;
        while (i)
          this->p[--i].~B();
        allocator_.deallocate(this->p,this->n);
      }
      if (nblocks>0) {
        size_type i=nblocks;
        while (i)
          block[--i].~window_type();
        windowAllocator_.deallocate(block,nblocks);
      }

    }


    //! same effect as constructor with same argument
    void resize (size_type _nblocks)
    {
      // deconstruct objects and deallocate memory if necessary
      if (this->n>0) {
        size_type i=this->n;
        while (i)
          this->p[--i].~B();
        allocator_.deallocate(this->p,this->n);
      }
      if (nblocks>0) {
        size_type i=nblocks;
        while (i)
          block[--i].~window_type();
        windowAllocator_.deallocate(block,nblocks);
      }
      this->n = 0;
      this->p = nullptr;

      // we can allocate the windows now
      nblocks = _nblocks;
      if (nblocks>0)
      {
        block = windowAllocator_.allocate(nblocks);
        new (block) window_type[nblocks];
      }
      else
      {
        nblocks = 0;
        block = nullptr;
      }

      // and the vector not fully usable
      initialized = false;
    }

    //! same effect as constructor with same argument
    void resize (size_type _nblocks, size_type m)
    {
      // deconstruct objects and deallocate memory if necessary
      if (this->n>0) {
        size_type i=this->n;
        while (i)
          this->p[--i].~B();
        allocator_.deallocate(this->p,this->n);
      }
      if (nblocks>0) {
        size_type i=nblocks;
        while (i)
          block[--i].~window_type();
        windowAllocator_.deallocate(block,nblocks);
      }

      // and we can allocate the big array in the base class
      this->n = _nblocks*m;
      if (this->n>0)
      {
        this->p = allocator_.allocate(this->n);
        new (this->p)B[this->n];
      }
      else
      {
        this->n = 0;
        this->p = nullptr;
      }

      // we can allocate the windows now
      nblocks = _nblocks;
      if (nblocks>0)
      {
        // allocate and construct objects
        block = windowAllocator_.allocate(nblocks);
        new (block) window_type[nblocks];

        // set the windows into the big array
        for (size_type i=0; i<nblocks; ++i)
          block[i].set(m,this->p+(i*m));
      }
      else
      {
        nblocks = 0;
        block = nullptr;
      }

      // and the vector is usable
      initialized = true;
    }

    //! assignment
    VariableBlockVector& operator= (const VariableBlockVector& a)
    {
      if (&a!=this)     // check if this and a are different objects
      {
        // reallocate arrays if necessary
        // Note: still the block sizes may vary !
        if (this->n!=a.n || nblocks!=a.nblocks)
        {
          // deconstruct objects and deallocate memory if necessary
          if (this->n>0) {
            size_type i=this->n;
            while (i)
              this->p[--i].~B();
            allocator_.deallocate(this->p,this->n);
          }
          if (nblocks>0) {
            size_type i=nblocks;
            while (i)
              block[--i].~window_type();
            windowAllocator_.deallocate(block,nblocks);
          }

          // allocate the big array in the base class
          this->n = a.n;
          if (this->n>0)
          {
            // allocate and construct objects
            this->p = allocator_.allocate(this->n);
            new (this->p)B[this->n];
          }
          else
          {
            this->n = 0;
            this->p = nullptr;
          }

          // we can allocate the windows now
          nblocks = a.nblocks;
          if (nblocks>0)
          {
            // alloc
            block = windowAllocator_.allocate(nblocks);
            new (block) window_type[nblocks];
          }
          else
          {
            nblocks = 0;
            block = nullptr;
          }
        }

        // copy block structure, might be different although
        // sizes are the same !
        if (nblocks>0)
        {
          block[0].set(a.block[0].getsize(),this->p);                 // first block
          for (size_type i=1; i<nblocks; ++i)                               // and the rest
            block[i].set(a.block[i].getsize(),block[i-1].getptr()+block[i-1].getsize());
        }

        // and copy the data
        for (size_type i=0; i<this->n; i++) this->p[i]=a.p[i];
      }

      // and we have a usable vector
      initialized = true;

      return *this;     // Gebe Referenz zurueck damit a=b=c; klappt
    }


    //===== assignment from scalar

    //! assign from scalar
    VariableBlockVector& operator= (const field_type& k)
    {
      (static_cast<Imp::block_vector_unmanaged<B,A>&>(*this)) = k;
      return *this;
    }


    //===== the creation interface

    class CreateIterator;

#ifndef DOXYGEN

    // The window_type does not hand out a reference to its size,
    // so in order to provide a valid iterator, we need a workaround
    // to make assignment possible. This proxy enables just that by
    // implicitly converting to the stored size for read access and
    // tunneling assignment to the accessor method of the window.
    struct SizeProxy
    {

      operator size_type() const
      {
        return target->getsize();
      }

      SizeProxy& operator=(size_type size)
      {
        target->setsize(size);
        return *this;
      }

    private:

      friend class CreateIterator;

      SizeProxy(window_type& t)
        : target(&t)
      {}

      window_type* target;
    };

#endif // DOXYGEN

    //! Iterator class for sequential creation of blocks
    class CreateIterator
    {
    public:
      //! iterator category
      using iterator_category = std::output_iterator_tag;

      //! value type
      using value_type = size_type;

      /**
       * \brief difference type (unused)
       *
       * This type is required by the C++ standard, but not used for
       * output iterators.
       */
      using difference_type = void;

      //! pointer type
      using pointer = size_type*;

      //! reference type
      using reference = SizeProxy;

      //! constructor
      CreateIterator (VariableBlockVector& _v, int _i, bool _isEnd) :
        v(_v),
        i(_i),
        isEnd(_isEnd) {}

      ~CreateIterator() {
        // When the iterator gets destructed, we allocate the memory
        // for the VariableBlockVector if
        // 1. the current iterator was not created as enditerator
        // 2. we're at the last block
        // 3. the vector hasn't been initialized earlier
        if (not isEnd && i==v.nblocks && not v.initialized)
          v.allocate();
      }

      //! prefix increment
      CreateIterator& operator++()
      {
        // go to next block
        ++i;

        return *this;
      }

      /** \brief postfix increment operator */
      CreateIterator operator++ (int)
      {
        CreateIterator tmp(*this);
        this->operator++();
        return tmp;
      }

      //! inequality
      bool operator!= (const CreateIterator& it) const
      {
        return (i!=it.i) || (&v!=&it.v);
      }

      //! equality
      bool operator== (const CreateIterator& it) const
      {
        return (i==it.i) && (&v==&it.v);
      }

      //! dereferencing
      size_type index () const
      {
        return i;
      }

      //! set size of current block
      void setblocksize (size_type _k)
      {
        v.block[i].setsize(_k);
      }

      //! Access size of current block
#ifdef DOXYGEN
      size_type&
#else
      SizeProxy
#endif
      operator*()
      {
        return {v.block[i]};
      }

    private:
      VariableBlockVector& v;     // my vector
      size_type i;                      // current block to be defined
      const bool isEnd; // flag if this object was created as the end iterator.
    };

    // CreateIterator wants to set all the arrays ...
    friend class CreateIterator;

    //! get initial create iterator
    CreateIterator createbegin ()
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (initialized) DUNE_THROW(ISTLError,"no CreateIterator in initialized state");
#endif
      return CreateIterator(*this,0, false);
    }

    //! get create iterator pointing to one after the last block
    CreateIterator createend ()
    {
      return CreateIterator(*this,nblocks, true);
    }


    //===== access to components
    // has to be overwritten from base class because it must
    // return access to the windows

    //! random access to blocks
    window_type& operator[] (size_type i)
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (i>=nblocks) DUNE_THROW(ISTLError,"index out of range");
#endif
      return block[i];
    }

    //! same for read only access
    const window_type& operator[] (size_type i) const
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (i<0 || i>=nblocks) DUNE_THROW(ISTLError,"index out of range");
#endif
      return block[i];
    }

    //! Iterator class for sequential access
    template <class T, class R>
    class RealIterator
    : public RandomAccessIteratorFacade<RealIterator<T,R>, T, R>
    {
    public:
      //! constructor, no arguments
      RealIterator ()
      {
        p = nullptr;
        i = 0;
      }

      //! constructor
      RealIterator (window_type* _p, size_type _i)
      : p(_p), i(_i)
      {}

      //! prefix increment
      void increment()
      {
        ++i;
      }

      //! prefix decrement
      void decrement()
      {
        --i;
      }

      //! equality
      bool equals (const RealIterator& it) const
      {
        return (p+i)==(it.p+it.i);
      }

      //! dereferencing
      window_type& dereference () const
      {
        return p[i];
      }

      void advance(std::ptrdiff_t d)
      {
        i+=d;
      }

      std::ptrdiff_t distanceTo(const RealIterator& o) const
      {
        return o.i-i;
      }

      // Needed for operator[] of the iterator
      window_type& elementAt (std::ptrdiff_t offset) const
      {
        return p[i+offset];
      }

      /** \brief Return the index of the entry this iterator is pointing to */
      size_type index() const
      {
        return i;
      }

    private:
      window_type* p;
      size_type i;
    };

    using Iterator = RealIterator<value_type,window_type&>;

    //! begin Iterator
    Iterator begin ()
    {
      return Iterator(block,0);
    }

    //! end Iterator
    Iterator end ()
    {
      return Iterator(block,nblocks);
    }

    //! @returns an iterator that is positioned before
    //! the end iterator of the vector, i.e. at the last entry.
    Iterator beforeEnd ()
    {
      return Iterator(block,nblocks-1);
    }

    //! @returns an iterator that is positioned before
    //! the first entry of the vector.
    Iterator beforeBegin () const
    {
      return Iterator(block,-1);
    }

    /** \brief Export the iterator type using std naming rules */
    using iterator = Iterator;

    /** \brief Const iterator */
    using ConstIterator = RealIterator<const value_type, const window_type&>;

    /** \brief Export the const iterator type using std naming rules */
    using const_iterator = ConstIterator;

    //! begin ConstIterator
    ConstIterator begin () const
    {
      return ConstIterator(block,0);
    }

    //! end ConstIterator
    ConstIterator end () const
    {
      return ConstIterator(block,nblocks);
    }

    //! @returns an iterator that is positioned before
    //! the end iterator of the vector. i.e. at the last element.
    ConstIterator beforeEnd() const
    {
      return ConstIterator(block,nblocks-1);
    }

    //! end ConstIterator
    ConstIterator rend () const
    {
      return ConstIterator(block,-1);
    }

    //! random access returning iterator (end if not contained)
    Iterator find (size_type i)
    {
      return Iterator(block,std::min(i,nblocks));
    }

    //! random access returning iterator (end if not contained)
    ConstIterator find (size_type i) const
    {
      return ConstIterator(block,std::min(i,nblocks));
    }

    //===== sizes

    //! number of blocks in the vector (are of variable size here)
    size_type N () const
    {
      return nblocks;
    }

    /** Number of blocks in the vector
     *
     * Returns the same value as method N(), because the vector is dense
    */
    size_type size () const
    {
      return nblocks;
    }


  private:

    void allocate() {
      if (this->initialized)
        DUNE_THROW(ISTLError, "Attempt to re-allocate already initialized VariableBlockVector");

      // calculate space needed:
      this->n=0;
      for(size_type i = 0; i < nblocks; i++) {
        this->n += block[i].size();
      }

      // now we can allocate the big array in the base class of v
      if (this->n>0)
      {
        // allocate and construct objects
        this->p = allocator_.allocate(this->n);
        new (this->p)B[this->n];
      }
      else
      {
        this->p = nullptr;
      }

      // and we set the window pointers
      this->block[0].setptr(this->p); // pointer to first block
      for (size_type j=1; j<nblocks; ++j) // and the rest
        block[j].setptr(block[j-1].getptr()+block[j-1].getsize());

      // and the vector is ready
      this->initialized = true;
    }

    size_type nblocks;            // number of blocks in vector
    window_type* block;     // array of blocks pointing to the array in the base class
    bool initialized;       // true if vector has been initialized

    A allocator_;

    typename std::allocator_traits<A>::template rebind_alloc<window_type> windowAllocator_;
  };



  /** @} end documentation */

} // end namespace

#endif
