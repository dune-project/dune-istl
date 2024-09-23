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

#include <dune/common/ftraits.hh>
#include <dune/common/indexediterator.hh>
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
  class VariableBlockVector : public Imp::block_vector_unmanaged<B,typename A::size_type>
                              // this derivation gives us all the blas level 1 and norms
                              // on the large array. However, access operators have to be
                              // overwritten.
  {
    using Base = Imp::block_vector_unmanaged<B,typename A::size_type>;

    // just a shorthand
    using window_type = Imp::BlockVectorWindow<B,A>;

    // data-structure holding the windows (but not the actual data)
    using VectorWindows = std::vector<window_type, typename std::allocator_traits<A>::template rebind_alloc<window_type>>;

    // block type bool is not supported since std::vector<bool> is used for storage
    static_assert(not std::is_same_v<B,bool>, "Block type 'bool' not supported by VariableBlockVector.");

  public:

    //===== type definitions and constants

    //! export the type representing the field
    using field_type = typename Imp::BlockTraits<B>::field_type;

    //! export the allocator type
    using allocator_type = A;

    /** \brief Export type used for references to container entries
     *
     * \note This is not B&, but an internal proxy class!
     */
    using reference = window_type&;

    /** \brief Export type used for const references to container entries
     *
     * \note This is not B&, but an internal proxy class!
     */
    using const_reference = const window_type&;

    //! The size type for the index access
    using size_type = typename A::size_type;

    /** \brief Type of the elements of the outer vector, i.e., dynamic vectors of B
     *
     * Note that this is *not* the type referred to by the iterators and random access operators,
     * which return proxy objects.
     */
    using value_type = BlockVector<B,A>;

    /** \brief Same as value_type, here for historical reasons
     */
    using block_type = BlockVector<B,A>;

    //===== constructors and such

    /**
     * \brief Constructor without arguments makes an empty vector.
     * \note object cannot be used yet. The size and block sizes need to be initialized.
     */
    VariableBlockVector () :
      Base()
    {}

    /**
     * \brief Construct a vector with given number of blocks, but size of each
     * block is not yet known.
     * \note Object cannot be used yet. Needs to be initialized using the
     * `createbegin()` and `createend()` create-iterators to fill the block sizes.
     */
    explicit VariableBlockVector (size_type numBlocks) :
      Base(),
      block(numBlocks)
    {}

    /**
     * \brief Construct a vector with given number of blocks each having a
     * constant size.
     * \note Object is fully usable after construction.
     *
     * \param numBlocks Number of blocks
     * \param blockSize Number of elements in each block
     */
    VariableBlockVector (size_type numBlocks, size_type blockSize) :
      Base(),
      block(numBlocks),
      storage_(numBlocks*blockSize)
    {
      // and we can allocate the big array in the base class
      syncBaseArray();

      // set the windows into the big array
      for (size_type i=0; i<numBlocks; ++i)
        block[i].set(blockSize,this->p+(i*blockSize));

      // and the vector is usable
      initialized = true;
    }

    //! Copy constructor, has copy semantics
    VariableBlockVector (const VariableBlockVector& a) :
      Base(static_cast<const Base&>(a)),
      block(a.block),
      storage_(a.storage_)
    {
      syncBaseArray();

      // and we must set the windows
      if (block.size()>0) {
        block[0].set(block[0].getsize(),this->p);           // first block
        for (size_type i=1; i<block.size(); ++i)                         // and the rest
          block[i].set(block[i].getsize(),block[i-1].getptr()+block[i-1].getsize());
      }

      // and we have a usable vector
      initialized = a.initialized;
    }

    //! Move constructor:
    VariableBlockVector (VariableBlockVector&& tmp) :
      Base()
    {
      tmp.swap(*this);
    }

    ~VariableBlockVector () = default;


    //! Copy and move assignment
    VariableBlockVector& operator= (VariableBlockVector tmp)
    {
      tmp.swap(*this);
      return *this;
    }

    //! Exchange the storage and internal state with `other`.
    void swap (VariableBlockVector& other) noexcept
    {
      using std::swap;
      swap(storage_, other.storage_);
      swap(block, other.block);
      swap(initialized, other.initialized);

      other.syncBaseArray();
      syncBaseArray();
    }

    //! Free function to swap the storage and internal state of `lhs` with `rhs`.
    friend void swap (VariableBlockVector& lhs, VariableBlockVector& rhs) noexcept
    {
      lhs.swap(rhs);
    }

    //! same effect as constructor with same argument
    void resize (size_type numBlocks)
    {
      storage_.clear();

      syncBaseArray();

      // we can allocate the windows now
      block.resize(numBlocks);

      // and the vector not fully usable
      initialized = false;
    }

    //! same effect as constructor with same argument
    void resize (size_type numBlocks, size_type blockSize)
    {
      // and we can allocate the big array in the base class
      storage_.resize(numBlocks*blockSize);
      block.resize(numBlocks);
      syncBaseArray();

      // set the windows into the big array
      for (size_type i=0; i<block.size(); ++i)
        block[i].set(blockSize,this->p+(i*blockSize));

      // and the vector is usable
      initialized = true;
    }

    //===== assignment from scalar

    //! Set all entries to the given scalar `k`.
    VariableBlockVector& operator= (const field_type& k)
    {
      (static_cast<Imp::block_vector_unmanaged<B,size_type>&>(*this)) = k;
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

      operator size_type () const
      {
        return target->getsize();
      }

      SizeProxy& operator= (size_type size)
      {
        target->setsize(size);
        return *this;
      }

    private:

      friend class CreateIterator;

      SizeProxy (window_type& t) :
        target(&t)
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
        v(&_v),
        i(_i),
        isEnd(_isEnd)
      {}

      ~CreateIterator ()
      {
        // When the iterator gets destructed, we allocate the memory
        // for the VariableBlockVector if
        // 1. the current iterator was not created as enditerator
        // 2. we're at the last block
        // 3. the vector hasn't been initialized earlier
        if (not isEnd && i==v->block.size() && not v->initialized)
          v->allocate();
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
          return not (*this == it);
      }

      //! equality
      bool operator== (const CreateIterator& it) const
      {
        return (i==it.i) && (v==it.v);
      }

      //! dereferencing
      size_type index () const
      {
        return i;
      }

      //! set size of current block
      void setblocksize (size_type _k)
      {
        v->block[i].setsize(_k);
      }

      //! Access size of current block
#ifdef DOXYGEN
      size_type&
#else
      SizeProxy
#endif
      operator*()
      {
        return {v->block[i]};
      }

    private:
      VariableBlockVector* v;     // my vector
      size_type i;                      // current block to be defined
      bool isEnd; // flag if this object was created as the end iterator.
    };

    // CreateIterator wants to set all the arrays ...
    friend class CreateIterator;

    //! get initial create iterator
    CreateIterator createbegin ()
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (initialized) DUNE_THROW(ISTLError,"no CreateIterator in initialized state");
#endif
      return CreateIterator(*this, 0, false);
    }

    //! get create iterator pointing to one after the last block
    CreateIterator createend ()
    {
      return CreateIterator(*this, block.size(), true);
    }


    //===== access to components
    // has to be overwritten from base class because it must
    // return access to the windows

    //! random access to blocks
    window_type& operator[] (size_type i)
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (i>=block.size()) DUNE_THROW(ISTLError,"index out of range");
#endif
      return block[i];
    }

    //! same for read only access
    const window_type& operator[] (size_type i) const
    {
#ifdef DUNE_ISTL_WITH_CHECKING
      if (i<0 || i>=block.size()) DUNE_THROW(ISTLError,"index out of range");
#endif
      return block[i];
    }

    using Iterator = IndexedIterator<typename VectorWindows::iterator>;

    //! begin Iterator
    Iterator begin ()
    {
      return Iterator{block.begin()};
    }

    //! end Iterator
    Iterator end ()
    {
      return Iterator{block.end()};
    }

    //! @returns an iterator that is positioned before
    //! the end iterator of the vector, i.e. at the last entry.
    Iterator beforeEnd ()
    {
      return Iterator{--block.end()};
    }

    //! @returns an iterator that is positioned before
    //! the first entry of the vector.
    Iterator beforeBegin ()
    {
      return Iterator{--block.begin()};
    }

    /** \brief Export the iterator type using std naming rules */
    using iterator = Iterator;

    /** \brief Const iterator */
    using ConstIterator = IndexedIterator<typename VectorWindows::const_iterator>;

    /** \brief Export the const iterator type using std naming rules */
    using const_iterator = ConstIterator;

    //! begin ConstIterator
    ConstIterator begin () const
    {
      return ConstIterator{block.begin()};
    }

    //! end ConstIterator
    ConstIterator end () const
    {
      return ConstIterator{block.end()};
    }

    //! @returns an iterator that is positioned before
    //! the end iterator of the vector. i.e. at the last element.
    ConstIterator beforeEnd () const
    {
      return ConstIterator{--block.end()};
    }

    //! @returns an iterator that is positioned before
    //! the first entry of the vector.
    ConstIterator beforeBegin () const
    {
      return ConstIterator{--block.begin()};
    }

    //! end ConstIterator
    ConstIterator rend () const
    {
      return ConstIterator{block.rend()};
    }

    //! random access returning iterator (end if not contained)
    Iterator find (size_type i)
    {
      Iterator tmp = block.begin();
      tmp+=std::min(i, block.size());
      return tmp;
    }

    //! random access returning iterator (end if not contained)
    ConstIterator find (size_type i) const
    {
      ConstIterator tmp = block.begin();
      tmp+=std::min(i, block.size());
      return tmp;
    }

    //===== sizes

    //! number of blocks in the vector (are of variable size here)
    size_type N () const noexcept
    {
      return block.size();
    }

    /** Number of blocks in the vector
     *
     * Returns the same value as method N(), because the vector is dense
    */
    size_type size () const noexcept
    {
      return block.size();
    }


  private:

    void allocate ()
    {
      if (this->initialized)
        DUNE_THROW(ISTLError, "Attempt to re-allocate already initialized VariableBlockVector");

      // calculate space needed:
      size_type storageNeeded = 0;
      for(size_type i = 0; i < block.size(); i++)
        storageNeeded += block[i].size();

      storage_.resize(storageNeeded);
      syncBaseArray();

      // and we set the window pointers
      block[0].setptr(this->p); // pointer to first block
      for (size_type j=1; j<block.size(); ++j) // and the rest
        block[j].setptr(block[j-1].getptr()+block[j-1].getsize());

      // and the vector is ready
      this->initialized = true;
    }

    void syncBaseArray () noexcept
    {
      this->p = storage_.data();
      this->n = storage_.size();
    }

    VectorWindows block = {}; // vector of blocks pointing to the array in the base class
    std::vector<B, A> storage_ = {};
    bool initialized = false; // true if vector has been initialized
  };

  /** @addtogroup DenseMatVec
      @{
   */
  template<class B, class A>
  struct FieldTraits< VariableBlockVector<B, A> >
  {
    typedef typename FieldTraits<B>::field_type field_type;
    typedef typename FieldTraits<B>::real_type real_type;
  };
  /**
      @}
   */


  /** @} end documentation */

} // end namespace

#endif
