// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_AMGHIERARCHY_HH
#define DUNE_AMGHIERARCHY_HH

#include <list>
#include <memory>
#include <limits>
#include <dune/common/stdstreams.hh>
#include <dune/common/timer.hh>
#include <dune/common/bigunsignedint.hh>
#include <dune/istl/paamg/construction.hh>

namespace Dune
{
  namespace Amg
  {
    /**
     * @addtogroup ISTL_PAAMG
     *
     * @{
     */

    /** @file
     * @author Markus Blatt
     * @brief Provides a classes representing the hierarchies in AMG.
     */

    /**
     * @brief A hierarchy of containers (e.g. matrices or vectors)
     *
     * Because sometimes a redistribution of the parallel data might be
     * advisable one can add redistributed version of the container at
     * each level.
     */
    template<typename T, typename A=std::allocator<T> >
    class Hierarchy
    {
    public:
      /**
       * @brief The type of the container we store.
       */
      typedef T MemberType;

      template<typename T1, typename T2>
      class LevelIterator;

    private:
      /**
       * @brief An element in the hierarchy.
       */
      struct Element
      {
        friend class LevelIterator<Hierarchy<T,A>, T>;
        friend class LevelIterator<const Hierarchy<T,A>, const T>;

        /** @brief The next coarser element in the list. */
        std::weak_ptr<Element> coarser_;

        /** @brief The next finer element in the list. */
        std::shared_ptr<Element> finer_;

        /** @brief Pointer to the element. */
        std::shared_ptr<MemberType> element_;

        /** @brief The redistributed version of the element. */
        std::shared_ptr<MemberType> redistributed_;
      };
    public:

      /**
       * @brief The allocator to use for the list elements.
       */
      using Allocator = typename std::allocator_traits<A>::template rebind_alloc<Element>;

      typedef typename ConstructionTraits<T>::Arguments Arguments;

      /**
       * @brief Construct a new hierarchy.
       * @param first std::shared_ptr to the first element in the hierarchy.
       */
      Hierarchy(const std::shared_ptr<MemberType> & first);

      /**
       * @brief Construct an empty hierarchy.
       */
      Hierarchy() : levels_(0)
      {}

      /**
       * @brief Copy constructor (deep copy!).
       */
      Hierarchy(const Hierarchy& other);

      /**
       * @brief Add an element on a coarser level.
       * @param args The arguments needed for the construction.
       */
      void addCoarser(Arguments& args);

      void addRedistributedOnCoarsest(Arguments& args);

      /**
       * @brief Add an element on a finer level.
       * @param args The arguments needed for the construction.
       */
      void addFiner(Arguments& args);

      /**
       * @brief Iterator over the levels in the hierarchy.
       *
       * operator++() moves to the next coarser level in the hierarchy.
       * while operator--() moves to the next finer level in the hierarchy.
       */
      template<class C, class T1>
      class LevelIterator
        : public BidirectionalIteratorFacade<LevelIterator<C,T1>,T1,T1&>
      {
        friend class LevelIterator<typename std::remove_const<C>::type,
            typename std::remove_const<T1>::type >;
        friend class LevelIterator<const typename std::remove_const<C>::type,
            const typename std::remove_const<T1>::type >;

      public:
        /** @brief Constructor. */
        LevelIterator()
        {}

        LevelIterator(std::shared_ptr<Element> element)
          : element_(element)
        {}

        /** @brief Copy constructor. */
        LevelIterator(const LevelIterator<typename std::remove_const<C>::type,
                          typename std::remove_const<T1>::type>& other)
          : element_(other.element_)
        {}

        /** @brief Copy constructor. */
        LevelIterator(const LevelIterator<const typename std::remove_const<C>::type,
                          const typename std::remove_const<T1>::type>& other)
          : element_(other.element_)
        {}

        /**
         * @brief Equality check.
         */
        bool equals(const LevelIterator<typename std::remove_const<C>::type,
                        typename std::remove_const<T1>::type>& other) const
        {
          return element_ == other.element_;
        }

        /**
         * @brief Equality check.
         */
        bool equals(const LevelIterator<const typename std::remove_const<C>::type,
                        const typename std::remove_const<T1>::type>& other) const
        {
          return element_ == other.element_;
        }

        /** @brief Dereference the iterator. */
        T1& dereference() const
        {
          return *(element_->element_);
        }

        /** @brief Move to the next coarser level */
        void increment()
        {
          element_ = element_->coarser_.lock();
        }

        /** @brief Move to the next fine level */
        void decrement()
        {
          element_ = element_->finer_;
        }

        /**
         * @brief Check whether there was a redistribution at the current level.
         * @return True if there is a redistributed version of the container at the current level.
         */
        bool isRedistributed() const
        {
          return (bool)element_->redistributed_;
        }

        /**
         * @brief Get the redistributed container.
         * @return The redistributed container.
         */
        T1& getRedistributed() const
        {
          assert(element_->redistributed_);
          return *element_->redistributed_;
        }
        void addRedistributed(std::shared_ptr<T1> t)
        {
          element_->redistributed_ = t;
        }

        void deleteRedistributed()
        {
          element_->redistributed_ = nullptr;
        }

      private:
        std::shared_ptr<Element> element_;
      };

      /** @brief Type of the mutable iterator. */
      typedef LevelIterator<Hierarchy<T,A>,T> Iterator;

      /** @brief Type of the const iterator. */
      typedef LevelIterator<const Hierarchy<T,A>, const T> ConstIterator;

      /**
       * @brief Get an iterator positioned at the finest level.
       * @return An iterator positioned at the finest level.
       */
      Iterator finest();

      /**
       * @brief Get an iterator positioned at the coarsest level.
       * @return An iterator positioned at the coarsest level.
       */
      Iterator coarsest();


      /**
       * @brief Get an iterator positioned at the finest level.
       * @return An iterator positioned at the finest level.
       */
      ConstIterator finest() const;

      /**
       * @brief Get an iterator positioned at the coarsest level.
       * @return An iterator positioned at the coarsest level.
       */
      ConstIterator coarsest() const;

      /**
       * @brief Get the number of levels in the hierarchy.
       * @return The number of levels.
       */
      std::size_t levels() const;

    private:
      /** @brief fix memory management of the finest element in the hierarchy

          This object is passed in from outside, so that we have to
          deal with shared ownership.
      */
      std::shared_ptr<MemberType> originalFinest_;
      /** @brief The finest element in the hierarchy. */
      std::shared_ptr<Element> finest_;
      /** @brief The coarsest element in the hierarchy. */
      std::shared_ptr<Element> coarsest_;
      /** @brief The allocator for the list elements. */
      Allocator allocator_;
      /** @brief The number of levels in the hierarchy. */
      int levels_;
    };

    template<class T, class A>
    Hierarchy<T,A>::Hierarchy(const std::shared_ptr<MemberType> & first)
      : originalFinest_(first)
    {
      finest_ = std::allocate_shared<Element>(allocator_);
      finest_->element_ = originalFinest_;
      coarsest_ = finest_;
      levels_ = 1;
    }

    //! \brief deep copy of a given hierarchy
    //TODO: do we actually want to support this? This might be very expensive?!
    template<class T, class A>
    Hierarchy<T,A>::Hierarchy(const Hierarchy& other)
    : allocator_(other.allocator_),
      levels_(other.levels_)
    {
      if(!other.finest_)
      {
        finest_=coarsest_=nullptr;
        return;
      }
      finest_ = std::allocate_shared<Element>(allocator_);
      std::shared_ptr<Element> finer_;
      std::shared_ptr<Element> current_ = finest_;
      std::weak_ptr<Element> otherWeak_ = other.finest_;

      while(! otherWeak_.expired())
      {
        // create shared_ptr from weak_ptr, we just checked that this is safe
        std::shared_ptr<Element> otherCurrent_ = std::shared_ptr<Element>(otherWeak_);
        // clone current level
        //TODO: should we use the allocator?
        current_->element_ =
          std::make_shared<MemberType>(*(otherCurrent_->element_));
        current_->finer_=finer_;
        if(otherCurrent_->redistributed_)
          current_->redistributed_ =
            std::make_shared<MemberType>(*(otherCurrent_->redistributed_));
        finer_=current_;
        if(not otherCurrent_->coarser_.expired())
        {
          auto c = std::allocate_shared<Element>(allocator_);
          current_->coarser_ = c;
          current_ = c;
        }
        // go to coarser level
        otherWeak_ = otherCurrent_->coarser_;
      }
      coarsest_=current_;
    }

    template<class T, class A>
    std::size_t Hierarchy<T,A>::levels() const
    {
      return levels_;
    }

    template<class T, class A>
    void Hierarchy<T,A>::addRedistributedOnCoarsest(Arguments& args)
    {
      coarsest_->redistributed_ = ConstructionTraits<MemberType>::construct(args);
    }

    template<class T, class A>
    void Hierarchy<T,A>::addCoarser(Arguments& args)
    {
      if(!coarsest_) {
        // we have no levels at all...
        assert(!finest_);
        // allocate into the shared_ptr
        originalFinest_ = ConstructionTraits<MemberType>::construct(args);
        coarsest_ = std::allocate_shared<Element>(allocator_);
        coarsest_->element_ = originalFinest_;
        finest_ = coarsest_;
      }else{
        auto old_coarsest = coarsest_;
        coarsest_ = std::allocate_shared<Element>(allocator_);
        coarsest_->finer_ = old_coarsest;
        coarsest_->element_ = ConstructionTraits<MemberType>::construct(args);
        old_coarsest->coarser_ = coarsest_;
      }
      ++levels_;
    }


    template<class T, class A>
    void Hierarchy<T,A>::addFiner(Arguments& args)
    {
      //TODO: wouldn't it be better to do this in the constructor?'
      if(!finest_) {
        // we have no levels at all...
        assert(!coarsest_);
        // allocate into the shared_ptr
        originalFinest_ = ConstructionTraits<MemberType>::construct(args);
        finest_ = std::allocate_shared<Element>(allocator_);
        finest_->element = originalFinest_;
        coarsest_ = finest_;
      }else{
        finest_->finer_ = std::allocate_shared<Element>(allocator_);
        finest_->finer_->coarser_ = finest_;
        finest_ = finest_->finer_;
        finest_->element = ConstructionTraits<T>::construct(args);
      }
      ++levels_;
    }

    template<class T, class A>
    typename Hierarchy<T,A>::Iterator Hierarchy<T,A>::finest()
    {
      return Iterator(finest_);
    }

    template<class T, class A>
    typename Hierarchy<T,A>::Iterator Hierarchy<T,A>::coarsest()
    {
      return Iterator(coarsest_);
    }

    template<class T, class A>
    typename Hierarchy<T,A>::ConstIterator Hierarchy<T,A>::finest() const
    {
      return ConstIterator(finest_);
    }

    template<class T, class A>
    typename Hierarchy<T,A>::ConstIterator Hierarchy<T,A>::coarsest() const
    {
      return ConstIterator(coarsest_);
    }
    /** @} */
  } // namespace Amg
} // namespace Dune

#endif
