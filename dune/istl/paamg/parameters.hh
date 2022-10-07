// SPDX-FileCopyrightText: Copyright © DUNE Project contributors, see file LICENSE.md in module root
// SPDX-License-Identifier: LicenseRef-GPL-2.0-only-with-DUNE-exception
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_AMG_PARAMETERS_HH
#define DUNE_AMG_PARAMETERS_HH

#include <cstddef>

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
     * @brief Parameter classes for customizing AMG
     *
     * All parameters of the AMG can be set by using the class Parameter, which
     * can be provided to CoarsenCriterion via its constructor.
     */

    /**
     * @brief Parameters needed to check whether a node depends on another.
     */
    class DependencyParameters
    {
    public:
      /** @brief Constructor */
      DependencyParameters()
        : alpha_(1.0/3.0), beta_(1.0E-5)
      {}

      /**
       * @brief Set threshold for marking nodes as isolated.
       * The default value is 1.0E-5.
       */
      void setBeta(double b)
      {
        beta_ = b;
      }

      /**
       * @brief Get the threshold for marking nodes as isolated.
       * The default value is 1.0E-5.
       * @return beta
       */
      double beta() const
      {
        return beta_;
      }

      /**
       * @brief Set the scaling value for marking connections as strong.
       * Default value is 1/3
       */
      void setAlpha(double a)
      {
        alpha_ = a;
      }

      /**
       * @brief Get the scaling value for marking connections as strong.
       * Default value is 1/3
       */
      double alpha() const
      {
        return alpha_;
      }

    private:
      double alpha_, beta_;
    };

    /**
     * @brief Parameters needed for the aggregation process
     */
    class AggregationParameters :
      public DependencyParameters
    {
    public:
      /**
       * @brief Constructor.
       *
       * The parameters will be initialized with default values suitable
       * for 2D isotropic problems.
       *
       * If that does not fit your needs either use setDefaultValuesIsotropic
       * setDefaultValuesAnisotropic or setup the values by hand
       */
      AggregationParameters()
        : maxDistance_(2), minAggregateSize_(4), maxAggregateSize_(6),
          connectivity_(15), skipiso_(false)
      {}

      /**
       * @brief Sets reasonable default values for an isotropic problem.
       *
       * Reasonable means that we should end up with cube aggregates of
       * diameter 2.
       *
       * @param dim The dimension of the problem.
       * @param diameter The preferred diameter for the aggregation.
       */
      void setDefaultValuesIsotropic(std::size_t dim, std::size_t diameter=2)
      {
        maxDistance_=diameter-1;
        std::size_t csize=1;

        for(; dim>0; dim--) {
          csize*=diameter;
          maxDistance_+=diameter-1;
        }
        minAggregateSize_=csize;
        maxAggregateSize_=static_cast<std::size_t>(csize*1.5);
      }

      /**
       * @brief Sets reasonable default values for an anisotropic problem.
       *
       * Reasonable means that we should end up with cube aggregates with
       * sides of diameter 2 and sides in one dimension that are longer
       * (e.g. for 3D: 2x2x3).
       *
       * @param dim The dimension of the problem.
       * @param diameter The preferred diameter for the aggregation.
       */
      void setDefaultValuesAnisotropic(std::size_t dim,std::size_t diameter=2)
      {
        setDefaultValuesIsotropic(dim, diameter);
        maxDistance_+=dim-1;
      }
      /**
       * @brief Get the maximal distance allowed between two nodes in a aggregate.
       *
       * The distance between two nodes in a aggregate is the minimal number of edges
       * it takes to travel from one node to the other without leaving the aggregate.
       * @return The maximum distance allowed.
       */
      std::size_t maxDistance() const { return maxDistance_;}

      /**
       * @brief Set the maximal distance allowed between two nodes in a aggregate.
       *
       * The distance between two nodes in a aggregate is the minimal number of edges
       * it takes to travel from one node to the other without leaving the aggregate.
       * The default value is 2.
       * @param distance The maximum distance allowed.
       */
      void setMaxDistance(std::size_t distance) { maxDistance_ = distance;}

      /**
       * @brief Whether isolated aggregates will not be represented on
       * the coarse level.
       * @return True if these aggregates will be skipped.
       */
      bool skipIsolated() const
      {
        return skipiso_;
      }

      /**
       * @brief Set whether isolated aggregates will not be represented on
       * the coarse level.
       * @param skip True if these aggregates will be skipped.
       */
      void setSkipIsolated(bool skip)
      {
        skipiso_=skip;
      }

      /**
       * @brief Get the minimum number of nodes a aggregate has to consist of.
       * @return The minimum number of nodes.
       */
      std::size_t minAggregateSize() const { return minAggregateSize_;}

      /**
       * @brief Set the minimum number of nodes a aggregate has to consist of.
       *
       * the default value is 4.
       * @return The minimum number of nodes.
       */
      void setMinAggregateSize(std::size_t size){ minAggregateSize_=size;}

      /**
       * @brief Get the maximum number of nodes a aggregate is allowed to have.
       * @return The maximum number of nodes.
       */
      std::size_t maxAggregateSize() const { return maxAggregateSize_;}

      /**
       * @brief Set the maximum number of nodes a aggregate is allowed to have.
       *
       * The default values is 6.
       * @param size The maximum number of nodes.
       */
      void setMaxAggregateSize(std::size_t size){ maxAggregateSize_ = size;}

      /**
       * @brief Get the maximum number of connections a aggregate is allowed to have.
       *
       * This limit exists to achieve sparsity of the coarse matrix. the default value is 15.
       *
       * @return The maximum number of connections a aggregate is allowed to have.
       */
      std::size_t maxConnectivity() const { return connectivity_;}

      /**
       * @brief Set the maximum number of connections a aggregate is allowed to have.
       *
       * This limit exists to achieve sparsity of the coarse matrix. the default value is 15.
       *
       * @param connectivity The maximum number of connections a aggregate is allowed to have.
       */
      void setMaxConnectivity(std::size_t connectivity){ connectivity_ = connectivity;}

    private:
      std::size_t maxDistance_, minAggregateSize_, maxAggregateSize_, connectivity_;
      bool skipiso_;

    };


    /**
     * @brief Identifiers for the different accumulation modes.
     */
    enum AccumulationMode {
      /**
       * @brief No data accumulution.
       *
       * The coarse level data will be distributed to all processes.
       */
      noAccu = 0,
      /**
       * @brief Accumulate data to one process at once
       *
       * Once no further coarsening is possible all data will be accumulated to one process
       */
      atOnceAccu=1,
      /**
       * @brief Successively accumulate to fewer processes.
       */
      successiveAccu=2
    };




    /**
     * @brief Parameters for the complete coarsening process.
     */
    class CoarseningParameters : public AggregationParameters
    {
    public:
      /**
       * @brief Set the maximum number of levels allowed in the hierarchy.
       */
      void setMaxLevel(int l)
      {
        maxLevel_ = l;
      }
      /**
       * @brief Get the maximum number of levels allowed in the hierarchy.
       */
      int maxLevel() const
      {
        return maxLevel_;
      }

      /**
       * @brief Set the maximum number of unknowns allowed on the coarsest level.
       */
      void setCoarsenTarget(int nodes)
      {
        coarsenTarget_ = nodes;
      }

      /**
       * @brief Get the maximum number of unknowns allowed on the coarsest level.
       */
      int coarsenTarget() const
      {
        return coarsenTarget_;
      }

      /**
       * @brief Set the minimum coarsening rate to be achieved in each coarsening.
       *
       * The default value is 1.2
       */
      void setMinCoarsenRate(double rate)
      {
        minCoarsenRate_ = rate;
      }

      /**
       * @brief Get the minimum coarsening rate to be achieved.
       */
      double minCoarsenRate() const
      {
        return minCoarsenRate_;
      }

      /**
       * @brief Whether the data should be accumulated on fewer processes on coarser levels.
       */
      AccumulationMode accumulate() const
      {
        return accumulate_;
      }
      /**
       * @brief Set whether he data should be accumulated on fewer processes on coarser levels.
       */
      void setAccumulate(AccumulationMode accu)
      {
        accumulate_=accu;
      }

      void setAccumulate(bool accu){
        accumulate_=accu ? successiveAccu : noAccu;
      }
      /**
       * @brief Set the damping factor for the prolongation.
       *
       * @param d The new damping factor.
       */
      void setProlongationDampingFactor(double d)
      {
        dampingFactor_ = d;
      }

      /**
       * @brief Get the damping factor for the prolongation.
       *
       * @return d The damping factor.
       */
      double getProlongationDampingFactor() const
      {
        return dampingFactor_;
      }
      /**
       * @brief Constructor
       * @param maxLevel The maximum number of levels allowed in the matrix hierarchy (default: 100).
       * @param coarsenTarget If the number of nodes in the matrix is below this threshold the
       * coarsening will stop (default: 1000).
       * @param minCoarsenRate If the coarsening rate falls below this threshold the
       * coarsening will stop (default: 1.2)
       * @param prolongDamp The damping factor to apply to the prolongated update (default: 1.6)
       * @param accumulate Whether to accumulate the data onto fewer processors on coarser levels.
       */
      CoarseningParameters(int maxLevel=100, int coarsenTarget=1000, double minCoarsenRate=1.2,
                           double prolongDamp=1.6, AccumulationMode accumulate=successiveAccu)
        : maxLevel_(maxLevel), coarsenTarget_(coarsenTarget), minCoarsenRate_(minCoarsenRate),
          dampingFactor_(prolongDamp), accumulate_( accumulate)
      {}

    private:
      /**
       * @brief The maximum number of levels allowed in the hierarchy.
       */
      int maxLevel_;
      /**
       * @brief The maximum number of unknowns allowed on the coarsest level.
       */
      int coarsenTarget_;
      /**
       * @brief The minimum coarsening rate to be achieved.
       */
      double minCoarsenRate_;
      /**
       * @brief The damping factor to apply to the prologated correction.
       */
      double dampingFactor_;
      /**
       * @brief Whether the data should be agglomerated to fewer processor on
       * coarser levels.
       */
      AccumulationMode accumulate_;
    };

    /**
     * @brief All parameters for AMG.
     *
     * Instances of this class can be provided to CoarsenCriterion via its
     * constructor.
     */
    class Parameters : public CoarseningParameters
    {
    public:
      /**
       * @brief Set the debugging level.
       *
       * @param level If 0 no debugging output will be generated.
       * @warning In parallel the level has to be consistent over all procceses.
       */
      void setDebugLevel(int level)
      {
        debugLevel_ = level;
      }

      /**
       * @brief Get the debugging Level.
       *
       * @return 0 if no debugging output will be generated.
       */
      int debugLevel() const
      {
        return debugLevel_;
      }

      /**
       * @brief Set the number of presmoothing steps to apply
       * @param steps The number of steps:
       */
      void setNoPreSmoothSteps(std::size_t steps)
      {
        preSmoothSteps_=steps;
      }
      /**
       * @brief Get the number of presmoothing steps to apply
       * @return The number of steps:
       */
      std::size_t getNoPreSmoothSteps() const
      {
        return preSmoothSteps_;
      }

      /**
       * @brief Set the number of postsmoothing steps to apply
       * @param steps The number of steps:
       */
      void setNoPostSmoothSteps(std::size_t steps)
      {
        postSmoothSteps_=steps;
      }
      /**
       * @brief Get the number of postsmoothing steps to apply
       * @return The number of steps:
       */
      std::size_t getNoPostSmoothSteps() const
      {
        return postSmoothSteps_;
      }

      /**
       * @brief Set the value of gamma; 1 for V-cycle, 2 for W-cycle
       */
      void setGamma(std::size_t gamma)
      {
        gamma_=gamma;
      }
      /**
       * @brief Get the value of gamma; 1 for V-cycle, 2 for W-cycle
       */
      std::size_t getGamma() const
      {
        return gamma_;
      }

      /**
       * @brief Set whether to use additive multigrid.
       * @param additive True if multigrid should be additive.
       */
      void setAdditive(bool additive)
      {
        additive_=additive;
      }

      /**
       * @brief Get whether to use additive multigrid.
       * @return True if multigrid should be additive.
       */
      bool getAdditive() const
      {
        return additive_;
      }

      /**
       * @brief Constructor
       * @param maxLevel The maximum number of levels allowed in the matrix hierarchy (default: 100).
       * @param coarsenTarget If the number of nodes in the matrix is below this threshold the
       * coarsening will stop (default: 1000).
       * @param minCoarsenRate If the coarsening rate falls below this threshold the
       * coarsening will stop (default: 1.2)
       * @param prolongDamp The damping factor to apply to the prolongated update (default: 1.6)
       * @param accumulate Whether to accumulate the data onto fewer processors on coarser levels.
       */
      Parameters(int maxLevel=100, int coarsenTarget=1000, double minCoarsenRate=1.2,
                 double prolongDamp=1.6, AccumulationMode accumulate=successiveAccu)
        : CoarseningParameters(maxLevel, coarsenTarget, minCoarsenRate, prolongDamp, accumulate)
          , debugLevel_(2), preSmoothSteps_(2), postSmoothSteps_(2), gamma_(1),
          additive_(false)
      {}
    private:
      int debugLevel_;
      std::size_t preSmoothSteps_;
      std::size_t postSmoothSteps_;
      std::size_t gamma_;
      bool additive_;
    };

  } //namespace AMG

} //namespace Dune
#endif
