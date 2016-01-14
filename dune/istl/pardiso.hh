// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_ISTL_PARDISO_HH
#define DUNE_ISTL_PARDISO_HH

#include <dune/istl/preconditioners.hh>
#include <dune/istl/solvertype.hh>

#if HAVE_PARDISO
// PARDISO prototypes
extern "C" void pardisoinit(void *, int *, int *, int *, double *, int *);

extern "C" void pardiso(void *, int *, int *, int *, int *, int *,
                        double *, int *, int *, int *, int *, int *,
                        int *, double *, double *, int *, double *);

namespace Dune {


  /*! \brief The sequential Pardiso preconditioner.

     Put the Pardiso direct solver into the preconditioner framework.
   */
  template<class M, class X, class Y>
  class SeqPardiso : public Preconditioner<X,Y> {
  public:
    //! \brief The matrix type the preconditioner is for.
    typedef M matrix_type;
    //! \brief The domain type of the preconditioner.
    typedef X domain_type;
    //! \brief The range type of the preconditioner.
    typedef Y range_type;
    //! \brief The field type of the preconditioner.
    typedef typename X::field_type field_type;

    typedef typename M::RowIterator RowIterator;
    typedef typename M::ColIterator ColIterator;

    // define the category
    enum {
      //! \brief The category the preconditioner is part of
      category=SolverCategory::sequential
    };

    /*! \brief Constructor.

       Constructor gets all parameters to operate the prec.
       \param A The matrix to operate on.
     */
    SeqPardiso (const M& A)
      : A_(A)
    {
      mtype_ = 11;
      nrhs_ = 1;
      num_procs_ = 1;
      maxfct_ = 1;
      mnum_   = 1;
      msglvl_ = 0;
      error_  = 0;

      n_ = A_.N();
      int nnz = 0;
      RowIterator endi = A_.end();
      for (RowIterator i = A_.begin(); i != endi; ++i)
      {
        ColIterator endj = (*i).end();
        for (ColIterator j = (*i).begin(); j != endj; ++j) {
          if (j->size() != 1)
            DUNE_THROW(NotImplemented, "SeqPardiso: column blocksize != 1.");
          nnz++;
        }
      }

      std::cout << "dimension = " << n_ << ", number of nonzeros = " << nnz << std::endl;

      a_ = new double[nnz];
      ia_ = new int[n_+1];
      ja_ = new int[nnz];

      int count = 0;
      for (RowIterator i = A_.begin(); i != endi; ++i)
      {
        ia_[i.index()] = count+1;
        ColIterator endj = (*i).end();
        for (ColIterator j = (*i).begin(); j != endj; ++j) {
          a_[count] = *j;
          ja_[count] = j.index()+1;

          count++;
        }
      }
      ia_[n_] = count+1;

      pardisoinit(pt_,  &mtype_, &solver_, iparm_, dparm_, &error_);

      int phase = 11;
      int idum;
      double ddum;
      iparm_[2]  = num_procs_;

      pardiso(pt_, &maxfct_, &mnum_, &mtype_, &phase,
              &n_, a_, ia_, ja_, &idum, &nrhs_,
              iparm_, &msglvl_, &ddum, &ddum, &error_, dparm_);

      if (error_ != 0)
        DUNE_THROW(MathError, "Constructor SeqPardiso: Factorization failed. Error code " << error_);

      std::cout << "Constructor SeqPardiso: Factorization completed." << std::endl;
    }

    /*!
       \brief Prepare the preconditioner.

       \copydoc Preconditioner::pre(X&,Y&)
     */
    virtual void pre (X& x, Y& b) {}

    /*!
       \brief Apply the preconditioner.

       \copydoc Preconditioner::apply(X&,const Y&)
     */
    virtual void apply (X& v, const Y& d)
    {
      int phase = 33;

      iparm_[7] = 1;         /* Max numbers of iterative refinement steps. */
      int idum;

      double x[n_];
      double b[n_];
      for (int i = 0; i < n_; i++) {
        x[i] = v[i];
        b[i] = d[i];
      }

      pardiso(pt_, &maxfct_, &mnum_, &mtype_, &phase,
              &n_, a_, ia_, ja_, &idum, &nrhs_,
              iparm_, &msglvl_, b, x, &error_, dparm_);

      if (error_ != 0)
        DUNE_THROW(MathError, "SeqPardiso.apply: Backsolve failed. Error code " << error_);

      for (int i = 0; i < n_; i++)
        v[i] = x[i];

      std::cout << "SeqPardiso: Backsolve completed." << std::endl;
    }

    /*!
       \brief Clean up.

       \copydoc Preconditioner::post(X&)
     */
    virtual void post (X& x) {}

    ~SeqPardiso()
    {
      int phase = -1;                   // Release internal memory.
      int idum;
      double ddum;

      pardiso(pt_, &maxfct_, &mnum_, &mtype_, &phase,
              &n_, &ddum, ia_, ja_, &idum, &nrhs_,
              iparm_, &msglvl_, &ddum, &ddum, &error_, dparm_);
      delete[] a_;
      delete[] ia_;
      delete[] ja_;
    }

  private:
    M A_; //!< The matrix we operate on.
    int n_; //!< dimension of the system
    double *a_; //!< matrix values
    int *ia_; //!< indices to rows
    int *ja_; //!< column indices
    int mtype_; //!< matrix type, currently only 11 (real unsymmetric matrix) is supported
    int solver_; //!< solver method
    int nrhs_; //!< number of right hand sides
    void *pt_[64]; //!< internal solver memory pointer
    int iparm_[64]; //!< Pardiso integer control parameters
    double dparm_[64]; //!< Pardiso double control parameters
    int maxfct_;        //!< Maximum number of numerical factorizations
    int mnum_;  //!< Which factorization to use
    int msglvl_;    //!< flag to print statistical information
    int error_;      //!< error flag
    int num_procs_; //!< number of processors
  };

  template<class M, class X, class Y>
  struct IsDirectSolver<SeqPardiso<M,X,Y> >
  {
    enum { value=true};
  };
} // end namespace Dune

#endif //HAVE_PARDISO
#endif
