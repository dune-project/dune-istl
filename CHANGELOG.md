# Master (will become release 2.8)

- Added public access of the `cholmod_common` object in class `Cholmod`.

- Python bindings have been moved from the `dune-python` module which is now
  obsolete.
  Note that for `dune-istl` bindings are still very much work in progress.
  To activate Python bindings the CMake flag
  `DUNE_ENABLE_PYTHONBINDINGS` needs to be turned on (default is off).
  Furthermore, flags for either shared library or position independent code
  needs to be used.

- Added new utility functions templates `maxBlockLevel`, `minBlockLevel`, `blockLevel` in `dnue/istl/blocklevel.hh` to
  automatically determine the block level of possibly nested ISTL vectors & matrices at compile time.
  The block level cannot be always uniquely determined for `MultiTypeBlockMatrix`/`MultiTypeBlockVector` since the nesting level
  of different block types might differ. Hence, `maxBlockLevel`, `minBlockLevel` always works
  but if they yield different results `blockLevel` will not compile.
  This condition can be checked with the function template `hasUniqueBlockLevel`.

## Deprecations and removals
- Drop deprecated bindings of direct solver Pardiso.

- Remove deprecated preconditioner implementations `SeqILU0` and `SeqILUn`. Use
  `SeqILU` instead, which implements incomplete LU decomposition of any order.

- Remove deprecated methods 'BlockVector::resize' and 'BlockVecotor::reserve'
  with two arguments.

- Drop support SuperLU 4.

- Rename the exception `ImplicitModeOverflowExhausted` to `ImplicitModeCompressionBufferExhausted`,
  to better reflect its meaning.  The old exception is still there, but it triggers
  a deprecation warning.

# Release 2.7

- New `SolverFactory` for generating sequential direct or iterative solvers and
  preconditioners from a `ParameterTree` configuration.

- `BDMatrix` objects now have the method `solve`, which implements that
  canonical way to solve block-diagonal linear systems.

- The class `VariableBlockVector::CreateIterator` is a true STL output iterator now.
  This means that you can use STL algorithms like `std::fill` or `std::copy`
  to set the block sizes.

- `MultiTypeBlockVector<Args...>` now inherits the constructors from its
  parent type (`std::tuple<Args...>`). This means you can now also construct
  `MultiTypeBlockVector`s from values or references of BlockVectors.

- All matrix and vector classes can now be instantiated with number types
  directly (A number type is any type for which `Dune::IsNumber<T>::value`
  is true).  For example, you can now use `BlockVector<double>` instead of
  the more cumbersome `BlockVector<FieldVector<double,1> >`.  Similarly, you can use
  `BCRSMatrix<double>` instead of `BCRSMatrix<FieldMatrix<double,1,1>>`.
  The old forms still work, and `FieldVector` and `FieldMatrix` types with
  a single entry can still be cast to their `field_type`.  Therefore, the
  change is completely backward-compatible.

- Added a right-preconditioned flexible restarted GMRes solver

- The UMFPack binding use the long int functions to compute larger systems.
  With the \*_dl_\* versions instead of the \*_di_\* versions UMFPACK will not
  have a memory limit of just 2 GiB.

- Support for SuiteSparse's CHOLMOD providing a sparse Cholesky
  factorization.

- The interface methods `dot()` and `norm()` of ScalarProduct are now `const`. You will
  have to adjust the method signatures in your own scalar product implementations.

- `MultiTypeBlockVector` now implements the interface method `N()`, which
  returns the number of vector entries.

- `MultiTypeBlockVector` now implements the interface method `dim()`, which
  returns the number of scalar vector entries.

- `MultiTypeBlockVector::count()` is now `const`

- `SeqILU` can now be used with SIMD data types.

## Deprecations and removals

- Deprecated support for SuperLU 4.x. It will be removed after Dune 2.7.

- Deprecated the preconditioner implementations `SeqILU0` and `SeqILUn`.
  Use `SeqILU` instead, which implements incomplete LU decomposition
  of any order.

- The method `setSolverCategory` of `OwnerOverlapCopyCommunication` is deprecated and
  will be removed after Dune 2.7. The solver category can only be set in the constructor.

- The method `getSolverCategory` of `OwnerOverlapCopyCommunication` is deprecated and
  will be removed after Dune 2.7. Use `category()` instead.

- The method `MultiTypeBlockVector::count()` has been deprecated, because its name
  is inconsistent with the name mandated by the `dune-istl` vector interface.

- The method `MultiTypeBlockMatrix::size()` has been deprecated, because its name
  is inconsistent with the name mandated by the `dune-istl` vector interface.

# Release 2.6

- `BDMatrix` objects can now be constructed and assigned from `std::initializer_list`.

- `BDMatrix` and `BTDMatrix` now implement the `setSize` method, which allows to
  resize existing matrix objects.

- The solver infrastructure was updated to support SIMD data types (see
  current changes in `dune-common`). This allows to solve multiple systems
  simultaniously using vectorization.
