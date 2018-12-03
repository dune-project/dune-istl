# Master (will become release 2.7)

- Deprecated the preconditioner implementations `SeqILU0` and `SeqILUn`.
  Use `SeqILU` instead, which implements incomplete LU decomposition
  of any order.

- The class `VariableBlockVector::CreateIterator` is a true STL output iterator now.
  This means that you can use STL algorithms like `std::fill` or `std::copy`
  to set the block sizes.

- Support for SuiteSparse's CHOLMOD providing a sparse Cholesky
  factorization.

- `MultiTypeBlockVector<Args...>` now inherits the constructors from its
  parent type (`std::tuple<Args...>`). This means you can now also construct
  `MultiTypeBlockVector`s from values or references of BlockVectors.

- `MultiTypeBlockVector::count()` is now `const`

- All matrix and vector classes can now be instantiated with number types
  directly (A number type is any type for which `Dune::IsNumber<T>::value`
  is true).  For example, you can now use `BlockVector<double>` instead of
  the more cumbersome `BlockVector<FieldVector<double,1> >`.  Similarly, you can use
  `BCRSMatrix<double>` instead of `BCRSMatrix<FieldMatrix<double,1,1>>`.
  The old forms still work, and `FieldVector` and `FieldMatrix` types with
  a single entry can still be cast to their `field_type`.  Therefore, the
  change is completely backward-compatible.

# Release 2.6

- `BDMatrix` objects can now be constructed and assigned from `std::initializer_list`.

- `BDMatrix` and `BTDMatrix` now implement the `setSize` method, which allows to
  resize existing matrix objects.

- The solver infrastructure was updated to support SIMD data types (see
  current changes in `dune-common`). This allows to solve multiple systems
  simultaniously using vectorization.
