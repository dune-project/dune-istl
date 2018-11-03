# Master (will become release 2.7)

- Deprecated the preconditioner implementations `SeqILU0` and `SeqILUn`.
  Use `SeqILU` instead, which implements incomplete LU decomposition
  of any order.

- The class `VariableBlockVector::CreateIterator` is a true STL output iterator now.
  This means that you can use STL algorithms like `std::fill` or `std::copy`
  to set the block sizes.

- Support for SuiteSparse's CHOLMOD providing a sparse Cholesky
  factorization.

# Release 2.6

- `BDMatrix` objects can now be constructed and assigned from `std::initializer_list`.

- `BDMatrix` and `BTDMatrix` now implement the `setSize` method, which allows to
  resize existing matrix objects.

- The solver infrastructure was updated to support SIMD data types (see
  current changes in `dune-common`). This allows to solve multiple systems
  simultaniously using vectorization.
