# Master (will become release 2.7)

- Deprecated the preconditioner implementations `SeqILU0` and `SeqILUn`.
  Use `SeqILU` instead, which implements incomplete LU decomposition
  of any order.

- The vector class `BlockVector` can now be instantiated with number types
  directly.  For example, you can now use `BlockVector<double>` instead of
  the more cumbersome `BlockVector<FieldVector<double,1> >` (The latter
  still works, though).  A number type is any type for which `Dune::IsNumber<T>::value`
  is true.

# Release 2.6

- `BDMatrix` objects can now be constructed and assigned from `std::initializer_list`.

- `BDMatrix` and `BTDMatrix` now implement the `setSize` method, which allows to
  resize existing matrix objects.

- The solver infrastructure was updated to support SIMD data types (see
  current changes in `dune-common`). This allows to solve multiple systems
  simultaniously using vectorization.
