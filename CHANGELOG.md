# Release 2.6

- `BDMatrix` objects can now be constructed and assigned from `std::initializer_list`.

- `BDMatrix` and `BTDMatrix` now implement the `setSize` method, which allows to
  resize existing matrix objects.

- The solver infrastructure was updated to support SIMD data types (see
  current changes in `dune-common`). This allows to solve multiple systems
  simultaniously using vectorization.

- `MultiTypeBlockVector<Args...>` now inherits the constructors from its
parent type (`std::tuple<Args...>`). This means you can now also construct
`MultiTypeBlockVector`s from values or references of BlockVectors.

- `MultiTypeBlockVector::count()` is now `const`

- `MultiTypeBlockMatrix` now implements the methods `frobenius_norm` and `frobenius_norm2`.
