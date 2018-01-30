# Release 2.6

- Deprecated the preconditioner implementations `SeqILU0` and `SeqILUn`.
  Use `SeqILU` instead, which implements incomplete LU decomposition
  of any order.

- `BDMatrix` objects can now be constructed and assigned from `std::initializer_list`.

- `BDMatrix` and `BTDMatrix` now implement the `setSize` method, which allows to
  resize existing matrix objects.

