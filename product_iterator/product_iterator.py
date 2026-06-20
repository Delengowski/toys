import math
from collections import namedtuple
from collections.abc import Iterable, Iterator, Mapping
from types import MappingProxyType
from typing import Any, NamedTuple, Self

ProductItem = NamedTuple


class ProductIterator:
    """Stateful, splittable Cartesian product iterator.

    `ProductIterator` behaves like a labeled version of `itertools.product`,
    but supports stateful splitting. When split, outer dimensions are removed
    from the active iteration space and carried as fixed context.

    Example:
        >>> pi = ProductIterator(a=["x", "y"], b=[1, 2, 3])
        >>> list(pi)
        [ProductItem(a='x', b=1), ProductItem(a='x', b=2), ...]

    Example:
        Recursive splitting:
        >>> pi = ProductIterator(a=["x", "y"], b=[1, 2])
        >>> for sub_a in pi.yield_outer():
        ...     print(sub_a.context)
        ...     for sub_b in sub_a.yield_outer():
        ...         print(sub_b.context, list(sub_b))
        {'a': 'x'}
        {'a': 'x', 'b': 1} [ProductItem(a='x', b=1)]
        {'a': 'x', 'b': 2} [ProductItem(a='x', b=2)]
        {'a': 'y'}
        {'a': 'y', 'b': 1} [ProductItem(a='y', b=1)]
        {'a': 'y', 'b': 2} [ProductItem(a='y', b=2)]
    """

    def __init__(
        self,
        fixed: Mapping[str, Any] | None = None,
        last_fixed_key: str | None = None,
        **kwargs: Iterable[Any],
    ) -> None:
        """Initialize the product iterator.

        Args:
            fixed: Internal fixed context carried by child iterators.
            last_fixed_key: Internal key most recently fixed by `split`.
            **kwargs: Named dimensions to include in the product.

        Raises:
            ValueError: If no dimensions/context are provided, or if any
                iterable dimension is empty.

        Example:
            >>> pi = ProductIterator(a=["x", "y"], b=[1, 2])
            >>> next(iter(pi))
            ProductItem(a='x', b=1)
        """
        if not kwargs and not fixed:
            raise ValueError("Must provide at least one dimension")

        self._keys: list[str] = list(kwargs.keys())
        self._values: list[list[Any]] = [
            value if isinstance(value, list) else list(value)
            for value in kwargs.values()
        ]
        self._lengths: list[int] = [len(value) for value in self._values]

        if any(length == 0 for length in self._lengths):
            raise ValueError("All iterables must be non-empty")

        self._value_to_index: dict[str, dict[Any, int]] = {
            key: {value: i for i, value in enumerate(values)}
            for key, values in zip(self._keys, self._values, strict=False)
        }

        self._fixed: dict[str, Any] = dict(fixed or {})
        self._last_fixed_key: str | None = last_fixed_key

        self._indices: list[int] = [0] * len(self._keys)
        self._done: bool = False
        self._active_outer: bool = False
        self._tuple_type: type[tuple[Any, ...]] | None = None

    @property
    def context(self) -> Mapping[str, Any]:
        """Fixed values for this iterator context.

        Returns:
            A read-only mapping of fixed values.

        Example:
            >>> pi = ProductIterator(a=["x"], b=[1, 2])
            >>> sub = next(pi.yield_outer())
            >>> dict(sub.context)
            {'a': 'x'}
        """
        return MappingProxyType(self._fixed)

    @property
    def fixed(self) -> Mapping[str, Any]:
        """Alias for `context`.

        Returns:
            A read-only mapping of fixed values.

        Example:
            >>> pi = ProductIterator(a=["x"], b=[1])
            >>> sub = next(pi.yield_outer())
            >>> sub.fixed["a"]
            'x'
        """
        return self.context

    @property
    def current_key(self) -> str | None:
        """Most recently fixed dimension key.

        Returns:
            The most recently fixed key, or `None` for the root iterator.

        Example:
            >>> pi = ProductIterator(a=["x"], b=[1])
            >>> sub = next(pi.yield_outer())
            >>> sub.current_key
            'a'
        """
        return self._last_fixed_key

    @property
    def current_value(self) -> Any | None:
        """Most recently fixed dimension value.

        Returns:
            The most recently fixed value, or `None` for the root iterator.

        Example:
            >>> pi = ProductIterator(a=["x"], b=[1])
            >>> sub = next(pi.yield_outer())
            >>> sub.current_value
            'x'
        """
        if self._last_fixed_key is None:
            return None
        return self._fixed[self._last_fixed_key]

    def _make_tuple(self, active: Mapping[str, Any]) -> tuple[Any, ...]:
        """Create a namedtuple result from fixed and active values.

        Args:
            active: Active dimension values for one product item.

        Returns:
            A namedtuple containing fixed and active values.
        """
        full = {**self._fixed, **active}

        if self._tuple_type is None:
            self._tuple_type = namedtuple("ProductItem", full.keys())

        return self._tuple_type(**full)

    def _current(self) -> tuple[Any, ...]:
        """Return the current product item.

        Returns:
            The current product item as a namedtuple.
        """
        active = {
            key: self._values[i][idx]
            for i, (key, idx) in enumerate(zip(self._keys, self._indices, strict=False))
        }
        return self._make_tuple(active)

    def _advance(self) -> None:
        """Advance the internal state by one product item."""
        if self._done:
            return

        for i in reversed(range(len(self._indices))):
            self._indices[i] += 1

            if self._indices[i] < self._lengths[i]:
                for j in range(i + 1, len(self._indices)):
                    self._indices[j] = 0
                return

            self._indices[i] = 0

        self._done = True

    def __iter__(self) -> Iterator[tuple[Any, ...]]:
        """Iterate over remaining product items.

        Yields:
            Namedtuple product items.

        Raises:
            RuntimeError: If a split iterator is currently active.

        Example:
            >>> pi = ProductIterator(a=["x"], b=[1, 2])
            >>> list(pi)
            [ProductItem(a='x', b=1), ProductItem(a='x', b=2)]
        """
        if self._active_outer:
            raise RuntimeError("Cannot iterate while split is active")

        while not self._done:
            yield self._current()
            self._advance()

    def reset(self) -> None:
        """Reset this iterator back to the beginning.

        Raises:
            RuntimeError: If a split iterator is currently active.

        Example:
            >>> pi = ProductIterator(a=["x"], b=[1])
            >>> list(pi)
            [ProductItem(a='x', b=1)]
            >>> pi.reset()
            >>> list(pi)
            [ProductItem(a='x', b=1)]
        """
        if self._active_outer:
            raise RuntimeError("Cannot reset while split is active")

        self._indices = [0 for _ in self._keys]
        self._done = False

    def yield_outer(self) -> Iterator[Self]:
        """Split on the current outermost active dimension.

        Returns:
            An iterator of child `ProductIterator` objects.

        Example:
            >>> pi = ProductIterator(a=["x", "y"], b=[1, 2])
            >>> sub = next(pi.yield_outer())
            >>> dict(sub.context)
            {'a': 'x'}
            >>> list(sub)
            [ProductItem(a='x', b=1), ProductItem(a='x', b=2)]
        """
        return self.split(depth=1)

    def split(self, depth: int = 1) -> Iterator[Self]:
        """Split by fixing the first `depth` active dimensions.

        Args:
            depth: Number of active dimensions to fix.

        Yields:
            Child `ProductIterator` instances with fixed context expanded and
            those dimensions removed from the active iteration space.

        Raises:
            ValueError: If `depth` is invalid.
            RuntimeError: If another split is already active.

        Example:
            >>> pi = ProductIterator(a=["x"], b=[1, 2], c=["p", "q"])
            >>> sub = next(pi.split(depth=2))
            >>> dict(sub.context)
            {'a': 'x', 'b': 1}
            >>> list(sub)
            [ProductItem(a='x', b=1, c='p'), ProductItem(a='x', b=1, c='q')]
        """
        if depth <= 0 or depth > len(self._keys):
            raise ValueError("Invalid depth")

        if self._active_outer:
            raise RuntimeError("Split already active")

        if self._done:
            return iter(())

        self._active_outer = True

        def generator() -> Iterator[Self]:
            try:
                while not self._done:
                    prefix_indices = self._indices[:depth]

                    new_fixed = dict(self._fixed)
                    new_kwargs: dict[str, Iterable[Any]] = {}

                    for i, key in enumerate(self._keys):
                        if i < depth:
                            new_fixed[key] = self._values[i][prefix_indices[i]]
                        else:
                            new_kwargs[key] = self._values[i]

                    yield ProductIterator(
                        _fixed=new_fixed,
                        _last_fixed_key=self._keys[depth - 1],
                        **new_kwargs,
                    )

                    while not self._done and self._indices[:depth] == prefix_indices:
                        self._advance()

            finally:
                self._active_outer = False

        return generator()

    def size(self) -> int:
        """Return the number of active combinations.

        Fixed context values do not contribute to size.

        Returns:
            Number of combinations remaining in this iterator's active space.

        Example:
            >>> ProductIterator(a=["x", "y"], b=[1, 2, 3]).size()
            6
        """
        return math.prod(self._lengths) if self._lengths else 1

    def _ravel_index(self, indices: Iterable[int]) -> int:
        """Convert a multi-index into a flat product index.

        Args:
            indices: Per-dimension integer indices.

        Returns:
            Flat product index.

        Example:
            For lengths `[2, 3, 2]`, index `[0, 1, 1]` becomes `3`.
        """
        idx = 0
        indices_list = list(indices)

        if len(indices_list) != len(self._lengths):
            raise ValueError("Incorrect number of indices")

        for i, value in enumerate(indices_list):
            if value < 0 or value >= self._lengths[i]:
                raise IndexError("Index out of bounds")

            stride = math.prod(self._lengths[i + 1 :])
            idx += value * stride

        return idx

    def index_of(self, **kwargs: Any) -> int:
        """Compute the flat product index for active dimension values.

        Args:
            **kwargs: Values for each active dimension.

        Returns:
            Flat index relative to this iterator's active product space.

        Raises:
            KeyError: If a required active key is missing.
            ValueError: If extra keys are provided.
            KeyError: If a value is not present in its dimension.

        Example:
            >>> pi = ProductIterator(a=["x", "y"], b=[1, 2, 3])
            >>> pi.index_of(a="x", b=2)
            1
            >>> pi.index_of(a="y", b=1)
            3
        """
        expected = set(self._keys)
        received = set(kwargs)

        if received != expected:
            missing = expected - received
            extra = received - expected
            raise ValueError(
                f"Expected keys {expected}; missing={missing}, extra={extra}"
            )

        indices = [self._value_to_index[key][kwargs[key]] for key in self._keys]

        return self._ravel_index(indices)

    def _unravel_index(self, idx: int) -> list[int]:
        """Convert a flat product index into a multi-index.

        Args:
            idx: Flat product index.

        Returns:
            Per-dimension integer indices.

        Raises:
            IndexError: If `idx` is out of bounds.

        Example:
            >>> pi = ProductIterator(a=["x", "y"], b=[1, 2, 3])
            >>> pi._unravel_index(4)
            [1, 1]
        """
        total = self.size()

        if idx < 0 or idx >= total:
            raise IndexError("Flat index out of bounds")

        indices: list[int] = []

        for length in reversed(self._lengths):
            indices.append(idx % length)
            idx //= length

        return list(reversed(indices))

    def chunk(self, chunk_size: int) -> Iterator["ProductSlice"]:
        """Split the active product space into fixed-size slices.

        Args:
            chunk_size: Maximum number of items per slice.

        Yields:
            `ProductSlice` objects.

        Raises:
            ValueError: If `chunk_size <= 0`.

        Example:
            >>> pi = ProductIterator(a=["x", "y"], b=[1, 2, 3])
            >>> [list(chunk) for chunk in pi.chunk(2)]
            [[ProductItem(a='x', b=1), ProductItem(a='x', b=2)], ...]
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")

        total = self.size()

        for start in range(0, total, chunk_size):
            yield ProductSlice(self, start, min(start + chunk_size, total))

    def partition(self, k: int) -> Iterator["ProductSlice"]:
        """Partition the active product space into up to `k` slices.

        Args:
            k: Number of desired partitions.

        Yields:
            `ProductSlice` objects with roughly equal sizes.

        Raises:
            ValueError: If `k <= 0`.

        Example:
            >>> pi = ProductIterator(a=["x", "y"], b=[1, 2, 3])
            >>> parts = list(pi.partition(3))
            >>> [list(part) for part in parts]
            [[ProductItem(a='x', b=1), ProductItem(a='x', b=2)], ...]
        """
        if k <= 0:
            raise ValueError("k must be > 0")

        total = self.size()
        chunk_size = (total + k - 1) // k

        for i in range(k):
            start = i * chunk_size
            end = min(start + chunk_size, total)

            if start >= total:
                break

            yield ProductSlice(self, start, end)


class ProductSlice:
    """Stateless slice of a `ProductIterator`.

    A `ProductSlice` represents a range of flat product indices. It does not
    mutate the parent iterator's state.

    Example:
        >>> pi = ProductIterator(a=["x", "y"], b=[1, 2, 3])
        >>> slice_ = ProductSlice(pi, 1, 3)
        >>> list(slice_)
        [ProductItem(a='x', b=2), ProductItem(a='x', b=3)]
    """

    def __init__(self, parent: ProductIterator, start: int, end: int) -> None:
        """Initialize the product slice.

        Args:
            parent: Source product iterator.
            start: Inclusive flat start index.
            end: Exclusive flat end index.

        Raises:
            ValueError: If the slice bounds are invalid.
        """
        if start < 0 or end < start or end > parent.size():
            raise ValueError("Invalid slice bounds")

        self._parent: ProductIterator = parent
        self._start: int = start
        self._end: int = end

    def __iter__(self) -> Iterator[tuple[Any, ...]]:
        """Iterate over this slice.

        Yields:
            Namedtuple product items.

        Example:
            >>> pi = ProductIterator(a=["x"], b=[1, 2])
            >>> list(ProductSlice(pi, 0, 1))
            [ProductItem(a='x', b=1)]
        """
        for idx in range(self._start, self._end):
            multi_idx = self._parent._unravel_index(idx)

            active = {
                self._parent._keys[i]: self._parent._values[i][multi_idx[i]]
                for i in range(len(multi_idx))
            }

            yield self._parent._make_tuple(active)
