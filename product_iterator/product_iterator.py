import math
from collections import namedtuple
from collections.abc import Iterable, Iterator, Mapping
from types import MappingProxyType
from typing import Any, Self


class ProductIterator:
    """Stateful, splittable Cartesian product iterator.

    This class behaves like a labeled `itertools.product`, but supports:

    * namedtuple output
    * fixed context from previous splits
    * recursive outer splitting
    * flat index lookup
    * chunking and partitioning through `ProductIterator` views

    Example:
        >>> pi = ProductIterator(a=["x", "y"], b=[1, 2])
        >>> list(pi)
        [ProductItem(a='x', b=1), ProductItem(a='x', b=2),
         ProductItem(a='y', b=1), ProductItem(a='y', b=2)]

    Example:
        Recursive splitting:

        >>> pi = ProductIterator(a=["x", "y"], b=[1, 2])
        >>> for sub_a in pi.yield_outer():
        ...     print(dict(sub_a.context))
        ...     for sub_b in sub_a.yield_outer():
        ...         print(dict(sub_b.context), list(sub_b))
        {'a': 'x'}
        {'a': 'x', 'b': 1} [ProductItem(a='x', b=1)]
        {'a': 'x', 'b': 2} [ProductItem(a='x', b=2)]
        {'a': 'y'}
        {'a': 'y', 'b': 1} [ProductItem(a='y', b=1)]
        {'a': 'y', 'b': 2} [ProductItem(a='y', b=2)]
    """

    def __init__(
        self,
        _fixed: Mapping[str, Any] | None = None,
        _last_fixed_key: str | None = None,
        _start: int = 0,
        _stop: int | None = None,
        **kwargs: Iterable[Any],
    ) -> None:
        """Initialize the product iterator.

        Args:
            _fixed: Internal fixed context carried by split/view children.
            _last_fixed_key: Internal key most recently fixed by a split.
            _start: Internal inclusive flat start index for this view.
            _stop: Internal exclusive flat stop index for this view.
            **kwargs: Named iterable dimensions.

        Raises:
            ValueError: If no dimensions/context are provided, if any dimension
                is empty, or if the flat view bounds are invalid.

        Example:
            >>> pi = ProductIterator(a=["x", "y"], b=[1, 2])
            >>> next(iter(pi))
            ProductItem(a='x', b=1)
        """
        if not kwargs and not _fixed:
            raise ValueError("Must provide at least one dimension")

        self._keys: list[str] = list(kwargs.keys())
        self._values: list[list[Any]] = [
            value if isinstance(value, list) else list(value)
            for value in kwargs.values()
        ]
        self._lengths: list[int] = [len(value) for value in self._values]

        if any(length == 0 for length in self._lengths):
            raise ValueError("All iterables must be non-empty")

        self._fixed: dict[str, Any] = dict(_fixed or {})
        self._last_fixed_key: str | None = _last_fixed_key

        self._value_to_index: dict[str, dict[Any, int]] = {
            key: {value: i for i, value in enumerate(values)}
            for key, values in zip(self._keys, self._values, strict=False)
        }

        self._tuple_type: type[tuple[Any, ...]] | None = None
        self._active_split: bool = False

        total = self._full_size()
        stop = total if _stop is None else _stop

        if _start < 0 or stop < _start or stop > total:
            raise ValueError("Invalid view bounds")

        self._start: int = _start
        self._stop: int = stop
        self._pos: int = _start

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

    @property
    def start(self) -> int:
        """Inclusive flat start index for this iterator view.

        Returns:
            The inclusive flat start index.

        Example:
            >>> pi = ProductIterator(a=["x", "y"], b=[1, 2])
            >>> chunk = next(pi.chunk(2))
            >>> chunk.start
            0
        """
        return self._start

    @property
    def stop(self) -> int:
        """Exclusive flat stop index for this iterator view.

        Returns:
            The exclusive flat stop index.

        Example:
            >>> pi = ProductIterator(a=["x", "y"], b=[1, 2])
            >>> chunk = next(pi.chunk(2))
            >>> chunk.stop
            2
        """
        return self._stop

    @property
    def pos(self) -> int:
        """Current flat cursor position for this iterator view.

        Returns:
            The current flat cursor position.

        Example:
            >>> pi = ProductIterator(a=["x"], b=[1, 2])
            >>> pi.pos
            0
        """
        return self._pos

    def _full_size(self) -> int:
        """Return the total active product size, ignoring view bounds.

        Returns:
            Total number of active combinations.
        """
        return math.prod(self._lengths) if self._lengths else 1

    @property
    def size(self) -> int:
        """Return the size of this iterator view.

        Fixed context values do not contribute to size.

        Returns:
            Number of items in this iterator's flat view.

        Example:
            >>> ProductIterator(a=["x", "y"], b=[1, 2, 3]).size()
            6
            >>> next(ProductIterator(a=["x", "y"], b=[1, 2, 3]).chunk(2)).size()
            2
        """
        return self._stop - self._start

    @property
    def remaining(self) -> int:
        """Return the number of items remaining from the current cursor.

        Returns:
            Number of unconsumed items in this iterator view.

        Example:
            >>> pi = ProductIterator(a=["x"], b=[1, 2])
            >>> next(iter(pi))
            ProductItem(a='x', b=1)
            >>> pi.remaining()
            1
        """
        return self._stop - self._pos

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

    def _unravel_index(self, idx: int) -> list[int]:
        """Convert a flat product index into a multi-index.

        Args:
            idx: Flat product index relative to the full active product space.

        Returns:
            Per-dimension integer indices.

        Raises:
            IndexError: If `idx` is out of bounds.

        Example:
            >>> pi = ProductIterator(a=["x", "y"], b=[1, 2, 3])
            >>> pi._unravel_index(4)
            [1, 1]
        """
        total = self._full_size()

        if idx < 0 or idx >= total:
            raise IndexError("Flat index out of bounds")

        indices: list[int] = []

        for length in reversed(self._lengths):
            indices.append(idx % length)
            idx //= length

        return list(reversed(indices))

    def _ravel_index(self, indices: Iterable[int]) -> int:
        """Convert a multi-index into a flat product index.

        Args:
            indices: Per-dimension integer indices.

        Returns:
            Flat product index relative to the full active product space.

        Raises:
            ValueError: If the number of indices is wrong.
            IndexError: If any dimension index is out of bounds.

        Example:
            For dimensions with lengths `[2, 3, 2]`, index `[0, 1, 1]`
            becomes `3`.
        """
        indices_list = list(indices)

        if len(indices_list) != len(self._lengths):
            raise ValueError("Incorrect number of indices")

        idx = 0

        for i, value in enumerate(indices_list):
            if value < 0 or value >= self._lengths[i]:
                raise IndexError("Index out of bounds")

            stride = math.prod(self._lengths[i + 1 :])
            idx += value * stride

        return idx

    def item_at(self, idx: int) -> tuple[Any, ...]:
        """Return the product item at a flat index.

        Args:
            idx: Flat index relative to the full active product space.

        Returns:
            Namedtuple product item.

        Raises:
            IndexError: If `idx` is out of bounds.

        Example:
            >>> pi = ProductIterator(a=["x", "y"], b=[1, 2, 3])
            >>> pi.item_at(4)
            ProductItem(a='y', b=2)
        """
        multi_idx = self._unravel_index(idx)

        active = {
            self._keys[i]: self._values[i][multi_idx[i]] for i in range(len(multi_idx))
        }

        return self._make_tuple(active)

    @property
    def current(self) -> tuple[Any, ...]:
        """Return the current item without advancing.

        Returns:
            Current namedtuple product item.

        Raises:
            StopIteration: If this iterator view is exhausted.

        Example:
            >>> pi = ProductIterator(a=["x"], b=[1, 2])
            >>> pi.current()
            ProductItem(a='x', b=1)
        """
        if self._pos >= self._stop:
            raise StopIteration

        return self.item_at(self._pos)

    def __iter__(self) -> Iterator[tuple[Any, ...]]:
        """Iterate over remaining product items in this view.

        Yields:
            Namedtuple product items.

        Raises:
            RuntimeError: If a stateful split is currently active.

        Example:
            >>> pi = ProductIterator(a=["x"], b=[1, 2])
            >>> list(pi)
            [ProductItem(a='x', b=1), ProductItem(a='x', b=2)]
        """
        if self._active_split:
            raise RuntimeError("Cannot iterate while split is active")

        while self._pos < self._stop:
            yield self.item_at(self._pos)
            self._pos += 1

    def reset(self) -> None:
        """Reset this iterator view back to its start index.

        Raises:
            RuntimeError: If a stateful split is currently active.

        Example:
            >>> pi = ProductIterator(a=["x"], b=[1])
            >>> list(pi)
            [ProductItem(a='x', b=1)]
            >>> pi.reset()
            >>> list(pi)
            [ProductItem(a='x', b=1)]
        """
        if self._active_split:
            raise RuntimeError("Cannot reset while split is active")

        self._pos = self._start

    def view(self, start: int, stop: int) -> Self:
        """Create an independent bounded view over this iterator's active space.

        The returned iterator has its own cursor. Consuming the view does not
        mutate this iterator.

        Args:
            start: Inclusive flat start index relative to this iterator's full
                active product space.
            stop: Exclusive flat stop index relative to this iterator's full
                active product space.

        Returns:
            A new `ProductIterator` with the same active dimensions and fixed
            context, bounded to `[start, stop)`.

        Raises:
            ValueError: If the view bounds are invalid.

        Example:
            >>> pi = ProductIterator(a=["x", "y"], b=[1, 2])
            >>> list(pi.view(1, 3))
            [ProductItem(a='x', b=2), ProductItem(a='y', b=1)]
        """
        return ProductIterator(
            _fixed=self._fixed,
            _last_fixed_key=self._last_fixed_key,
            _start=start,
            _stop=stop,
            **dict(zip(self._keys, self._values, strict=False)),
        )

    def index_of(self, **kwargs: Any) -> int:
        """Compute the flat product index for active dimension values.

        Args:
            **kwargs: Values for each active dimension.

        Returns:
            Flat index relative to this iterator's full active product space.

        Raises:
            ValueError: If required keys are missing or extra keys are provided.
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

        Split children remove the fixed dimensions from their active iteration
        space and carry them as context. This means repeated `yield_outer()`
        calls naturally progress deeper through the original dimensions.

        Args:
            depth: Number of active dimensions to fix.

        Yields:
            Child `ProductIterator` instances.

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

        if self._active_split:
            raise RuntimeError("Split already active")

        if self._pos >= self._stop:
            return iter(())

        self._active_split = True

        def generator() -> Iterator[Self]:
            try:
                while self._pos < self._stop:
                    prefix_indices = self._unravel_index(self._pos)[:depth]

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

                    while self._pos < self._stop:
                        current_prefix = self._unravel_index(self._pos)[:depth]
                        if current_prefix != prefix_indices:
                            break
                        self._pos += 1

            finally:
                self._active_split = False

        return generator()

    def chunk(self, chunk_size: int) -> Iterator[Self]:
        """Split this view into fixed-size independent iterator views.

        Args:
            chunk_size: Maximum number of items per chunk.

        Yields:
            Independent `ProductIterator` views.

        Raises:
            ValueError: If `chunk_size <= 0`.

        Example:
            >>> pi = ProductIterator(a=["x", "y"], b=[1, 2, 3])
            >>> [list(chunk) for chunk in pi.chunk(2)]
            [[ProductItem(a='x', b=1), ProductItem(a='x', b=2)],
             [ProductItem(a='x', b=3), ProductItem(a='y', b=1)],
             [ProductItem(a='y', b=2), ProductItem(a='y', b=3)]]
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")

        for start in range(self._start, self._stop, chunk_size):
            stop = min(start + chunk_size, self._stop)
            yield self.view(start, stop)

    def partition(self, k: int) -> Iterator[Self]:
        """Partition this view into up to `k` independent iterator views.

        Args:
            k: Number of desired partitions.

        Yields:
            Independent `ProductIterator` views with roughly equal sizes.

        Raises:
            ValueError: If `k <= 0`.

        Example:
            >>> pi = ProductIterator(a=["x", "y"], b=[1, 2, 3])
            >>> [list(part) for part in pi.partition(3)]
            [[ProductItem(a='x', b=1), ProductItem(a='x', b=2)],
             [ProductItem(a='x', b=3), ProductItem(a='y', b=1)],
             [ProductItem(a='y', b=2), ProductItem(a='y', b=3)]]
        """
        if k <= 0:
            raise ValueError("k must be > 0")

        total = self.size
        chunk_size = (total + k - 1) // k

        for i in range(k):
            start = self._start + i * chunk_size
            stop = min(start + chunk_size, self._stop)

            if start >= self._stop:
                break

            yield self.view(start, stop)
