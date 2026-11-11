"""Runtime validation helpers for backend-neutral layout transfers."""

import math

import sympy


def tensor_memory_span(value):
    """Return the conservative byte span occupied by a tensor-like value."""
    try:
        device = value.device
    except (AttributeError, RuntimeError, TypeError):
        device = None

    try:
        shape = tuple(value.shape)

        if any(size == 0 for size in shape):
            return (device, 0, 0)

        element_size = value.element_size()
        data_ptr = value.data_ptr()
        lower = upper = 0

        for size, stride in zip(shape, value.stride()):
            extent = (size - 1) * stride
            lower += min(0, extent)
            upper += max(0, extent)

        return (
            device,
            data_ptr + lower * element_size,
            data_ptr + (upper + 1) * element_size,
        )
    except (AttributeError, RuntimeError, TypeError):
        return (device, None, None)


def memory_spans_overlap(first, second):
    """Conservatively determine whether two tensor byte spans overlap."""
    first_device, first_start, first_end = first
    second_device, second_start, second_end = second

    if (
        first_device is not None
        and second_device is not None
        and first_device != second_device
    ):
        return False

    if None in (first_start, first_end, second_start, second_end):
        return True

    if first_start == first_end or second_start == second_end:
        return False

    return first_start < second_end and second_start < first_end


def build_layout_transfer_validator(metadata, abi):
    """Build a runtime validator from serialized layout-transfer metadata."""
    transfer = dict(metadata.get("layout_transfer", {}))

    if not transfer:
        return None

    permutation = tuple(int(axis) for axis in transfer.get("permutation", ()))
    physical_constraints = tuple(
        (sympy.sympify(str(left)), sympy.sympify(str(right)))
        for left, right in transfer.get("physical_constraints", ())
    )
    program_constraints = tuple(
        (
            tuple(sympy.sympify(str(value)) for value in actual),
            tuple(sympy.sympify(str(value)) for value in expected),
        )
        for actual, expected in transfer.get("program_constraints", ())
    )
    value_constraints = tuple(
        (
            tuple(sympy.sympify(str(value)) for value in actual),
            tuple(sympy.sympify(str(value)) for value in expected),
        )
        for actual, expected in transfer.get("value_constraints", ())
    )
    expressions = (
        *(expression for pair in physical_constraints for expression in pair),
        *(
            expression
            for constraint in value_constraints
            for shape in constraint
            for expression in shape
        ),
        *(
            expression
            for constraint in program_constraints
            for shape in constraint
            for expression in shape
        ),
    )
    bindings = {binding.name: binding for binding in abi.kernel_args}

    try:
        symbols = tuple(
            (symbol, bindings[str(symbol)])
            for symbol in sorted(
                set().union(*(expression.free_symbols for expression in expressions)),
                key=str,
            )
        )
        source_binding = bindings[str(transfer["source_binding"])]
        destination_binding = bindings[str(transfer["destination_binding"])]
    except KeyError as exc:
        raise ValueError(
            f"Serialized layout transfer references unknown binding `{exc.args[0]}`."
        ) from exc

    from ninetoothed.compiler.runtime import _binding_value

    def validate(public):
        source = _binding_value(source_binding, public)
        destination = _binding_value(destination_binding, public)

        if (
            len(permutation) != len(source.shape)
            or len(permutation) != len(destination.shape)
            or set(permutation) != set(range(len(permutation)))
            or any(
                source.shape[source_axis] != destination.shape[destination_axis]
                for destination_axis, source_axis in enumerate(permutation)
            )
        ):
            raise ValueError("Layout transfer physical shapes do not match.")

        if not _tensor_has_non_overlapping_strides(destination):
            raise ValueError(
                "Layout transfer requires non-overlapping destination strides."
            )

        substitutions = {
            symbol: _binding_value(binding, public) for symbol, binding in symbols
        }

        def resolve(expression):
            resolved = expression.subs(substitutions)

            if resolved.free_symbols:
                names = ", ".join(sorted(map(str, resolved.free_symbols)))
                raise ValueError(f"Unresolved layout transfer symbols: {names}.")
            return int(resolved)

        for left, right in physical_constraints:
            if resolve(left) != resolve(right):
                raise ValueError("Layout transfer physical shapes do not match.")

        for actual, expected in value_constraints:
            if len(actual) != len(expected) or any(
                resolve(left) != resolve(right) for left, right in zip(actual, expected)
            ):
                raise ValueError("Layout transfer value domains do not match.")

        for actual, expected in program_constraints:
            if len(actual) != len(expected) or any(
                resolve(left) != resolve(right) for left, right in zip(actual, expected)
            ):
                raise ValueError("Layout transfer program domains do not match.")

        if source is destination or memory_spans_overlap(
            tensor_memory_span(source),
            tensor_memory_span(destination),
        ):
            raise ValueError(
                "Layout transfer requires non-overlapping source and destination "
                "storage."
            )

    return validate


def _tensor_has_non_overlapping_strides(value):
    try:
        shape = tuple(int(size) for size in value.shape)
        strides = tuple(abs(int(stride)) for stride in value.stride())
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False

    if any(size == 0 for size in shape):
        return True

    dimensions = tuple(
        (stride, size) for size, stride in zip(shape, strides) if size > 1
    )

    if any(stride == 0 for stride, _size in dimensions):
        return False

    if len(dimensions) == 2:
        (first_stride, first_size), (second_stride, second_size) = dimensions
        divisor = math.gcd(first_stride, second_stride)

        return not (
            second_stride // divisor < first_size
            and first_stride // divisor < second_size
        )

    dimensions = sorted(dimensions)
    occupied_span = 1

    for stride, size in dimensions:
        if stride < occupied_span:
            return False

        occupied_span += (size - 1) * stride
    return True
