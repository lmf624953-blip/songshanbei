"""Backend-neutral analysis of rank-2 layout transfers."""

from dataclasses import dataclass

import sympy

from ninetoothed.ir import AccessMap, IndexExpr, TensorLayout, ssa


@dataclass(frozen=True, kw_only=True)
class TensorTransfer:
    """Structured physical access for one side of a layout transfer."""

    layout: TensorLayout
    access_map: AccessMap
    axis_mapping: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "axis_mapping", tuple(self.axis_mapping))


@dataclass(frozen=True, kw_only=True)
class LayoutTransfer:
    """A proven direct rank-2 permutation between physical tensor bindings."""

    source_binding: str
    destination_binding: str
    source: TensorTransfer
    destination: TensorTransfer
    value_constraints: tuple[tuple[tuple[IndexExpr, ...], tuple[IndexExpr, ...]], ...]
    physical_constraints: tuple[tuple[IndexExpr, IndexExpr], ...]
    program_constraints: tuple[tuple[tuple[IndexExpr, ...], tuple[IndexExpr, ...]], ...]
    requires_tiling: bool
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "value_constraints",
            "physical_constraints",
            "program_constraints",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    @property
    def permutation(self) -> tuple[int, ...]:
        """Map destination physical axes to source physical axes."""
        return self.source.axis_mapping

    @property
    def schedulable(self) -> bool:
        return self.failure_reason is None


def serialize_layout_transfer(transfer: LayoutTransfer) -> dict[str, object]:
    """Serialize a proven transfer without adding backend scheduling details."""
    return {
        "source_binding": transfer.source_binding,
        "destination_binding": transfer.destination_binding,
        "permutation": transfer.permutation,
        "requires_tiling": transfer.requires_tiling,
        "value_constraints": tuple(
            (
                tuple(value.render() for value in actual),
                tuple(value.render() for value in expected),
            )
            for actual, expected in transfer.value_constraints
        ),
        "physical_constraints": tuple(
            (left.render(), right.render())
            for left, right in transfer.physical_constraints
        ),
        "program_constraints": tuple(
            (
                tuple(value.render() for value in actual),
                tuple(value.render() for value in expected),
            )
            for actual, expected in transfer.program_constraints
        ),
    }


def analyze_layout_transfer(program: ssa.Program, tensors=()) -> LayoutTransfer | None:
    """Return the unique proven direct rank-2 permutation in the program."""
    store = _sole_top_level_store(program)

    if store is None:
        return None

    if any(name in store.attrs for name in ("indices", "source", "subscript")):
        return None

    value_types = _value_types(program)
    definitions = _definitions(program)
    users = _users(program)
    tensor_specs = {tensor.name: tensor for tensor in tensors}
    inputs = {value.name for value in program.inputs}
    stored_value, destination_binding = store.operands[:2]
    traced = _trace_source(stored_value, definitions, users, store)

    if traced is None:
        return None

    source_binding, explicit_transpose = traced

    if source_binding not in inputs or destination_binding not in inputs:
        return None

    source_type = value_types.get(source_binding)
    stored_type = value_types.get(stored_value)
    destination_type = value_types.get(destination_binding)

    if (
        not _rank_two_tensor(source_type)
        or not _rank_two_tensor(stored_type)
        or not _rank_two_tensor(destination_type)
    ):
        return None

    source_spec = tensor_specs.get(source_binding)
    destination_spec = tensor_specs.get(destination_binding)

    if explicit_transpose:
        return _explicit_transfer(
            source_binding,
            destination_binding,
            source_type,
            stored_type,
            destination_type,
            source_spec,
            destination_spec,
        )

    return _arrangement_transfer(
        source_binding,
        destination_binding,
        source_type,
        destination_type,
        source_spec,
        destination_spec,
    )


def _explicit_transfer(
    source_binding,
    destination_binding,
    source_type,
    stored_type,
    destination_type,
    source_spec,
    destination_spec,
):
    if any(
        spec is not None and spec.layout is not None
        for spec in (source_spec, destination_spec)
    ):
        return None

    requires_tiling = True

    source_shape = _physical_shape(source_spec, source_type)
    destination_shape = _physical_shape(destination_spec, destination_type)
    logical_shape = _type_shape(stored_type)

    if len(source_shape) != 2 or len(destination_shape) != 2:
        return None

    source_strides = _physical_strides(source_spec, source_shape)
    destination_strides = _physical_strides(destination_spec, destination_shape)
    source_program_shape = _program_shape(source_spec, logical_shape)
    destination_program_shape = _program_shape(destination_spec, logical_shape)
    destination_value_shape = _application_shape(destination_spec, destination_type)
    logical_indices = _logical_indices()
    source_access = _normalized_source_access(
        source_spec,
        logical_indices,
        source_strides,
    )
    destination_access = _destination_access(
        destination_spec,
        logical_indices,
        destination_strides,
    )
    source_layout = _normalized_layout(
        source_shape,
        source_strides,
        logical_shape,
        source_program_shape,
        source_access,
    )
    destination_layout = _normalized_layout(
        destination_shape,
        destination_strides,
        destination_value_shape,
        destination_program_shape,
        destination_access,
    )

    if not _renderable_access(source_layout, source_access) or not _renderable_access(
        destination_layout,
        destination_access,
    ):
        return None

    if _relative_axis_mapping(source_access, destination_access) != (1, 0):
        return None

    return _contract(
        source_binding,
        destination_binding,
        TensorTransfer(
            layout=source_layout,
            access_map=source_access,
            axis_mapping=(1, 0),
        ),
        TensorTransfer(
            layout=destination_layout,
            access_map=destination_access,
            axis_mapping=(0, 1),
        ),
        requires_tiling=requires_tiling,
    )


def _arrangement_transfer(
    source_binding,
    destination_binding,
    source_type,
    destination_type,
    source_spec,
    destination_spec,
):
    if source_spec is None or destination_spec is None:
        return None

    source_layout = source_spec.layout
    destination_layout = destination_spec.layout

    if (
        not _rank_two_layout(source_layout)
        or not _rank_two_layout(destination_layout)
        or not source_layout.value_accesses
        or not destination_layout.value_accesses
    ):
        return None

    if _type_shape(source_type) != _type_shape(destination_type):
        return None

    source_access = _layout_access(source_layout)
    destination_access = _layout_access(destination_layout)

    if not _renderable_access(
        source_layout,
        source_access,
    ) or not _renderable_access(destination_layout, destination_access):
        return None

    if not _outer_domain_is_mapped(
        source_layout,
        source_access,
    ) or not _outer_domain_is_mapped(destination_layout, destination_access):
        return None

    permutation = _relative_axis_mapping(
        source_access,
        destination_access,
    )

    if permutation != (1, 0):
        return None

    if not _access_coordinates_match(
        source_layout,
        source_access,
        destination_layout,
        destination_access,
        permutation,
    ):
        return None

    source = _tensor_transfer(source_layout, source_access, permutation)
    destination = _tensor_transfer(destination_layout, destination_access, (0, 1))

    return _contract(
        source_binding,
        destination_binding,
        source,
        destination,
        requires_tiling=False,
    )


def _contract(
    source_binding,
    destination_binding,
    source,
    destination,
    *,
    requires_tiling,
):
    source_layout = source.layout
    destination_layout = destination.layout
    value_constraints = (
        (source_layout.application_shape, destination_layout.application_shape),
    )
    physical_constraints = tuple(
        (
            source_layout.source_shape[source_axis],
            destination_layout.source_shape[destination_axis],
        )
        for destination_axis, source_axis in enumerate(source.axis_mapping)
    )
    program_constraints = ((source_layout.view_shape, destination_layout.view_shape),)
    physical_compatible, physical_reason = _constraint_compatibility(
        physical_constraints,
        "physical shapes",
    )
    program_compatible, program_reason = _program_compatibility(
        source_layout.view_shape,
        destination_layout.view_shape,
    )

    if (
        len(source_layout.application_shape) != 2
        or len(destination_layout.application_shape) != 2
    ):
        return None

    value_compatible, value_reason = _static_shape_compatibility(
        source_layout.application_shape,
        destination_layout.application_shape,
        "logical value shapes",
    )

    if not value_compatible:
        failure_reason = value_reason
    elif not program_compatible:
        failure_reason = program_reason
    elif not physical_compatible:
        failure_reason = physical_reason
    else:
        failure_reason = None

    return LayoutTransfer(
        source_binding=source_binding,
        destination_binding=destination_binding,
        source=source,
        destination=destination,
        value_constraints=value_constraints,
        physical_constraints=physical_constraints,
        program_constraints=program_constraints,
        requires_tiling=requires_tiling,
        failure_reason=failure_reason,
    )


def _tensor_transfer(layout, access, axis_mapping):
    return TensorTransfer(
        layout=layout,
        access_map=access,
        axis_mapping=axis_mapping,
    )


def _sole_top_level_store(program):
    if len(program.blocks) != 1:
        return None

    effects = tuple(
        operation
        for operation in _walk_program(program)
        if operation.opcode in {"mem.atomic_add", "mem.store"}
    )

    if (
        len(effects) != 1
        or effects[0].opcode != "mem.store"
        or len(effects[0].operands) != 2
        or not any(
            effects[0] is operation for operation in program.blocks[0].operations
        )
    ):
        return None

    return effects[0]


def _trace_source(value, definitions, users, store):
    operation = definitions.get(value)

    if operation is None:
        return value, False

    if (
        operation.opcode != "linalg.transpose"
        or len(operation.operands) != 1
        or len(operation.results) != 1
        or operation.regions
        or tuple(operation.attrs.get("permutation", (1, 0))) != (1, 0)
        or users.get(value, ()) != (store,)
        or operation.operands[0] in definitions
    ):
        return None

    return operation.operands[0], True


def _relative_axis_mapping(source_access, destination_access):
    source_indices = source_access.source_indices
    destination_indices = destination_access.source_indices

    if len(source_indices) != 2 or len(destination_indices) != 2:
        return None

    source_axes = tuple(_value_axis(index) for index in source_indices)
    destination_axes = tuple(_value_axis(index) for index in destination_indices)

    if set(source_axes) == {0, 1} and set(destination_axes) == {0, 1}:
        return tuple(source_axes.index(axis) for axis in destination_axes)

    return None


def _access_coordinates_match(
    source_layout,
    source_access,
    destination_layout,
    destination_access,
    permutation,
):
    source_substitutions = {}
    destination_substitutions = {}

    for destination_axis, source_axis in enumerate(permutation):
        canonical = sympy.Symbol(f"_physical_{destination_axis}")
        source_extent = sympy.sympify(source_layout.source_shape[source_axis].render())
        destination_extent = sympy.sympify(
            destination_layout.source_shape[destination_axis].render()
        )

        if (
            source_extent in source_substitutions
            and source_substitutions[source_extent] != canonical
        ) or (
            destination_extent in destination_substitutions
            and destination_substitutions[destination_extent] != canonical
        ):
            return False

        source_substitutions[source_extent] = canonical
        destination_substitutions[destination_extent] = canonical

    try:
        return all(
            sympy.simplify(
                sympy.sympify(source_access.source_indices[source_axis].render()).subs(
                    source_substitutions
                )
                - sympy.sympify(
                    destination_access.source_indices[destination_axis].render()
                ).subs(destination_substitutions)
            )
            == 0
            for destination_axis, source_axis in enumerate(permutation)
        )
    except (SyntaxError, TypeError, ValueError):
        return False


def _renderable_access(layout, access):
    if access is None:
        return False

    axes = tuple(_value_axis(index) for index in access.source_indices)

    if set(axes) != {0, 1}:
        return False

    expected_linear_index = _linear_index(
        access.source_indices,
        layout.source_strides,
    )

    if not _equivalent_index_expr(access.linear_index, expected_linear_index):
        return False

    layout_expressions = (
        *layout.source_shape,
        *layout.source_strides,
        *layout.view_shape,
        *layout.application_shape,
    )
    allowed = {
        "index",
        "outer_index",
        "value_0",
        "value_1",
        *(
            symbol
            for expression in layout_expressions
            for symbol in _symbols(expression)
        ),
    }
    access_expressions = (
        *access.source_indices,
        access.linear_index,
        access.predicate,
    )

    return all(_symbols(expression) <= allowed for expression in access_expressions)


def _outer_domain_is_mapped(layout, access):
    single_program = all(_static_integer(extent) == 1 for extent in layout.view_shape)

    return single_program or any(
        "outer_index" in _symbols(index) for index in access.source_indices
    )


def _symbols(expression):
    if expression.op == "symbol":
        return frozenset({str(expression.value)})

    return frozenset().union(*(_symbols(operand) for operand in expression.operands))


def _value_axis(expression):
    try:
        rendered = sympy.sympify(expression.render())
    except (SyntaxError, TypeError, ValueError):
        return None

    value_symbols = tuple(sympy.Symbol(f"value_{axis}") for axis in range(2))
    coefficients = tuple(sympy.diff(rendered, symbol) for symbol in value_symbols)

    if coefficients.count(sympy.Integer(1)) != 1 or any(
        coefficient not in {sympy.Integer(0), sympy.Integer(1)}
        for coefficient in coefficients
    ):
        return None

    axis = coefficients.index(sympy.Integer(1))
    remainder = sympy.simplify(rendered - value_symbols[axis])

    if remainder == 0:
        return axis

    if sympy.Symbol("outer_index") in remainder.free_symbols and not any(
        symbol in remainder.free_symbols for symbol in value_symbols
    ):
        return axis
    return None


def _linear_index(indices, strides):
    terms = tuple(
        IndexExpr(op="mul", operands=(index, stride))
        for index, stride in zip(indices, strides)
    )

    if not terms:
        return IndexExpr.parse(0)

    result = terms[0]

    for term in terms[1:]:
        result = IndexExpr(op="add", operands=(result, term))
    return result


def _equivalent_index_expr(left, right):
    try:
        difference = sympy.sympify(left.render()) - sympy.sympify(right.render())
    except (SyntaxError, TypeError, ValueError):
        return False
    return sympy.simplify(difference) == 0


def _program_compatibility(source_shape, destination_shape):
    return _static_shape_compatibility(
        source_shape,
        destination_shape,
        "program shapes",
    )


def _constraint_compatibility(constraints, label):
    for left, right in constraints:
        left_value = _static_integer(left)
        right_value = _static_integer(right)

        if (
            left_value is not None
            and right_value is not None
            and left_value != right_value
        ):
            return False, f"{label} are statically incompatible"

    return True, None


def _static_shape_compatibility(left, right, label):
    if len(left) != len(right):
        return False, f"{label} have different ranks"

    for left_dim, right_dim in zip(left, right):
        left_value = _static_integer(left_dim)
        right_value = _static_integer(right_dim)

        if (
            left_value is not None
            and right_value is not None
            and left_value != right_value
        ):
            return False, f"{label} are statically incompatible"

    return True, None


def _static_integer(expression):
    if expression.op == "constant":
        value = expression.value

        if isinstance(value, int) and not isinstance(value, bool):
            return value

    return None


def _normalized_layout(
    source_shape,
    source_strides,
    value_shape,
    program_shape,
    access,
):
    return TensorLayout(
        source_shape=source_shape,
        source_strides=source_strides,
        view_shape=program_shape,
        application_shape=value_shape,
        view_access=access,
        value_accesses=(access,),
    )


def _access_map(indices, strides, predicate):
    linear = IndexExpr(op="constant", value=0)

    for index, stride in zip(indices, strides):
        linear = IndexExpr(
            op="add",
            operands=(
                linear,
                IndexExpr(op="mul", operands=(index, stride)),
            ),
        )

    return AccessMap(
        source_indices=indices,
        linear_index=linear,
        predicate=predicate,
    )


def _logical_indices():
    return (IndexExpr.parse("value_0"), IndexExpr.parse("value_1"))


def _normalized_source_access(spec, logical_indices, source_strides):
    if spec is not None and spec.layout is not None:
        access = _layout_access(spec.layout)

        if access is not None:
            return _permute_access(access, (1, 0))

    return _access_map(
        (logical_indices[1], logical_indices[0]),
        source_strides,
        IndexExpr.parse(True),
    )


def _destination_access(spec, logical_indices, destination_strides):
    if spec is not None and spec.layout is not None:
        access = _layout_access(spec.layout)

        if access is not None:
            return access

    return _access_map(
        logical_indices,
        destination_strides,
        IndexExpr.parse(True),
    )


def _permute_access(access, permutation):
    return AccessMap(
        source_indices=tuple(
            _substitute_value_axes(expression, permutation)
            for expression in access.source_indices
        ),
        linear_index=_substitute_value_axes(access.linear_index, permutation),
        predicate=_substitute_value_axes(access.predicate, permutation),
    )


def _substitute_value_axes(expression, permutation):
    if expression.op == "symbol":
        name = str(expression.value)
        suffix = name.removeprefix("value_")

        if name.startswith("value_") and suffix.isdigit():
            axis = int(suffix)

            if axis < len(permutation):
                return IndexExpr(op="symbol", value=f"value_{permutation[axis]}")

    return IndexExpr(
        op=expression.op,
        operands=tuple(
            _substitute_value_axes(operand, permutation)
            for operand in expression.operands
        ),
        value=expression.value,
    )


def _layout_access(layout):
    if layout is None:
        return None

    if layout.value_accesses:
        return layout.value_accesses[-1]

    return layout.view_access


def _physical_shape(spec, type_):
    if spec is not None and spec.layout is not None:
        return spec.layout.source_shape

    return _type_shape(type_)


def _physical_strides(spec, source_shape):
    if spec is not None and spec.layout is not None:
        return spec.layout.source_strides

    return (source_shape[1], IndexExpr.parse(1))


def _program_shape(spec, fallback):
    if spec is not None and spec.layout is not None:
        return spec.layout.view_shape

    return fallback


def _application_shape(spec, type_):
    if spec is not None and spec.layout is not None:
        return spec.layout.application_shape

    return _type_shape(type_)


def _type_shape(type_):
    return tuple(IndexExpr.parse(dim) for dim in type_.shape)


def _rank_two_layout(layout: TensorLayout | None) -> bool:
    access = _layout_access(layout)

    return bool(
        layout is not None
        and len(layout.source_shape) == 2
        and len(layout.source_strides) == 2
        and len(layout.application_shape) == 2
        and access is not None
        and len(access.source_indices) == 2
    )


def _rank_two_tensor(type_):
    return bool(type_ is not None and type_.kind == "tensor" and len(type_.shape) == 2)


def _value_types(program):
    result = {value.name: value.type for value in (*program.inputs, *program.outputs)}

    def visit_block(block):
        result.update({argument.name: argument.type for argument in block.args})

        for operation in block.operations:
            result.update({value.name: value.type for value in operation.results})

            for region in operation.regions:
                visit_block(region)

    for block in program.blocks:
        visit_block(block)
    return result


def _definitions(program):
    return {
        result.name: operation
        for operation in _walk_program(program)
        for result in operation.results
    }


def _users(program):
    result = {}

    for operation in _walk_program(program):
        for operand in operation.operands:
            result.setdefault(operand, []).append(operation)

    return {value: tuple(operations) for value, operations in result.items()}


def _walk_program(program):
    def visit_block(block):
        for operation in block.operations:
            yield operation

            for region in operation.regions:
                yield from visit_block(region)

    for block in program.blocks:
        yield from visit_block(block)


__all__ = ["LayoutTransfer", "TensorTransfer", "analyze_layout_transfer"]
