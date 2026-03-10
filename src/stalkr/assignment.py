from __future__ import annotations


def solve_assignment(
    cost_matrix: list[list[float]],
    *,
    backend: str = "auto",
) -> list[tuple[int, int]]:
    if backend not in {"auto", "hungarian", "lapjv"}:
        raise ValueError(f"unsupported assignment backend: {backend}")

    if backend in {"auto", "lapjv"}:
        lapjv_matches = _solve_with_lapjv(cost_matrix)
        if lapjv_matches is not None:
            return lapjv_matches
        if backend == "lapjv":
            raise ModuleNotFoundError("lapjv backend requested but lapjv is not installed.")

    return hungarian(cost_matrix)


def hungarian(cost_matrix: list[list[float]]) -> list[tuple[int, int]]:
    """
    Solve the rectangular linear assignment problem using the Hungarian method.

    Returns pairs of (row_index, column_index).
    """

    if not cost_matrix or not cost_matrix[0]:
        return []

    row_count = len(cost_matrix)
    column_count = len(cost_matrix[0])
    transposed = False
    matrix = cost_matrix

    if row_count > column_count:
        transposed = True
        matrix = [
            [cost_matrix[row_index][column_index] for row_index in range(row_count)]
            for column_index in range(column_count)
        ]
        row_count, column_count = column_count, row_count

    u = [0.0] * (row_count + 1)
    v = [0.0] * (column_count + 1)
    p = [0] * (column_count + 1)
    way = [0] * (column_count + 1)

    for row_index in range(1, row_count + 1):
        p[0] = row_index
        column_index = 0
        min_values = [float("inf")] * (column_count + 1)
        used = [False] * (column_count + 1)

        while True:
            used[column_index] = True
            current_row = p[column_index]
            delta = float("inf")
            next_column = 0

            for candidate_column in range(1, column_count + 1):
                if used[candidate_column]:
                    continue
                current_cost = (
                    matrix[current_row - 1][candidate_column - 1]
                    - u[current_row]
                    - v[candidate_column]
                )
                if current_cost < min_values[candidate_column]:
                    min_values[candidate_column] = current_cost
                    way[candidate_column] = column_index
                if min_values[candidate_column] < delta:
                    delta = min_values[candidate_column]
                    next_column = candidate_column

            for candidate_column in range(column_count + 1):
                if used[candidate_column]:
                    u[p[candidate_column]] += delta
                    v[candidate_column] -= delta
                else:
                    min_values[candidate_column] -= delta

            column_index = next_column
            if p[column_index] == 0:
                break

        while True:
            next_column = way[column_index]
            p[column_index] = p[next_column]
            column_index = next_column
            if column_index == 0:
                break

    assignments: list[tuple[int, int]] = []
    for column_index in range(1, column_count + 1):
        if p[column_index] == 0:
            continue
        row_index = p[column_index] - 1
        matched_column = column_index - 1
        if transposed:
            assignments.append((matched_column, row_index))
        else:
            assignments.append((row_index, matched_column))

    assignments.sort()
    return assignments


def _solve_with_lapjv(cost_matrix: list[list[float]]) -> list[tuple[int, int]] | None:
    try:
        import numpy as np
        from lapjv import lapjv
    except ModuleNotFoundError:
        return None

    if not cost_matrix or not cost_matrix[0]:
        return []

    cost_array = np.asarray(cost_matrix, dtype=np.float64)
    row_count, column_count = cost_array.shape

    if row_count != column_count:
        size = max(row_count, column_count)
        pad_value = float(cost_array.max()) + 1.0 if cost_array.size else 1.0
        padded = np.full((size, size), pad_value, dtype=np.float64)
        padded[:row_count, :column_count] = cost_array
        cost_array = padded

    _, row_assignment, _ = lapjv(cost_array)
    matches: list[tuple[int, int]] = []
    for row_index, column_index in enumerate(row_assignment.tolist()):
        if row_index >= row_count or column_index < 0 or column_index >= column_count:
            continue
        matches.append((row_index, column_index))
    matches.sort()
    return matches
