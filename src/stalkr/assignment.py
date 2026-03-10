from __future__ import annotations


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

