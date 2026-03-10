from __future__ import annotations

from dataclasses import dataclass, field

from .types import Box
from .utils import clamp_box

Matrix = list[list[float]]
Vector = list[float]


def _identity(size: int) -> Matrix:
    return [[1.0 if row == column else 0.0 for column in range(size)] for row in range(size)]


def _transpose(matrix: Matrix) -> Matrix:
    return [list(column) for column in zip(*matrix, strict=True)]


def _matmul(left: Matrix, right: Matrix) -> Matrix:
    right_t = _transpose(right)
    return [
        [sum(a * b for a, b in zip(row, column, strict=True)) for column in right_t]
        for row in left
    ]


def _matvec(matrix: Matrix, vector: Vector) -> Vector:
    return [sum(a * b for a, b in zip(row, vector, strict=True)) for row in matrix]


def _matadd(left: Matrix, right: Matrix) -> Matrix:
    return [
        [left[row][column] + right[row][column] for column in range(len(left[row]))]
        for row in range(len(left))
    ]


def _matsub(left: Matrix, right: Matrix) -> Matrix:
    return [
        [left[row][column] - right[row][column] for column in range(len(left[row]))]
        for row in range(len(left))
    ]


def _vector_add(left: Vector, right: Vector) -> Vector:
    return [a + b for a, b in zip(left, right, strict=True)]


def _inverse(matrix: Matrix) -> Matrix:
    size = len(matrix)
    augmented = [
        row[:] + identity_row[:]
        for row, identity_row in zip(matrix, _identity(size), strict=True)
    ]

    for pivot_index in range(size):
        pivot_row = max(
            range(pivot_index, size),
            key=lambda row_index: abs(augmented[row_index][pivot_index]),
        )
        if abs(augmented[pivot_row][pivot_index]) < 1e-12:
            raise ValueError("matrix is singular")
        augmented[pivot_index], augmented[pivot_row] = augmented[pivot_row], augmented[pivot_index]

        pivot = augmented[pivot_index][pivot_index]
        augmented[pivot_index] = [value / pivot for value in augmented[pivot_index]]

        for row_index in range(size):
            if row_index == pivot_index:
                continue
            factor = augmented[row_index][pivot_index]
            if factor == 0.0:
                continue
            augmented[row_index] = [
                current - factor * pivot_value
                for current, pivot_value in zip(
                    augmented[row_index],
                    augmented[pivot_index],
                    strict=True,
                )
            ]

    return [row[size:] for row in augmented]


def _diag(values: Vector) -> Matrix:
    matrix = [[0.0] * len(values) for _ in values]
    for index, value in enumerate(values):
        matrix[index][index] = value
    return matrix


def _box_scale(box: Box) -> float:
    x1, y1, x2, y2 = clamp_box(box)
    return max(abs(x2 - x1), abs(y2 - y1), 1.0)


@dataclass(slots=True)
class KalmanFilter:
    dt: float = 1.0
    process_position_weight: float = 1.0 / 20.0
    process_velocity_weight: float = 1.0 / 160.0
    measurement_weight: float = 1.0 / 20.0
    _motion_matrix: Matrix = field(init=False)
    _update_matrix: Matrix = field(init=False)

    def __post_init__(self) -> None:
        self._motion_matrix = _identity(8)
        for index in range(4):
            self._motion_matrix[index][index + 4] = self.dt
        self._update_matrix = [
            [1.0 if column == row else 0.0 for column in range(8)]
            for row in range(4)
        ]

    def initiate(self, box: Box) -> tuple[Vector, Matrix]:
        measurement = list(clamp_box(box))
        mean = measurement + [0.0, 0.0, 0.0, 0.0]
        scale = _box_scale(box)
        covariance = _diag(
            [
                (2.0 * self.measurement_weight * scale) ** 2,
                (2.0 * self.measurement_weight * scale) ** 2,
                (2.0 * self.measurement_weight * scale) ** 2,
                (2.0 * self.measurement_weight * scale) ** 2,
                (10.0 * self.process_velocity_weight * scale) ** 2,
                (10.0 * self.process_velocity_weight * scale) ** 2,
                (10.0 * self.process_velocity_weight * scale) ** 2,
                (10.0 * self.process_velocity_weight * scale) ** 2,
            ]
        )
        return mean, covariance

    def predict(self, mean: Vector, covariance: Matrix) -> tuple[Vector, Matrix]:
        box = (mean[0], mean[1], mean[2], mean[3])
        scale = _box_scale(box)
        motion_covariance = _diag(
            [
                (self.process_position_weight * scale) ** 2,
                (self.process_position_weight * scale) ** 2,
                (self.process_position_weight * scale) ** 2,
                (self.process_position_weight * scale) ** 2,
                (self.process_velocity_weight * scale) ** 2,
                (self.process_velocity_weight * scale) ** 2,
                (self.process_velocity_weight * scale) ** 2,
                (self.process_velocity_weight * scale) ** 2,
            ]
        )
        predicted_mean = _matvec(self._motion_matrix, mean)
        predicted_covariance = _matadd(
            _matmul(_matmul(self._motion_matrix, covariance), _transpose(self._motion_matrix)),
            motion_covariance,
        )
        return predicted_mean, predicted_covariance

    def update(self, mean: Vector, covariance: Matrix, box: Box) -> tuple[Vector, Matrix]:
        measurement = list(clamp_box(box))
        projected_mean = _matvec(self._update_matrix, mean)
        box_scale = _box_scale(box)
        measurement_covariance = _diag(
            [(self.measurement_weight * box_scale) ** 2] * 4
        )
        projected_covariance = _matadd(
            _matmul(_matmul(self._update_matrix, covariance), _transpose(self._update_matrix)),
            measurement_covariance,
        )

        kalman_gain = _matmul(
            _matmul(covariance, _transpose(self._update_matrix)),
            _inverse(projected_covariance),
        )
        innovation = [measurement_value - projected_value for measurement_value, projected_value in zip(measurement, projected_mean, strict=True)]
        updated_mean = _vector_add(mean, _matvec(kalman_gain, innovation))
        updated_covariance = _matmul(
            _matsub(_identity(8), _matmul(kalman_gain, self._update_matrix)),
            covariance,
        )
        return updated_mean, updated_covariance

    def box_from_mean(self, mean: Vector) -> Box:
        return clamp_box((mean[0], mean[1], mean[2], mean[3]))

    def velocity_from_mean(self, mean: Vector) -> tuple[float, float, float, float]:
        return (mean[4], mean[5], mean[6], mean[7])
