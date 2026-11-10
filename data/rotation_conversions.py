"""Minimal PyTorch3D-compatible rotation conversions used by RoboTest.

Adapted from PyTorch3D's BSD-licensed ``rotation_conversions.py``. Keeping
these pure-PyTorch operations local avoids installing PyTorch3D's compiled
CUDA extensions solely for four conversion functions.
"""

import functools

import torch


def quaternion_to_matrix(quaternions):
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    values = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return values.reshape(quaternions.shape[:-1] + (3, 3))


def _copysign(a, b):
    return torch.where((a < 0) != (b < 0), -a, a)


def _sqrt_positive_part(value):
    result = torch.zeros_like(value)
    positive = value > 0
    result[positive] = torch.sqrt(value[positive])
    return result


def matrix_to_quaternion(matrix):
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    m00 = matrix[..., 0, 0]
    m11 = matrix[..., 1, 1]
    m22 = matrix[..., 2, 2]
    real = 0.5 * _sqrt_positive_part(1 + m00 + m11 + m22)
    x_abs = 0.5 * _sqrt_positive_part(1 + m00 - m11 - m22)
    y_abs = 0.5 * _sqrt_positive_part(1 - m00 + m11 - m22)
    z_abs = 0.5 * _sqrt_positive_part(1 - m00 - m11 + m22)
    x = _copysign(x_abs, matrix[..., 2, 1] - matrix[..., 1, 2])
    y = _copysign(y_abs, matrix[..., 0, 2] - matrix[..., 2, 0])
    z = _copysign(z_abs, matrix[..., 1, 0] - matrix[..., 0, 1])
    return torch.stack((real, x, y, z), -1)


def _axis_angle_rotation(axis, angle):
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)
    if axis == "X":
        flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError(f"Unknown rotation axis {axis}.")
    return torch.stack(flat, -1).reshape(angle.shape + (3, 3))


def _validate_convention(convention):
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    if any(letter not in ("X", "Y", "Z") for letter in convention):
        raise ValueError(f"Invalid convention {convention}.")


def euler_angles_to_matrix(euler_angles, convention):
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input Euler angles.")
    _validate_convention(convention)
    matrices = map(
        _axis_angle_rotation, convention, torch.unbind(euler_angles, -1)
    )
    return functools.reduce(torch.matmul, matrices)


def _angle_from_tan(axis, other_axis, data, horizontal, tait_bryan):
    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


def _index_from_letter(letter):
    return {"X": 0, "Y": 1, "Z": 2}[letter]


def matrix_to_euler_angles(matrix, convention):
    _validate_convention(convention)
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    first_axis = _index_from_letter(convention[0])
    third_axis = _index_from_letter(convention[2])
    tait_bryan = first_axis != third_axis
    if tait_bryan:
        sign = -1.0 if first_axis - third_axis in [-1, 2] else 1.0
        central_angle = torch.asin(matrix[..., first_axis, third_axis] * sign)
    else:
        central_angle = torch.acos(matrix[..., first_axis, first_axis])
    return torch.stack(
        (
            _angle_from_tan(
                convention[0], convention[1], matrix[..., third_axis], False, tait_bryan
            ),
            central_angle,
            _angle_from_tan(
                convention[2], convention[1], matrix[..., first_axis, :], True, tait_bryan
            ),
        ),
        -1,
    )
