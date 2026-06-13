"""
Grid point generation utility.
Generates discrete 3D grid points for the robot's end-effector workspace.
"""

import itertools
import numpy as np # pyright: ignore[reportMissingImports]

def generar_malla_3x3x3(spacing):
    """
    Genera una malla 3x3x3 de puntos alrededor de un centro.

    Parámetros
    ----------
    center : list o tuple
        Centro de la malla [x, y, z].
    spacing : float
        Distancia entre puntos consecutivos.

    Retorna
    -------
    points : list
        Lista con 27 puntos [x, y, z].
    """

    head_pos = [-0.00327, 0.00055, 0.35546]

    distance_in_front = 0.15

    center = [
        head_pos[0] - distance_in_front,
        head_pos[1] - 0.06,
        head_pos[2]
    ]

    offsets = [-spacing, 0.0, spacing]

    points = []

    for dx, dy, dz in itertools.product(offsets, offsets, offsets):
        point = [
            center[0] + dx,
            center[1] + dy,
            center[2] + dz
        ]
        points.append(point)

    return points


if __name__ == "__main__":
    points = generar_malla_3x3x3(spacing=0.04)
    print("GRID_POINTS = [")
    for pnt in points:
        print(f"[{pnt[0]}, {pnt[1]}, {pnt[2]}],")
    print("]")