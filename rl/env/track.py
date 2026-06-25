"""The test_track race in ENU (our frame), ROS-free, shared by the deploy
backends. 12 visible gates (invisible loop-back excluded), matching
flight_plans/test_track.json after the Rx(180) transform.
"""

import math

GATES_ENU = [
    {'x': 12.5, 'y':  2.0, 'z': 1.45, 'yaw':  math.pi},      # gate01
    {'x':  6.5, 'y':  6.0, 'z': 1.45, 'yaw':  2.35619},      # gate02
    {'x':  5.5, 'y': 14.0, 'z': 1.45, 'yaw':  2.0944},       # gate03
    {'x':  2.5, 'y': 23.8, 'z': 1.45, 'yaw':  1.5708},       # gate04
    {'x':  7.5, 'y': 30.0, 'z': 1.45, 'yaw': -0.174533},     # gate05
    {'x': 12.2, 'y': 22.0, 'z': 1.45, 'yaw':  0.0},          # gate06
    {'x': 17.5, 'y': 30.0, 'z': 4.00, 'yaw':  1.39626},      # gate07 split-up
    {'x': 17.5, 'y': 30.0, 'z': 1.30, 'yaw': -1.74533},      # gate07 split-down
    {'x': 18.5, 'y': 22.0, 'z': 1.45, 'yaw': -1.39626},      # gate08
    {'x': 20.5, 'y': 14.0, 'z': 1.45, 'yaw': -1.74533},      # gate09
    {'x': 18.5, 'y':  6.0, 'z': 4.00, 'yaw': -2.35619},      # gate10 ladder-up
    {'x': 18.5, 'y':  6.0, 'z': 1.30, 'yaw': -2.35619},      # gate10 ladder-down
]
