"""Optional RViz visualization for the ROS-free bindings sim.

The simulator itself (multirotor_pysim) has no ROS. This bridge is a SEPARATE,
opt-in publisher: given the env's drone pose (ENU) and the track, it publishes
Markers to RViz so you can watch a policy fly. Everything lives in the 'earth'
frame (set that as RViz's Fixed Frame) — no TF tree / sim clock needed.

Used by run_quadrace_policy --backend sim --viz.
"""

from __future__ import annotations

import math

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, TransformStamped
from tf2_ros import TransformBroadcaster


def _yaw_quat(yaw):
    return math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)


class SimRvizBridge:
    """Visualize the ROS-free sim. Publishes gate meshes + flight trail on
    /rl/track_gates, and broadcasts the earth->base_link TF so the ground
    station's RobotModel renders the real drone model (as before). Optionally
    also draws a sphere/arrow body marker (body_marker=True)."""

    def __init__(self, node_name='sim_rviz_bridge', frame='earth',
                 base_frame='drone0/base_link', body_marker=False, trail_len=400):
        if not rclpy.ok():
            rclpy.init()
        self.node: Node = rclpy.create_node(node_name)
        self.frame = frame
        self.base_frame = base_frame
        self.body_marker = body_marker
        # Markers go on the SAME topic the ground station already shows.
        self._track_pub = self.node.create_publisher(MarkerArray, '/rl/track_gates', 10)
        self._tf = TransformBroadcaster(self.node)
        self._trail = []
        self._trail_len = trail_len

    def _stamp(self):
        return self.node.get_clock().now().to_msg()

    def publish_gates(self, gates, target_idx=None):
        arr = MarkerArray()
        stamp = self._stamp()
        for i, g in enumerate(gates):
            m = Marker()
            m.header.frame_id = self.frame
            m.header.stamp = stamp
            m.ns = 'quadrace_gates'
            m.id = i
            m.type = Marker.MESH_RESOURCE
            m.action = Marker.ADD
            m.mesh_resource = (
                'package://as2_gazebo_assets/models/cvar_gate/meshes/model.dae')
            m.mesh_use_embedded_materials = True
            m.pose.position.x = float(g['x'])
            m.pose.position.y = float(g['y'])
            m.pose.position.z = float(g['z']) - 2.7 / 2.0
            qw, qx, qy, qz = _yaw_quat(float(g['yaw']) + math.pi)
            m.pose.orientation.w, m.pose.orientation.x = qw, qx
            m.pose.orientation.y, m.pose.orientation.z = qy, qz
            m.scale.x = m.scale.y = m.scale.z = 1.0
            # Highlight the gate currently being chased (green tint) via color
            # override; non-target gates keep the embedded mesh material.
            if target_idx is not None and i == target_idx:
                m.mesh_use_embedded_materials = False
                m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 1.0, 0.1, 1.0
            arr.markers.append(m)
        self._track_pub.publish(arr)

    def publish_drone(self, pos, yaw=0.0, quat=None):
        """Broadcast earth->base_link TF (renders the RobotModel) and publish the
        trail. quat=(qw,qx,qy,qz) gives full attitude; else yaw-only."""
        stamp = self._stamp()
        if quat is not None:
            qw, qx, qy, qz = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
        else:
            qw, qx, qy, qz = _yaw_quat(float(yaw))

        # --- TF: earth -> base_link (drives the ground station RobotModel) ---
        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = self.frame
        tf.child_frame_id = self.base_frame
        tf.transform.translation.x = float(pos[0])
        tf.transform.translation.y = float(pos[1])
        tf.transform.translation.z = float(pos[2])
        tf.transform.rotation.w, tf.transform.rotation.x = qw, qx
        tf.transform.rotation.y, tf.transform.rotation.z = qy, qz
        self._tf.sendTransform(tf)

        # --- trail (+ optional body marker) on the shared markers topic ---
        arr = MarkerArray()
        self._trail.append((float(pos[0]), float(pos[1]), float(pos[2])))
        if len(self._trail) > self._trail_len:
            self._trail = self._trail[-self._trail_len:]
        tr = Marker()
        tr.header.frame_id = self.frame
        tr.header.stamp = stamp
        tr.ns, tr.id = 'quadrace_trail', 0
        tr.type, tr.action = Marker.LINE_STRIP, Marker.ADD
        tr.scale.x = 0.05
        tr.color.r, tr.color.g, tr.color.b, tr.color.a = 1.0, 0.8, 0.1, 0.9
        tr.pose.orientation.w = 1.0
        tr.points = [Point(x=p[0], y=p[1], z=p[2]) for p in self._trail]
        arr.markers.append(tr)

        if self.body_marker:
            sph = Marker()
            sph.header.frame_id = self.frame
            sph.header.stamp = stamp
            sph.ns, sph.id = 'quadrace_drone', 0
            sph.type, sph.action = Marker.SPHERE, Marker.ADD
            sph.pose.position.x = float(pos[0])
            sph.pose.position.y = float(pos[1])
            sph.pose.position.z = float(pos[2])
            sph.pose.orientation.w = 1.0
            sph.scale.x = sph.scale.y = sph.scale.z = 0.5
            sph.color.r, sph.color.g, sph.color.b, sph.color.a = 0.1, 0.5, 1.0, 1.0
            arr.markers.append(sph)

        self._track_pub.publish(arr)

    def reset_trail(self):
        self._trail = []

    def close(self):
        try:
            self.node.destroy_node()
        except Exception:
            pass
