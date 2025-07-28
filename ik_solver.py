import pybullet as p
import pybullet_data
import numpy as np
import time
from scipy.optimize import minimize

class IKSolver:
    def __init__(self, urdf_path, use_gui=False):
        if use_gui:
            if p.getConnectionInfo()["isConnected"]:
                p.disconnect()
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)

        # Ground plane
        p.loadURDF("plane.urdf")

        # Combined robot URDF (gantry + arm)
        self.robot_id = p.loadURDF(urdf_path, useFixedBase=True)

        # Movable joint indices (gantry + 6 arm joints)
        self.joint_indices = [
            j for j in range(p.getNumJoints(self.robot_id))
            if p.getJointInfo(self.robot_id, j)[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]
        ]

        # Locate flange (end-effector)
        self.flange_index = -1
        for j in range(p.getNumJoints(self.robot_id)):
            if p.getJointInfo(self.robot_id, j)[12].decode() == "flange":
                self.flange_index = j
                break
        if self.flange_index == -1:
            raise RuntimeError("Flange link not found in URDF")

    def solve_ik(self, target_pos, target_orn):
        """Compute IK using PyBullet."""
        ik_solution = list(p.calculateInverseKinematics(
            self.robot_id,
            self.flange_index,
            target_pos,
            target_orn
        ))

        # Check initial error
        self._apply_joints(ik_solution)
        pos_err, orn_err, _, _ = self.validate_solution(ik_solution, target_pos, target_orn)

        # Only refine if error is significant
        if pos_err > 0.01 or orn_err > 0.01:
            def cost(q):
                self._apply_joints(q)
                state = p.getLinkState(self.robot_id, self.flange_index)
                pos = np.array(state[4])
                orn = np.array(state[5])
                pos_cost = np.linalg.norm(pos - target_pos)
                dot = np.clip(np.dot(orn, target_orn), -1.0, 1.0)
                orn_cost = 2 * np.arccos(abs(dot))
                return pos_cost + 0.1 * orn_cost

            res = minimize(cost, ik_solution, method='L-BFGS-B')
            return list(res.x)

        return ik_solution

    def _apply_joints(self, q_values):
        """Helper to set joint states directly."""
        for idx, val in zip(self.joint_indices, q_values):
            p.resetJointState(self.robot_id, idx, val)
        p.stepSimulation()

    def apply_joint_positions(self, q_solution, steps=120):
        """Smoothly animate robot from current pose to q_solution."""
        current = [p.getJointState(self.robot_id, idx)[0] for idx in self.joint_indices]
        for t in range(steps):
            alpha = t / (steps - 1)
            interpolated = [(1 - alpha) * c + alpha * q for c, q in zip(current, q_solution)]
            for idx, val in zip(self.joint_indices, interpolated):
                p.resetJointState(self.robot_id, idx, val)
            p.stepSimulation()
            time.sleep(1. / 720)

    def validate_solution(self, q_solution, target_pos, target_orn):
        """Apply solution, step sim, and measure pose error."""
        self._apply_joints(q_solution)

        state = p.getLinkState(self.robot_id, self.flange_index)
        actual_pos = np.array(state[4])
        actual_orn = np.array(state[5])

        pos_error = np.linalg.norm(actual_pos - target_pos)
        dot = np.clip(np.dot(actual_orn, target_orn), -1.0, 1.0)
        orn_error = 2 * np.arccos(abs(dot))

        print("\n--- Validation ---")
        print(f"Target Position: {np.round(target_pos, 4)}")
        print(f"Actual Position: {np.round(actual_pos, 4)}")
        print(f"Position Error: {pos_error:.6f}")
        print(f"Orientation Error: {orn_error:.6f} rad")
        print("------------------\n")

        return pos_error, orn_error, actual_pos, actual_orn
