import pybullet as p
import numpy as np
import time
import pandas as pd
from ik_solver import IKSolver

# Initialize solver with GUI enabled
solver = IKSolver("robot-urdfs/combined/combined_gantry_robot.urdf", use_gui=True)

# Generate 5 random reachable targets (x:1-3, y:-1 to 1, z:1-2)
np.random.seed(42)
targets = np.column_stack([
    np.random.uniform(1.0, 3.0, 5),
    np.random.uniform(-1.0, 1.0, 5),
    np.random.uniform(1.0, 2.0, 5)
])

target_orn = np.array([0, 0, 0, 1])  

# Place static spheres at target locations
for pos in targets:
    p.createMultiBody(
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 0, 0, 1]),
        basePosition=pos,
        baseMass=0  
    )

results = []

# Process each target
for i, pos in enumerate(targets, 1):
    print(f"--- Target {i}: {np.round(pos, 4)} ---")
    solution = solver.solve_ik(pos, target_orn)
    print(f"IK Solution (gantry + 6 joints): {np.round(solution, 4)}")

    solver.apply_joint_positions(solution, steps=120)

    # Validate accuracy
    pos_error, orn_error, actual_pos, actual_orn = solver.validate_solution(solution, pos, target_orn)
    print(f"Position Error: {pos_error:.6f}, Orientation Error: {orn_error:.6f}\n")

    # Save results for this target
    results.append({
        "Target X": pos[0],
        "Target Y": pos[1],
        "Target Z": pos[2],
        "IK Gantry": solution[0],
        "IK Joint1": solution[1],
        "IK Joint2": solution[2],
        "IK Joint3": solution[3],
        "IK Joint4": solution[4],
        "IK Joint5": solution[5],
        "IK Joint6": solution[6],
        "Actual X": actual_pos[0],
        "Actual Y": actual_pos[1],
        "Actual Z": actual_pos[2],
        "Position Error": pos_error,
        "Orientation Error (rad)": orn_error
    })

# Export results to CSV
df = pd.DataFrame(results)
df.to_csv("ik_results.csv", index=False)
print("All targets processed. Results saved to 'ik_results.csv'. Closing in 5 seconds...")

time.sleep(5)
p.disconnect()
