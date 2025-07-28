import numpy as np
import open3d as o3d

# Load input files
depth_path = "one-box.depth.npdata.npy"
color_path = "one-box.color.npdata.npy"
intrinsics_path = "intrinsics.npy"
extrinsics_path = "extrinsics.npy"

depth = np.load(depth_path)        
color = np.load(color_path)        
K = np.load(intrinsics_path)      
extrinsics = np.load(extrinsics_path)  

H, W = depth.shape
print(f"Loaded depth map of shape {depth.shape}, color map of shape {color.shape}")
print("Intrinsics:\n", K)
print("Extrinsics (camera to world):\n", extrinsics)

fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]

# Back-project to 3D
u, v = np.meshgrid(np.arange(W), np.arange(H))
u, v = u.flatten(), v.flatten()
z = depth.flatten()

# Handle colors (fallback to white if mismatch)
if color.ndim != 3 or color.shape[:2] != (H, W):
    print("Warning: Color map doesn't match depth. Using white for all points.")
    colors = np.ones((H * W, 3))
else:
    colors = color.reshape(-1, 3) / 255.0

# Filter valid points
valid = z > 0
u, v, z, colors = u[valid], v[valid], z[valid], colors[valid]

x = (u - cx) * z / fx
y = (v - cy) * z / fy
points = np.vstack((x, y, z, np.ones_like(z)))

points_world = extrinsics @ points
points_world = points_world[:3, :].T 

# Build point cloud and filter
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_world)
pcd.colors = o3d.utility.Vector3dVector(colors)

# Basic statistical outlier removal
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
points_np = np.asarray(pcd.points)

# PCA method (for export only)
center_pca = points_np.mean(axis=0)
points_centered = points_np - center_pca
cov = np.cov(points_centered.T)
eigvals, eigvecs = np.linalg.eigh(cov)
order = np.argsort(eigvals)[::-1]
eigvecs = eigvecs[:, order]
if np.linalg.det(eigvecs) < 0:
    eigvecs[:, -1] *= -1
pose_matrix_pca = np.eye(4)
pose_matrix_pca[:3, :3] = eigvecs
pose_matrix_pca[:3, 3] = center_pca
np.savetxt("estimated_box_pose_pca.txt", pose_matrix_pca)

# OBB method (for export and visualization)
obb = pcd.get_oriented_bounding_box()
obb.color = (1, 0, 0)  
center_obb = obb.center
R_obb = obb.R
pose_matrix_obb = np.eye(4)
pose_matrix_obb[:3, :3] = R_obb
pose_matrix_obb[:3, 3] = center_obb
np.savetxt("estimated_box_pose_obb.txt", pose_matrix_obb)

# Results
print("\nEstimated Box Pose (PCA) saved to 'estimated_box_pose_pca.txt':")
print(pose_matrix_pca)
print("\nEstimated Box Pose (OBB) saved to 'estimated_box_pose_obb.txt':")
print(pose_matrix_obb)
print(f"\nPoint cloud contains {len(points_np)} points after filtering.")

# Visualize (OBB only)
axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=center_obb)
o3d.visualization.draw_geometries(
    [pcd, obb, axes],
    window_name="Box Pose Estimation (OBB)",
    width=1280,
    height=720
)
