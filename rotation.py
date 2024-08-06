import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def slerp(p0, p1, t):
    """
    Spherical linear interpolation between two rotations.
    """
    dot = np.dot(p0, p1)
    if dot < 0.0:
        p1 = -p1
        dot = -dot

    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        result = p0 + t * (p1 - p0)
        result /= np.linalg.norm(result)
        return result

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)

    theta = theta_0 * t
    sin_theta = np.sin(theta)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    return (s0 * p0) + (s1 * p1)

def plot_coordinate_frame(ax, origin, R, label, color):
    """Plot a coordinate frame given an origin and rotation matrix."""
    x_axis = R[:, 0]
    y_axis = R[:, 1]
    z_axis = R[:, 2]

    ax.quiver(*origin, *x_axis, color=color, length=1.0, normalize=True)
    ax.quiver(*origin, *y_axis, color=color, length=1.0, normalize=True)
    ax.quiver(*origin, *z_axis, color=color, length=1.0, normalize=True)

    ax.text(*(origin + x_axis), f'{label}x', color=color)
    ax.text(*(origin + y_axis), f'{label}y', color=color)
    ax.text(*(origin + z_axis), f'{label}z', color=color)

def update(frame, ax, origin, rotations, n_frames):
    ax.cla()
    num_rotations = len(rotations)
    current_index = frame // n_frames
    next_index = (current_index + 1) % num_rotations

    t = (frame % n_frames) / n_frames
    R_start = R.from_matrix(rotations[current_index])
    R_end = R.from_matrix(rotations[next_index])

    R_current = R_start * R.from_quat(slerp(R_start.as_quat(), R_end.as_quat(), t))
    plot_coordinate_frame(ax, origin, R_current.as_matrix(), 'R', 'b')
    plot_coordinate_frame(ax, origin, np.eye(3), 'O', 'r')

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

def main():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Original coordinate frame
    origin = np.array([0, 0, 0])

    # List of rotation matrices
    rotations = [
        np.eye(3),
        rotation_matrix([0, 0, 1], np.pi / 4),  # Rotate 45 degrees around z-axis
        rotation_matrix([0, 1, 0], np.pi / 4),  # Rotate 45 degrees around y-axis
        rotation_matrix([1, 0, 0], np.pi / 4)   # Rotate 45 degrees around x-axis
    ]

    # Number of frames in the animation for each transition
    n_frames = 100

    ani = FuncAnimation(fig, update, frames=n_frames * len(rotations), fargs=(ax, origin, rotations, n_frames), interval=50, repeat=True)
    plt.show()

if __name__ == '__main__':
    main()
