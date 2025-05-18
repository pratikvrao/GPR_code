import os
import pandas as pd
from scipy import io


def load_trajectory_data(file_path, variable_suffix, lat_key):
    """
    Load and flatten trajectory variables from a .mat file.
    Example: loading post-processed data from EUROCONTROL and AirTraf
    """
    data = io.loadmat(file_path)
    return pd.DataFrame({
        f"t_{variable_suffix}": data[f"t_{variable_suffix}"].flatten('F'),
        f"g_{variable_suffix}": data[f"g_{variable_suffix}"].flatten('F'),
        f"zi0_{variable_suffix}": data[f"zi0_{variable_suffix}"].flatten('F'),
        f"u_{variable_suffix}": data[f"u_{variable_suffix}"].flatten('F'),
        f"Rl_{variable_suffix}": data[lat_key].flatten('F')
    })

# Define trajectory file info [currently for 4 trajectories, see Fig 5a]
trajectory_info = [
    ('traj1.mat', '1', 'flight_latitude_f1'),
    ('traj2.mat', '2', 'flight_latitude_f2'),
    ('traj3.mat', '3', 'flight_latitude_f3'),
    ('traj4_AirTraf.mat', '4', 'flight_latitude_f4')
]

# Load all trajectories into a dictionary of DataFrames
trajectory_data = {
    f"X_traj{suffix}": load_trajectory_data(
        os.path.join('trajectory_data', filename), suffix, lat_key
    )
    for filename, suffix, lat_key in trajectory_info
}


