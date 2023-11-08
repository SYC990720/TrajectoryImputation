from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import torch
import csv


class TrajectoryDataset(Dataset):
    def __init__(self, file_path, mode, missing_rate):
        self.trajectories = self._extract_trajectories(file_path, mode)
        self.mode = mode
        self.missing_rate = missing_rate

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, index):
        data = self.trajectories[index]
        complete_trajectory = torch.tensor(data[1:], dtype=torch.float32)
        incomplete_trajectory = self.mask_trajectory(complete_trajectory)
        mode = data[0]
        return incomplete_trajectory, complete_trajectory, mode

    def mask_trajectory(self, traj):
        mask = torch.rand_like(traj) > self.missing_rate
        masked_traj = traj * mask.float()

        return masked_traj

    def _extract_trajectories(self, file_path, mode):
        trajectories = []
        current_trajectory = []

        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)

            prev_lat, prev_lon = None, None

            for row in reader:
                if row[4] == 's':
                    if current_trajectory:
                        trajectories.append(current_trajectory)
                    current_trajectory = []
                    current_trajectory.append(mode)
                    prev_lat, prev_lon = None, None
                else:
                    lat, lon = float(row[1]), float(row[2])

                    if prev_lat is not None and prev_lon is not None:
                        lat_diff = lat - prev_lat
                        lon_diff = lon - prev_lon
                        current_trajectory.append([lat_diff*1e4, lon_diff*1e4])

                    prev_lat, prev_lon = lat, lon

        if current_trajectory:
            trajectories.append(current_trajectory)

        return trajectories

def collate_fn(batch):
    incomplete_traj = [item[0] for item in batch]
    complete_traj = [item[1] for item in batch]
    mode = [item[2] for item in batch]

    padded_inputs = pad_sequence(incomplete_traj, batch_first=True)
    padded_targets = pad_sequence(complete_traj, batch_first=True)
    mode = torch.tensor(mode)

    return padded_inputs, padded_targets, mode
    
