from typing import Dict, List

from torch.utils.data import Dataset


class AlfWorldEpisodeDataset(Dataset):
    """A lightweight epoch clock for environment-sampled ALFWorld episodes.

    ALFWorld chooses a new game whenever its batched environment is reset, so
    dataset rows only determine how many episodes are collected in one epoch.
    """

    def __init__(self, num_episodes: int):
        if num_episodes <= 0:
            raise ValueError("num_episodes must be positive")
        self.num_episodes = int(num_episodes)

    def __len__(self) -> int:
        return self.num_episodes

    def __getitem__(self, index: int) -> Dict[str, int]:
        return {"episode_slot": index}

    @staticmethod
    def collate_fn(batch: List[Dict[str, int]]) -> List[Dict[str, int]]:
        return batch
