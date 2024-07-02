from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self) -> None:
        """Initializes a new instance of the BaseDataset class."""

    def __getitem__(self, idx: int) -> dict:
        """Get the item at the specified index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            dict: The item at the specified index.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """Gets the number of items in the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        raise NotImplementedError
