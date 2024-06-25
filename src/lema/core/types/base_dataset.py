class BaseDataset:
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
