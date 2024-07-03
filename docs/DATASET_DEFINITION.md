## Dataset Definition

WIP

```python
class MapDataset(Dataset):
    def __init__(self, *args, **kwargs):
        pass

    def load(self):
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Tuple :
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return
```
