from trl import DPOTrainer


class TrlDpoTrainer(DPOTrainer):
    """Light wrapper around the DPOTrainer to handle vision models."""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """Initializes the TrlDpoTrainer."""
        super().__init__(*args, **kwargs)

    def _prepare_dataset(self, dataset, processing_class, args, dataset_name):
        self.is_vision_model = True

        return super()._prepare_dataset(dataset, processing_class, args, dataset_name)
