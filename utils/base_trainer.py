import time
from abc import ABC, abstractmethod
from typing import Dict

from utils.logger import CustomLogger
from utils.seed import set_seed


class BaseTrainer(ABC):
    """Abstract base trainer implementing the template method pattern.

    Provides a standardized workflow for model training, including logging,
    seeding, and evaluation. Subclasses must implement the abstract methods
    for building, training, and evaluating their specific models.

    Attributes:
        model_name: Name of the model being trained.
        seed: Random seed for reproducibility.
        logger: Logger for training information.
        output_logger: Logger for results output.
    """

    def __init__(
        self,
        model_name: str,
        log_dir: str,
        seed: int = 42,
    ) -> None:
        """Initialize the base trainer.

        Args:
            model_name: Name of the model for logging purposes.
            log_dir: Directory path for log files.
            seed: Random seed for reproducibility. Defaults to 42.
        """
        self.model_name = model_name
        self.seed = seed

        set_seed(seed)

        self.logger = CustomLogger().get_logger(
            name=f"{model_name}.Trainer", log_file=f"{log_dir}/train.log"
        )

        self.output_logger = CustomLogger().get_logger(
            name=f"{model_name}.Outputs", log_file=f"{log_dir}/outputs.log"
        )

        self.logger.info("Trainer initialized")
        self.logger.info(f"Seed set to {seed}")

    def run(self, data: Dict) -> Dict[str, float]:
        """Execute the complete training workflow.

        Template method that orchestrates model building, training, and evaluation.
        Subclasses should not override this method but instead implement the
        abstract methods it calls.

        Args:
            data: Dictionary containing 'train', 'val', and 'test' splits,
                  where each split is a tuple (X, y).

        Returns:
            Dictionary of evaluation metrics from _evaluate().
        """
        self.logger.info("Run started")
        start_time = time.time()

        self._build_model()
        self._train(data)
        results = self._evaluate(data)

        elapsed = time.time() - start_time
        self.logger.info(f"Run finished in {elapsed:.2f} seconds")

        self._log_results(results)
        return results

    @abstractmethod
    def _build_model(self) -> None:
        """Build the model.

        Subclasses must implement this method to initialize their model,
        optimizer, and any other training components.
        """
        pass

    @abstractmethod
    def _train(self, data: Dict) -> None:
        """Train the model.

        Subclasses must implement this method to define their training loop.

        Args:
            data: Dictionary containing 'train', 'val', and 'test' splits,
                  where each split is a tuple (X, y).
        """
        pass

    @abstractmethod
    def _evaluate(self, data: Dict) -> Dict[str, float]:
        """Evaluate the model.

        Subclasses must implement this method to compute evaluation metrics.

        Args:
            data: Dictionary containing 'train', 'val', and 'test' splits,
                  where each split is a tuple (X, y).

        Returns:
            Dictionary of evaluation metrics (e.g., accuracy, precision, etc.).
        """
        pass

    def _log_results(self, results: Dict[str, float]) -> None:
        """Log evaluation results.

        Args:
            results: Dictionary of metric names and values.
        """
        self.output_logger.info("Final results:")
        for key, value in results.items():
            self.output_logger.info(f"{key}: {value:.4f}")
