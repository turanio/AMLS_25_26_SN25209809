from abc import ABC, abstractmethod
from utils.logger import CustomLogger
from utils.seed import set_seed
import time


class BaseTrainer(ABC):
    def __init__(
        self,
        model_name: str,
        log_dir: str,
        seed: int = 42,
    ):
        self.model_name = model_name
        self.seed = seed

        set_seed(seed)

        self.logger = CustomLogger().get_logger(
            name=f"{model_name}.Trainer",
            log_file=f"{log_dir}/train.log"
        )

        self.output_logger = CustomLogger().get_logger(
            name=f"{model_name}.Outputs",
            log_file=f"{log_dir}/outputs.log"
        )

        self.logger.info("Trainer initialized")
        self.logger.info(f"Seed set to {seed}")

    def run(self, data: dict):
        """
        Template method defining the training workflow.
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
    def _build_model(self):
        pass

    @abstractmethod
    def _train(self, data: dict):
        pass

    @abstractmethod
    def _evaluate(self, data: dict) -> dict:
        pass

    def _log_results(self, results: dict):
        self.output_logger.info("Final results:")
        for key, value in results.items():
            self.output_logger.info(f"{key}: {value}")
