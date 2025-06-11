import os
import signal
import logging
from abc import ABC, abstractmethod
from argparse import Namespace, ArgumentParser
from typing import Any, Dict, List
from utils.model_loader import LLM
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.helpers import calculate_average_scores, calculate_average_scores_dict
import numpy as np
logger = logging.getLogger("evaluation")
np.random.seed(42)

class BaseEvaluator(ABC):
    """
    Every concrete evaluator must implement:
        load_data()      -> iterable
        evaluate_one(ex) -> dict with at least 'score' key
    """

    NAME: str = "base"

    def __init__(self, args: Namespace):
        self.args = args

        self.cand = LLM(
            args.model_path,
            api_key=os.getenv("CAND_API_KEY"),
            base_url=os.getenv("CAND_BASE_URL"),
        )
        if self.args.judge_model:
            self.judge = LLM(
                args.judge_model,
                api_key=os.getenv("JUDGE_API_KEY"),
                base_url=os.getenv("JUDGE_BASE_URL"),
            )

    @classmethod
    @abstractmethod
    def add_args(cls, p: ArgumentParser): ...
    @abstractmethod
    def load_data(self): ...
    @abstractmethod
    def evaluate_one(self, example) -> Dict[str, Any]: ...

    def run(self):
        data = list(self.load_data())
        results = []

        # We want CTRL+C to stop the whole thing immediately, even while the
        # executor threads are blocked in network I/O.  A small trick is to
        # temporarily ignore SIGINT while we create the pool, then restore it.
        original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        executor = ThreadPoolExecutor(max_workers=self.args.workers)
        signal.signal(signal.SIGINT, original_sigint_handler)

        try:
            futures = [executor.submit(self.evaluate_one, ex) for ex in data]

            for future in as_completed(futures):
                res = future.result()
                results.append(res)
                scores_string=calculate_average_scores(results)
                logger.info(
                    f"{self.NAME}: {len(results)}/{len(data)} " f"scores={scores_string}"
                )

        except KeyboardInterrupt:
            logger.warning("Interrupted by user – cancelling leftover jobs…")
            for f in futures:
                f.cancel()
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

            if results:
                self._dump(results)

        return results

    def _dump(self, res):
        if res: 
            with open(f"results_{self.NAME}.txt", "a") as f:
                scores_string = calculate_average_scores(res)
                # Prepare to write all arguments
                args_dict = vars(self.args)  # Convert Namespace to a dictionary
                args_string = ', '.join([f"{key}={value}" for key, value in args_dict.items()])  # Create a string of all args
                f.write(f"{self.args.model_path}  scores={scores_string}  args={{ {args_string} }}\n")


