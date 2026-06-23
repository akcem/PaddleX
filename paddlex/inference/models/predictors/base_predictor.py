# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, ABCMeta, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

from ....utils import logging
from ....utils.flags import (
    INFER_BENCHMARK,
    INFER_BENCHMARK_ITERS,
    INFER_BENCHMARK_WARMUP,
    PIPELINE_BENCHMARK,
)
from ...common.batch_sampler import BaseBatchSampler
from ...utils.benchmark import ENTRY_POINT_NAME, benchmark


class BasePredictor(ABC, metaclass=ABCMeta):
    """Abstract predictor interface."""

    def __init__(
        self,
        *,
        model_name: str = "",
        engine_config: Optional[Dict[str, Any]] = None,
        batch_size: int = 1,
        **kwargs: Any,
    ) -> None:
        del kwargs
        self.model_name = model_name
        self._engine_config = dict(engine_config or {})

        self.batch_sampler = self._build_batch_sampler()
        self.result_class = self._get_result_class()

        # alias predict() to the __call__()
        self.predict = self.__call__

        self.batch_sampler.batch_size = batch_size

    @property
    def engine_config(self) -> Dict[str, Any]:
        return dict(self._engine_config)

    @property
    def supports_benchmark(self) -> bool:
        return True

    def __call__(
        self,
        input: Any,
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        """Default: delegate to apply."""
        if batch_size is not None:
            self.batch_sampler.batch_size = batch_size

        benchmark_enabled = INFER_BENCHMARK or PIPELINE_BENCHMARK
        if benchmark_enabled and not self.supports_benchmark:
            logging.warning(
                "%s does not support benchmark, but benchmark is enabled. Skipping.",
                self.__class__.__name__,
            )
        elif INFER_BENCHMARK:
            # TODO(zhang-prog): Get metadata of input data
            @benchmark.timeit_with_options(name=ENTRY_POINT_NAME)
            def _apply(input, **kwargs):
                return list(self.apply(input, **kwargs))

            if isinstance(input, list):
                raise TypeError("`input` cannot be a list in benchmark mode")

            if batch_size is None:
                batch_size = 1
            input = [input] * batch_size

            if not (INFER_BENCHMARK_WARMUP > 0 or INFER_BENCHMARK_ITERS > 0):
                raise RuntimeError(
                    "At least one of `INFER_BENCHMARK_WARMUP` and `INFER_BENCHMARK_ITERS` must be greater than zero"
                )

            benchmark.reset()
            if INFER_BENCHMARK_WARMUP > 0:
                benchmark.start_warmup()
                for _ in range(INFER_BENCHMARK_WARMUP):
                    output = _apply(input, **kwargs)
                benchmark.collect(batch_size)
                benchmark.stop_warmup()

            if INFER_BENCHMARK_ITERS > 0:
                for _ in range(INFER_BENCHMARK_ITERS):
                    output = _apply(input, **kwargs)
                benchmark.collect(batch_size)

            yield output[0]
        elif PIPELINE_BENCHMARK:

            @benchmark.timeit_with_options(name=type(self).__name__ + ".apply")
            def _apply(input, **kwargs):
                return list(self.apply(input, **kwargs))

            yield from _apply(input, **kwargs)
        else:
            yield from self.apply(input, **kwargs)

    def _get_batch_field_size(self, value: Any) -> int:
        """
        【推理结果】【批次拆分】获取字段值对应的 batch 条数。

        Args:
            value: process 返回的单个字段值。

        Returns:
            int: 字段值可拆分的 batch 条数，无法拆分时返回 1。
        """
        if isinstance(value, list):
            return len(value)
        if isinstance(value, np.ndarray) and value.ndim > 0:
            return value.shape[0]
        return 1

    def _get_single_batch_field(self, value: Any, idx: int, batch_size: int) -> Any:
        """
        【推理结果】【批次拆分】获取当前样本对应的单条字段值。

        Args:
            value: process 返回的单个字段值。
            idx: 当前样本在 batch 中的序号。
            batch_size: 当前 batch 的样本数量。

        Returns:
            Any: 如果字段是按 batch 组织的 list 或 ndarray，则返回第 idx 条，否则返回原值。
        """
        if isinstance(value, list) and idx < len(value):
            return value[idx]
        if (
            isinstance(value, np.ndarray)
            and value.ndim > 0
            and value.shape[0] == batch_size
            and idx < value.shape[0]
        ):
            return value[idx]
        return value

    def apply(self, input: Any, **kwargs: Any) -> Iterator[Any]:
        """Default implementation: batch_sampler -> process -> wrap with result_class.

        Handles two process return formats:
        1. pred["result"] is a list of per-item results
        2. pred is a dict of batch fields (e.g. input_path, class_ids, scores) - split by index
        """
        if INFER_BENCHMARK:
            if not isinstance(input, list):
                raise TypeError("In benchmark mode, `input` must be a list")
            batches = list(self.batch_sampler(input))
            if len(batches) != 1 or len(batches[0]) != len(input):
                raise ValueError("Unexpected number of instances")
        else:
            batches = self.batch_sampler(input)

        for batch_data in batches:
            if hasattr(batch_data, "instances"):
                input_paths = getattr(batch_data, "input_paths", None)
            else:
                input_paths = None
            pred = self.process(batch_data, **kwargs)
            results = pred.get("result", pred)
            if isinstance(results, list):
                n = len(results)
                for idx, single in enumerate(results):
                    item = {"result": single}
                    for k, v in pred.items():
                        if k == "result":
                            continue
                        item[k] = self._get_single_batch_field(v, idx, n)
                    if input_paths and idx < len(input_paths):
                        item["input_path"] = input_paths[idx]
                    yield self.result_class(item)
            else:
                first_val = next(iter(pred.values()), None)
                n = (
                    len(input_paths)
                    if input_paths is not None
                    else self._get_batch_field_size(first_val)
                )
                for idx in range(n):
                    item = {}
                    for k, v in pred.items():
                        item[k] = self._get_single_batch_field(v, idx, n)
                    if input_paths and idx < len(input_paths):
                        item["input_path"] = input_paths[idx]
                    yield self.result_class(item)

    @abstractmethod
    def process(self, batch_data: List[Any]) -> Dict[str, List[Any]]:
        raise NotImplementedError

    @abstractmethod
    def _build_batch_sampler(self) -> BaseBatchSampler:
        raise NotImplementedError

    @abstractmethod
    def _get_result_class(self) -> type:
        raise NotImplementedError

    def close(self) -> None:
        pass

    @classmethod
    def get_config_path(cls, model_dir: Path) -> Path:
        from ..utils.model_config import get_model_config_path

        return get_model_config_path(model_dir)

    @classmethod
    def load_config(cls, model_dir: Path) -> Dict:
        from ..utils.model_config import load_model_config

        return load_model_config(model_dir)
