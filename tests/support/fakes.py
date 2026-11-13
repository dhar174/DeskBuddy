"""Deterministic fakes with call recording for DeskBuddy behavioral contract tests."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any


@dataclass
class CallRecord:
    """Single recorded subsystem invocation."""

    target: str
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)


class CallRecorder:
    """Thread-safe-enough deterministic call log for asserting routing & bypasses."""

    def __init__(self) -> None:
        self.calls: list[CallRecord] = []

    def record(self, target: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append(CallRecord(target=target, args=args, kwargs=kwargs))

    def for_target(self, target: str) -> list[CallRecord]:
        return [call for call in self.calls if call.target == target]

    def count(self, target: str) -> int:
        return len(self.for_target(target))

    def targets(self) -> list[str]:
        return [call.target for call in self.calls]


class FakeNLIClassifier:
    """Deterministic NLI zero-shot classifier fake.

    Matches candidate label tuples/sets to score vectors and returns fresh dictionaries
    to guarantee caller behavior cannot depend on dict identity.
    """

    def __init__(
        self,
        recorder: CallRecorder | None = None,
        *,
        routes: dict[tuple[str, ...], list[float]] | None = None,
        reverse_keys: bool = False,
        target_name: str = "nli_classifier",
    ) -> None:
        self.recorder = recorder or CallRecorder()
        self.routes = routes or {}
        self.reverse_keys = reverse_keys
        self.target_name = target_name

    def __call__(self, text: str, candidate_labels: Sequence[str]) -> dict[str, Any]:
        labels = list(candidate_labels)
        self.recorder.record(self.target_name, text, tuple(labels))
        scores = self.routes.get(tuple(labels))
        if scores is None:
            scores = [1.0] + [0.0] * max(0, len(labels) - 1)
        if self.reverse_keys:
            return {"scores": list(scores), "labels": labels, "sequence": text}
        return {"sequence": text, "labels": labels, "scores": list(scores)}


class FakeTokenMatrix:
    """Minimal 2D token tensor replacement supporting .to(), .shape, slicing, and .tolist()."""

    def __init__(self, rows: list[list[int]]) -> None:
        self._rows = rows
        cols = len(rows[0]) if rows else 0
        self.shape = (len(rows), cols)

    def to(self, _device: Any) -> "FakeTokenMatrix":
        return self

    def unsqueeze(self, _axis: int) -> "FakeTokenMatrix":
        return self

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, tuple) and len(item) == 2:
            row_sel, col_sel = item
            selected_rows = self._rows[row_sel] if isinstance(row_sel, slice) else [self._rows[row_sel]]
            sliced = [row[col_sel] for row in selected_rows]
            return FakeTokenMatrix(sliced)
        if isinstance(item, int):
            return FakeTokenRow(self._rows[item])
        return FakeTokenMatrix(self._rows[item])

    def __iter__(self):
        return (FakeTokenRow(row) for row in self._rows)

    def tolist(self) -> list[list[int]]:
        return [list(row) for row in self._rows]


class FakeTokenRow:
    """1D row returned when indexing a FakeTokenMatrix."""

    def __init__(self, tokens: list[int]) -> None:
        self._tokens = list(tokens)

    def tolist(self) -> list[int]:
        return list(self._tokens)

    def __len__(self) -> int:
        return len(self._tokens)

    def __iter__(self):
        return iter(self._tokens)


class FakeBatchEncoding(dict[str, Any]):
    """Dictionary-like tokenizer output with .to(device)."""

    def to(self, _device: Any) -> "FakeBatchEncoding":
        return self


class FakeTokenizer:
    """Deterministic tokenizer fake for chat, question, command, and summary models."""

    def __init__(
        self,
        recorder: CallRecorder | None = None,
        *,
        name: str = "tokenizer",
        decoded_prefix: str = "FAKE_DECODED",
        eos_token: str = "<eos>",
        bos_token_id: int = 1,
    ) -> None:
        self.recorder = recorder or CallRecorder()
        self.name = name
        self.decoded_prefix = decoded_prefix
        self.eos_token = eos_token
        self.pad_token: str | None = None
        self.bos_token_id = bos_token_id
        self.cls_token_id = 101

    def __call__(self, text: Any, **_kwargs: Any) -> FakeBatchEncoding:
        self.recorder.record(f"{self.name}.__call__", text)
        if isinstance(text, list):
            rows = [[10 + i, 11 + i, 12 + i] for i, _ in enumerate(text)]
        else:
            rows = [[10, 11, 12, 13]]
        return FakeBatchEncoding(
            {
                "input_ids": FakeTokenMatrix(rows),
                "attention_mask": FakeTokenMatrix([[1] * len(row) for row in rows]),
            }
        )

    def encode(self, text: str, **_kwargs: Any) -> FakeTokenMatrix:
        self.recorder.record(f"{self.name}.encode", text)
        return FakeTokenMatrix([[10, 11, 12]])

    def decode(self, token_ids: Any, **_kwargs: Any) -> str:
        self.recorder.record(f"{self.name}.decode", token_ids)
        return f"{self.decoded_prefix}:{len(token_ids) if hasattr(token_ids, '__len__') else 0}"

    def build_inputs_with_special_tokens(self, tokens: list[Any]) -> list[Any]:
        self.recorder.record(f"{self.name}.build_inputs_with_special_tokens", list(tokens))
        return list(tokens)


class FakeCausalLM:
    """Deterministic LLM / Seq2Seq fake returning tagged token matrices."""

    def __init__(
        self,
        recorder: CallRecorder | None = None,
        *,
        name: str = "model",
        output_tokens: list[list[int]] | None = None,
    ) -> None:
        self.recorder = recorder or CallRecorder()
        self.name = name
        self.output_tokens = output_tokens or [[10, 11, 12, 99, 100]]

    def to(self, _device: Any) -> "FakeCausalLM":
        return self

    def eval(self) -> "FakeCausalLM":
        return self

    def parameters(self) -> list[Any]:
        return []

    def generate(self, *args: Any, **kwargs: Any) -> FakeTokenMatrix:
        self.recorder.record(f"{self.name}.generate", *args, **kwargs)
        return FakeTokenMatrix([list(row) for row in self.output_tokens])


class FakeLogitsArray:
    """Minimal numpy-like array supporting .detach().cpu().numpy() and argmax(axis=1)."""

    def __init__(self, values: list[list[float]]) -> None:
        self._values = values

    def detach(self) -> "FakeLogitsArray":
        return self

    def cpu(self) -> "FakeLogitsArray":
        return self

    def numpy(self) -> "FakeLogitsArray":
        return self

    def argmax(self, axis: int = 1) -> int:
        assert axis == 1
        row = self._values[0]
        return row.index(max(row))


class FakeQuestionClassifier:
    """Simulates BotCortex.isquestion_model returning logits dict."""

    def __init__(
        self,
        recorder: CallRecorder | None = None,
        *,
        is_question: bool = True,
    ) -> None:
        self.recorder = recorder or CallRecorder()
        self.is_question = is_question

    def __call__(self, encoded: Any) -> dict[str, FakeLogitsArray]:
        self.recorder.record("isquestion_model", encoded)
        logits = [[0.1, 0.9]] if self.is_question else [[0.9, 0.1]]
        return {"logits": FakeLogitsArray(logits)}


class FakeCommandModel:
    """Simulates helpers.command_model for interpret_command."""

    def __init__(
        self,
        recorder: CallRecorder | None = None,
        *,
        pred_index: int = 0,
        id2label: dict[int, str] | None = None,
    ) -> None:
        self.recorder = recorder or CallRecorder()
        self.pred_index = pred_index
        self.config = SimpleNamespace(id2label=id2label or {0: "general_greet", 1: "weather_query"})

    def __call__(self, **kwargs: Any) -> list[list[float]]:
        self.recorder.record("command_model", **kwargs)
        num_labels = max(2, len(self.config.id2label), self.pred_index + 1)
        return [[1.0 if i == self.pred_index else 0.0 for i in range(num_labels)]]


class FakeOpenAISDK:
    """Deterministic OpenAI 0.28.1 SDK fake with strict call recording."""

    def __init__(
        self,
        recorder: CallRecorder | None = None,
        *,
        chat_reply: str | Callable[[dict[str, Any]], str] = "OPENAI_CHAT_FAKE",
        completion_reply: str | Callable[[dict[str, Any]], str] = "False",
    ) -> None:
        self.recorder = recorder or CallRecorder()
        self.api_key: str | None = None
        outer = self

        class _ChatCompletion:
            @staticmethod
            def create(**kwargs: Any) -> dict[str, Any]:
                recorded = dict(kwargs)
                if "messages" in recorded and isinstance(recorded["messages"], list):
                    recorded["messages"] = [
                        dict(msg) if isinstance(msg, dict) else msg for msg in recorded["messages"]
                    ]
                outer.recorder.record("openai.ChatCompletion.create", **recorded)
                text = chat_reply(kwargs) if callable(chat_reply) else chat_reply
                return {"choices": [{"message": {"content": text}}]}

        class _Completion:
            @staticmethod
            def create(**kwargs: Any) -> dict[str, Any]:
                outer.recorder.record("openai.Completion.create", **kwargs)
                text = completion_reply(kwargs) if callable(completion_reply) else completion_reply
                return {"choices": [{"text": text}]}

        class _OpenAIError(Exception):
            pass

        self.ChatCompletion = _ChatCompletion
        self.Completion = _Completion
        self.error = SimpleNamespace(OpenAIError=_OpenAIError)


class FakeTorch:
    """Lightweight torch substitute covering helpers and BotCortex contract calls."""

    __version__ = "2.4.1-fake"

    def __init__(self, *, cuda_available: bool = False, argmax_value: int = 0) -> None:
        self._argmax_value = argmax_value
        self.version = SimpleNamespace(cuda=None)
        self.cuda = SimpleNamespace(
            is_available=lambda: cuda_available,
            amp=SimpleNamespace(autocast=lambda: nullcontext()),
            memory_allocated=lambda: 0,
            memory_reserved=lambda: 0,
        )
        self.optim = SimpleNamespace(Adam=lambda _params: object())

    @staticmethod
    def device(name: str) -> str:
        return name

    @staticmethod
    def tensor(_values: Any) -> FakeTokenMatrix:
        return FakeTokenMatrix([[1]])

    @staticmethod
    def no_grad():
        return nullcontext()

    def argmax(self, tensor_like: Any) -> SimpleNamespace:
        if isinstance(tensor_like, (list, tuple)) and tensor_like:
            val = max(range(len(tensor_like)), key=lambda idx: tensor_like[idx])
        else:
            val = self._argmax_value
        return SimpleNamespace(item=lambda: val)

    @staticmethod
    def mm(left: FakeEmbeddingVector, _right: FakeEmbeddingVector) -> FakeEmbeddingVector:
        return FakeEmbeddingVector(list(left.values))

    @staticmethod
    def sort(
        value: FakeEmbeddingVector, **_kwargs: Any
    ) -> tuple[FakeEmbeddingVector, FakeEmbeddingVector]:
        return FakeEmbeddingRuntime.sort(value, **_kwargs)


class FakeEmbeddingVector:
    """Minimal vector fake supporting search_question_in_bank tensor ops."""

    def __init__(self, values: list[float]) -> None:
        self.values = list(values)

    def detach(self) -> "FakeEmbeddingVector":
        return self

    def transpose(self, *_args: Any) -> "FakeEmbeddingVector":
        return self

    def squeeze(self) -> "FakeEmbeddingVector":
        return self

    def size(self, _dimension: int) -> int:
        return len(self.values)

    def __getitem__(self, index: int | slice) -> Any:
        if isinstance(index, slice):
            return FakeEmbeddingVector(self.values[index])
        return FakeScalar(self.values[index])


class FakeScalar:
    """Scalar element extracted from FakeEmbeddingVector."""

    def __init__(self, value: float | int) -> None:
        self.value = value

    def item(self) -> float:
        return float(self.value)

    def __index__(self) -> int:
        return int(self.value)


class FakeEmbeddingRuntime:
    """Provides deterministic similarity scores for search_question_in_bank."""

    def __init__(
        self,
        recorder: CallRecorder | None = None,
        *,
        scores: list[float] | None = None,
    ) -> None:
        self.recorder = recorder or CallRecorder()
        self.scores = scores or [0.95, 0.2]

    async def get_embeddings(self, value: Any) -> FakeEmbeddingVector:
        self.recorder.record("get_embeddings", value)
        return FakeEmbeddingVector(self.scores)

    @staticmethod
    def normalize(value: FakeEmbeddingVector, **_kwargs: Any) -> FakeEmbeddingVector:
        return value

    def mm(self, _left: FakeEmbeddingVector, _right: FakeEmbeddingVector) -> FakeEmbeddingVector:
        return FakeEmbeddingVector(list(self.scores))

    @staticmethod
    def sort(
        value: FakeEmbeddingVector, **_kwargs: Any
    ) -> tuple[FakeEmbeddingVector, FakeEmbeddingVector]:
        pairs = sorted(enumerate(value.values), key=lambda pair: pair[1], reverse=True)
        return (
            FakeEmbeddingVector([score for _idx, score in pairs]),
            FakeEmbeddingVector([float(idx) for idx, _score in pairs]),
        )


class FakeNLP:
    """Deterministic spaCy nlp(text) fake returning PERSON entities."""

    def __init__(
        self,
        recorder: CallRecorder | None = None,
        *,
        entities: list[tuple[str, str]] | None = None,
    ) -> None:
        self.recorder = recorder or CallRecorder()
        self.entities = entities or []

    def __call__(self, text: str) -> SimpleNamespace:
        self.recorder.record("nlp", text)
        ents = [SimpleNamespace(text=ent_text, label_=label) for ent_text, label in self.entities]
        return SimpleNamespace(ents=ents)


class FakeCountVectorizer:
    """Minimal CountVectorizer replacement for extract_topics."""

    def __init__(self) -> None:
        self._features = ["zelda", "cats", "games"]

    def fit_transform(self, words: list[str]) -> list[str]:
        return words

    def get_feature_names_out(self) -> list[str]:
        return self._features


class FakeLDA:
    """Minimal LatentDirichletAllocation replacement for extract_topics."""

    def __init__(self, recorder: CallRecorder | None = None) -> None:
        self.recorder = recorder or CallRecorder()
        self.components_ = [FakeLDAArray([0.9, 0.5, 0.2])]

    def fit(self, _matrix: Any) -> "FakeLDA":
        self.recorder.record("lda.fit")
        return self


class FakeLDAArray:
    """Row inside FakeLDA.components_ supporting .argsort()."""

    def __init__(self, weights: list[float]) -> None:
        self._weights = weights

    def argsort(self) -> list[int]:
        return sorted(range(len(self._weights)), key=lambda i: self._weights[i])
