from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

from deskbuddy.settings import Settings

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LEGACY_MODEL_DEFAULTS = {
    "medium_model_id": "PygmalionAI/pygmalion-350m",
    "fast_model_id": "microsoft/DialoGPT-large",
    "small_model_id": "facebook/blenderbot_small-90M",
    "blenderbot_model_id": "facebook/blenderbot-400M-distill",
    "vision_model_id": "microsoft/git-large-vqav2",
    "topic_model_id": "knkarthick/TOPIC-DIALOGSUM",
    "summary_model_id": "philschmid/bart-large-cnn-samsum",
    "short_summary_model_id": "knkarthick/meeting-summary-samsum",
    "title_model_id": "fabiochiu/t5-small-medium-title-generation",
    "spacy_model_id": "en_core_web_trf",
    "question_classifier_model_id": "shahrukhx01/question-vs-statement-classifier",
    "nli_model_id": "valhalla/distilbart-mnli-12-3",
    "command_model_id": "gokuls/bert-tiny-Massive-intent-KD-BERT",
    "question_generator_model_id": "voidful/context-only-question-generator",
    "embedding_model_id": "sentence-transformers/all-mpnet-base-v2",
    "completion_model_id": "text-curie-001",
    "sd_classifier_model_id": "curie",
    "qa_model_id": "distilbert-base-cased-distilled-squad",
}


def test_model_identifier_defaults_preserve_legacy_runtime_choices() -> None:
    settings = Settings()

    assert {name: getattr(settings, name) for name in LEGACY_MODEL_DEFAULTS} == (
        LEGACY_MODEL_DEFAULTS
    )


def test_env_and_toml_examples_document_exact_settings_schema() -> None:
    fields = set(Settings.model_fields)
    env_text = (REPOSITORY_ROOT / ".env.example").read_text(encoding="utf-8")
    env_names = set(re.findall(r"^\s*#?\s*([A-Z][A-Z0-9_]*)\s*=", env_text, re.MULTILINE))
    env_names -= {"DESKBUDDY_CONFIG_FILE", "DESKBUDDY_ENV_FILE"}
    expected_env_names = {
        "OPENAI_API_KEY" if field == "openai_api_key" else f"DESKBUDDY_{field.upper()}"
        for field in fields
    }
    assert env_names == expected_env_names

    toml_text = (REPOSITORY_ROOT / "deskbuddy.example.toml").read_text(encoding="utf-8")
    toml_names = set(re.findall(r"^\s*#?\s*([a-z][a-z0-9_]*)\s*=", toml_text, re.MULTILINE))
    assert toml_names == fields

    parsed = tomllib.loads(toml_text)
    assert set(parsed) == {"deskbuddy"}
    assert set(parsed["deskbuddy"]) <= fields
    assert Settings.model_validate(parsed["deskbuddy"]).model_provider == "disabled"


def _is_settings_model_or_path_reference(expression: ast.expr) -> bool:
    if (
        isinstance(expression, ast.Attribute)
        and isinstance(expression.value, ast.Name)
        and expression.value.id == "settings"
    ):
        return expression.attr in Settings.model_fields
    if (
        isinstance(expression, ast.Call)
        and isinstance(expression.func, ast.Name)
        and expression.func.id == "str"
        and len(expression.args) == 1
        and isinstance(expression.args[0], ast.Name)
    ):
        return expression.args[0].id in {"checkpoint_dir", "tokenizer_dir"}
    return False


def test_active_model_loaders_use_settings_or_explicit_model_paths() -> None:
    runtime_files = [REPOSITORY_ROOT / "BotCortex.py", REPOSITORY_ROOT / "helpers.py"]
    violations: list[str] = []

    for path in runtime_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Attribute) and node.func.attr == "from_pretrained":
                if not node.args or not _is_settings_model_or_path_reference(node.args[0]):
                    violations.append(
                        f"{path.name}:{node.lineno} from_pretrained has a literal/unknown source"
                    )
            if isinstance(node.func, ast.Name) and node.func.id == "pipeline":
                model_argument = next(
                    (keyword.value for keyword in node.keywords if keyword.arg == "model"), None
                )
                if model_argument is None or not _is_settings_model_or_path_reference(
                    model_argument
                ):
                    violations.append(
                        f"{path.name}:{node.lineno} pipeline model is not settings-backed"
                    )

    assert not violations, "\n".join(violations)
