#!/usr/bin/env python3

"""Run mini-SWE-agent on SWE-bench instances in batch mode."""
# Read this first: https://mini-swe-agent.com/latest/usage/swebench/  (usage docs)

import concurrent.futures
import json
import os
import random
import re
import threading
import time
import traceback
from pathlib import Path

import typer
import yaml
from jinja2 import StrictUndefined, Template
from rich.live import Live

from minisweagent import Environment
from minisweagent.agents.default import DefaultAgent
from minisweagent.config import builtin_config_dir, get_config_from_spec, get_config_path
from minisweagent.environments import get_environment
from minisweagent.exceptions import InterruptAgentFlow
from minisweagent.models import get_model
from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager
from minisweagent.utils.log import add_file_handler, logger
from minisweagent.utils.serialize import UNSET, recursive_merge

_HELP_TEXT = """Run mini-SWE-agent on SWEBench instances.

[not dim]
More information about the usage: [bold green]https://mini-swe-agent.com/latest/usage/swebench/[/bold green]
[/not dim]
"""

_CONFIG_SPEC_HELP_TEXT = """Path to config files, filenames, or key-value pairs.

[bold red]IMPORTANT:[/bold red] [red]If you set this option, the default config file will not be used.[/red]
So you need to explicitly set it e.g., with [bold green]-c swebench.yaml <other options>[/bold green]

Multiple configs will be recursively merged.

Examples:

[bold red]-c model.model_kwargs.temperature=0[/bold red] [red]You forgot to add the default config file! See above.[/red]

[bold green]-c swebench.yaml -c model.model_kwargs.temperature=0.5[/bold green]

[bold green]-c swebench.yaml -c agent.max_iterations=50[/bold green]
"""

DEFAULT_CONFIG_FILE = builtin_config_dir / "benchmarks" / "swebench.yaml"

DATASET_MAPPING = {
    "full": "princeton-nlp/SWE-Bench",
    "verified": "princeton-nlp/SWE-Bench_Verified",
    "lite": "princeton-nlp/SWE-Bench_Lite",
    "multimodal": "princeton-nlp/SWE-Bench_Multimodal",
    "multilingual": "swe-bench/SWE-Bench_Multilingual",
    "smith": "SWE-bench/SWE-smith",
    "_test": "klieret/swe-bench-dummy-test-dataset",
    "rebench": "nebius/SWE-rebench",
}

app = typer.Typer(rich_markup_mode="rich", add_completion=False)
_OUTPUT_FILE_LOCK = threading.Lock()

_MAX_CONTEXT_ITEMS = 12
_MAX_HINTS_CHARS = 1200


def _parse_key_value_value(raw_value: str):
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        return raw_value


def _extract_cli_model_override(config_spec: list[str], model_option: str | None) -> str | None:
    if model_option:
        return model_option
    for spec in reversed(config_spec):
        if isinstance(spec, str) and spec.startswith("model.model_name="):
            return str(_parse_key_value_value(spec.split("=", 1)[1]))
    return None


def _first_yaml_spec(config_spec: list[str]) -> str | None:
    for spec in config_spec:
        if isinstance(spec, str) and "=" not in spec:
            return spec
    return None


def maybe_sync_model_name_to_config(
    config_spec: list[str],
    *,
    model_option: str | None,
    merged_config: dict,
    sync_enabled: bool,
) -> None:
    """Optionally persist the final model name into the first YAML config spec."""
    if not sync_enabled:
        return

    cli_model_name = _extract_cli_model_override(config_spec, model_option)
    if not cli_model_name:
        logger.info("Model sync enabled, but no CLI model override was provided. Skipping sync.")
        return

    yaml_spec = _first_yaml_spec(config_spec)
    if not yaml_spec:
        logger.warning("Model sync enabled, but no YAML config spec was provided. Skipping sync.")
        return

    try:
        config_path = get_config_path(yaml_spec)
        config_data = yaml.safe_load(config_path.read_text()) or {}
        if not isinstance(config_data, dict):
            logger.warning("Config file is not a YAML mapping, cannot sync model name: %s", config_path)
            return
        effective_model_name = merged_config.get("model", {}).get("model_name") if isinstance(merged_config, dict) else None
        if not effective_model_name:
            logger.warning("Cannot determine effective model name for sync. Skipping sync.")
            return
        model_block = config_data.setdefault("model", {})
        if not isinstance(model_block, dict):
            logger.warning("Config key 'model' is not a mapping in %s, cannot sync model name.", config_path)
            return
        previous_model_name = model_block.get("model_name")
        if previous_model_name == effective_model_name:
            logger.info("Config model.model_name already matches runtime model: %s", effective_model_name)
            return
        model_block["model_name"] = effective_model_name
        config_path.write_text(yaml.safe_dump(config_data, sort_keys=False))
        logger.info(
            "Synchronized config model.model_name in %s: %s -> %s",
            config_path,
            previous_model_name,
            effective_model_name,
        )
    except Exception as e:
        logger.warning("Failed to synchronize model.model_name to config: %s", e)


def normalize_model_provider_prefix(config: dict) -> None:
    """Normalize model provider prefix for OpenAI-compatible endpoints.

    If model_name has no explicit provider prefix but an OpenAI-compatible
    base URL is configured, prefix with "openai/" to avoid LiteLLM provider
    resolution errors.
    """
    if not isinstance(config, dict):
        return
    model_block = config.get("model")
    if not isinstance(model_block, dict):
        return

    model_name = model_block.get("model_name")
    if not isinstance(model_name, str) or not model_name.strip():
        return

    provider_prefixes = (
        "openai/",
        "anthropic/",
        "gemini/",
        "ollama/",
        "azure/",
        "vertex_ai/",
        "bedrock/",
        "groq/",
        "mistral/",
        "cohere/",
        "deepseek/",
        "xai/",
        "fireworks_ai/",
        "together_ai/",
        "openrouter/",
    )
    if model_name.startswith(provider_prefixes):
        return

    model_kwargs = model_block.get("model_kwargs")
    if not isinstance(model_kwargs, dict):
        return
    base_url = model_kwargs.get("base_url") or model_kwargs.get("api_base")
    if not isinstance(base_url, str) or not base_url.strip():
        return

    normalized_model_name = model_name
    if model_name.startswith("qwen/"):
        normalized_model_name = model_name.split("/", 1)[1]

    model_block["model_name"] = f"openai/{normalized_model_name}"
    logger.info(
        "Auto-normalized model name for OpenAI-compatible endpoint: %s -> %s",
        model_name,
        model_block["model_name"],
    )


class ProgressTrackingAgent(DefaultAgent):
    """Simple wrapper around DefaultAgent that provides progress updates."""

    def __init__(
        self,
        *args,
        progress_manager: RunBatchProgressManager,
        instance_id: str = "",
        repeat_action_limit: int = 6,
        echo_action_limit: int = 10,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.progress_manager: RunBatchProgressManager = progress_manager
        self.instance_id = instance_id
        self.repeat_action_limit = max(2, int(repeat_action_limit))
        self.echo_action_limit = max(3, int(echo_action_limit))
        self._last_action_signature: str | None = None
        self._same_action_streak = 0
        self._echo_action_streak = 0

    @staticmethod
    def _normalize_command(command: str) -> str:
        return " ".join((command or "").strip().split())

    @staticmethod
    def _is_echo_like(command: str) -> bool:
        c = command.lower().strip()
        return c.startswith("echo ") or c.startswith("printf ")

    def _maybe_abort_loop(self, actions: list[dict]) -> None:
        commands = [self._normalize_command(a.get("command", "")) for a in actions]
        commands = [c for c in commands if c]
        if not commands:
            self._same_action_streak = 0
            self._echo_action_streak = 0
            self._last_action_signature = None
            return

        signature = " || ".join(commands)
        if signature == self._last_action_signature:
            self._same_action_streak += 1
        else:
            self._same_action_streak = 1
            self._last_action_signature = signature

        if all(self._is_echo_like(c) for c in commands):
            self._echo_action_streak += 1
        else:
            self._echo_action_streak = 0

        if self._same_action_streak < self.repeat_action_limit and self._echo_action_streak < self.echo_action_limit:
            return

        reason = (
            f"LoopDetected: repeated identical actions {self._same_action_streak}x"
            if self._same_action_streak >= self.repeat_action_limit
            else f"LoopDetected: echo-like actions {self._echo_action_streak}x"
        )
        raise InterruptAgentFlow(
            self.model.format_message(
                role="exit",
                content=reason,
                extra={
                    "exit_status": "LoopDetected",
                    "submission": "",
                    "loop_guard": {
                        "same_action_streak": self._same_action_streak,
                        "echo_action_streak": self._echo_action_streak,
                        "repeat_action_limit": self.repeat_action_limit,
                        "echo_action_limit": self.echo_action_limit,
                        "last_action_signature": signature[:500],
                    },
                },
            )
        )

    def step(self) -> dict:
        """Override step to provide progress updates."""
        self.progress_manager.update_instance_status(self.instance_id, f"Step {self.n_calls + 1:3d} (${self.cost:.2f})")
        return super().step()

    def execute_actions(self, message: dict) -> list[dict]:
        actions = message.get("extra", {}).get("actions", [])
        self._maybe_abort_loop(actions)
        return super().execute_actions(message)


def get_swebench_docker_image_name(instance: dict) -> str:
    """Get the image name for a SWEBench instance."""
    image_name = instance.get("image_name", None) or instance.get("docker_image", None)
    if image_name is None:
        # Docker doesn't allow double underscore, so we replace them with a magic token
        iid = instance["instance_id"]
        id_docker_compatible = iid.replace("__", "_1776_")
        image_name = f"docker.io/swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
    return image_name


def get_sb_environment(
    config: dict,
    instance: dict,
    *,
    fail_to_pass: list[str],
    pass_to_pass: list[str],
    validation_command: str,
    feedback_loop: bool = True,
) -> Environment:
    # Use a per-instance environment config copy to avoid cross-instance leakage in parallel runs.
    env_config = dict(config.get("environment", {}))
    env_config["environment_class"] = env_config.get("environment_class", "docker")
    
    # Use instance docker_image if not already set in config
    if "image" not in env_config:
        image_name = get_swebench_docker_image_name(instance)
        if env_config["environment_class"] in ["docker", "swerex_modal"]:
            env_config["image"] = image_name
        elif env_config["environment_class"] in ["singularity", "contree"]:
            env_config["image"] = "docker://" + image_name

    # Inject validation contract only when feedback-loop mode is active.
    if feedback_loop:
        env_config["required_fail_to_pass"] = fail_to_pass
        env_config["required_pass_to_pass"] = pass_to_pass
        env_config["required_validation_command"] = validation_command
        env_config["enforce_non_empty_submission"] = True
        env_config["enforce_patch_sanity"] = True

    env = get_environment(env_config)
    if startup_command := config.get("run", {}).get("env_startup_command"):
        startup_command = Template(startup_command, undefined=StrictUndefined).render(**instance)
        out = env.execute({"command": startup_command})
        if out["returncode"] != 0:
            raise RuntimeError(f"Error executing startup command: {out}")
    return env


def update_preds_file(output_path: Path, instance_id: str, model_name: str, result: str):
    """Update the output JSON file with results from a single instance."""
    with _OUTPUT_FILE_LOCK:
        output_data = {}
        if output_path.exists():
            output_data = json.loads(output_path.read_text())
        output_data[instance_id] = {
            "model_name_or_path": model_name,
            "instance_id": instance_id,
            "model_patch": result,
        }
        output_path.write_text(json.dumps(output_data, indent=2))


def normalize_patch_for_harness(patch: str) -> str:
    """Normalize git patch headers to the a/ b/ form expected by swebench harness."""
    if not patch:
        return patch

    normalized_lines: list[str] = []
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            m = re.match(r"^diff --git\s+(\S+)\s+(\S+)$", line)
            if m:
                left, right = m.group(1), m.group(2)
                if not left.startswith(("a/", "b/")):
                    left = f"a/{left}"
                if not right.startswith(("a/", "b/")):
                    right = f"b/{right}"
                line = f"diff --git {left} {right}"
        elif line.startswith("--- "):
            path = line[4:]
            if path != "/dev/null" and not path.startswith(("a/", "b/")):
                line = f"--- a/{path}"
        elif line.startswith("+++ "):
            path = line[4:]
            if path != "/dev/null" and not path.startswith(("a/", "b/")):
                line = f"+++ b/{path}"
        normalized_lines.append(line)

    return "\n".join(normalized_lines) + ("\n" if patch.endswith("\n") else "")


def sanitize_feedback_submission(submission: str) -> str:
    """Clean common wrapper noise from feedback-loop submissions while keeping raw diff content."""
    if not submission:
        return submission

    cleaned = submission.replace("\r\n", "\n")
    lines = cleaned.splitlines()

    # Remove markdown code fences if present.
    if "```" in cleaned:
        lines = [line for line in lines if not line.strip().startswith("```")]

    patch_markers = ("diff --git ", "--- ", "Index: ")
    start_idx = None
    for idx, line in enumerate(lines):
        if any(line.startswith(marker) for marker in patch_markers):
            start_idx = idx
            break
    if start_idx is not None:
        lines = lines[start_idx:]

    cleaned = "\n".join(lines)
    if submission.endswith("\n") and cleaned:
        cleaned += "\n"
    return cleaned


def remove_from_preds_file(output_path: Path, instance_id: str):
    """Remove an instance from the predictions file."""
    if not output_path.exists():
        return
    with _OUTPUT_FILE_LOCK:
        output_data = json.loads(output_path.read_text())
        if instance_id in output_data:
            del output_data[instance_id]
            output_path.write_text(json.dumps(output_data, indent=2))


def _normalize_required_tests(value: object) -> list[str]:
    """Normalize required test fields that may arrive as JSON-encoded strings."""
    if value is None:
        return []

    parsed = value
    if isinstance(value, str):
        parsed = _parse_key_value_value(value)

    if isinstance(parsed, list):
        return [str(item) for item in parsed if item]
    if isinstance(parsed, tuple):
        return [str(item) for item in parsed if item]
    if isinstance(parsed, set):
        return [str(item) for item in parsed if item]
    return []


def build_validation_command(instance: dict) -> str:
    """Build a focused pytest command from instance-specific target tests."""
    tests = [
        *_normalize_required_tests(instance.get("FAIL_TO_PASS")),
        *_normalize_required_tests(instance.get("PASS_TO_PASS")),
    ]
    unique_tests = list(dict.fromkeys(tests))
    if unique_tests:
        return "pytest --no-header -rA --tb=no -p no:cacheprovider " + " ".join(unique_tests)
    return "pytest --no-header -rA --tb=no -p no:cacheprovider"


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    return [x for x in dict.fromkeys(items) if x]


def _parse_test_nodeid_path(nodeid: str) -> str:
    if not nodeid:
        return ""
    file_part = nodeid.split("::", 1)[0].strip()
    if file_part.endswith(".py"):
        return file_part
    if "/" not in file_part and "." in file_part:
        dotted = file_part.replace(".", "/")
        if not dotted.endswith(".py"):
            dotted += ".py"
        return dotted
    return ""


def _extract_paths_from_problem_statement(problem_statement: str) -> list[str]:
    if not problem_statement:
        return []
    path_like = re.findall(r"([A-Za-z0-9_./-]+\.(?:py|pyi|json|yaml|yml|toml|ini|cfg|md|txt))", problem_statement)
    return _dedupe_preserve_order(path_like)


def _extract_symbols_from_problem_statement(problem_statement: str) -> list[str]:
    if not problem_statement:
        return []
    candidates = re.findall(r"`([^`]+)`", problem_statement)
    symbols: list[str] = []
    for c in candidates:
        token = c.strip()
        if not token or "/" in token or token.endswith(".py"):
            continue
        if re.match(r"^[A-Za-z_][A-Za-z0-9_\.]*$", token):
            symbols.append(token)
    return _dedupe_preserve_order(symbols)


def _render_limited_bullets(items: list[str], *, max_items: int = _MAX_CONTEXT_ITEMS) -> str:
    if not items:
        return "- (none)"
    shown = items[:max_items]
    lines = [f"- {item}" for item in shown]
    remaining = len(items) - len(shown)
    if remaining > 0:
        lines.append(f"- ... and {remaining} more")
    return "\n".join(lines)


def build_feedback_enriched_task(instance: dict, validation_command: str) -> str:
    """Build a richer feedback-loop task prompt with requirement and dependency context.

    This must only be used when feedback-loop mode is enabled.
    """
    base_task = instance.get("problem_statement", "")
    fail_to_pass = _dedupe_preserve_order(_normalize_required_tests(instance.get("FAIL_TO_PASS")))
    pass_to_pass = _dedupe_preserve_order(_normalize_required_tests(instance.get("PASS_TO_PASS")))
    test_paths = _dedupe_preserve_order([_parse_test_nodeid_path(t) for t in [*fail_to_pass, *pass_to_pass]])
    statement_paths = _extract_paths_from_problem_statement(base_task)
    statement_symbols = _extract_symbols_from_problem_statement(base_task)
    related_paths = _dedupe_preserve_order([*statement_paths, *test_paths])

    dependency_roots = []
    for p in related_paths:
        root = p.split("/", 1)[0] if "/" in p else p
        if root:
            dependency_roots.append(root)
    dependency_roots = _dedupe_preserve_order(dependency_roots)

    hints_text = (instance.get("hints_text") or "").strip()
    if hints_text and len(hints_text) > _MAX_HINTS_CHARS:
        hints_text = hints_text[:_MAX_HINTS_CHARS].rstrip() + "\n... (truncated)"

    enriched_sections = [
        "\n\n[FEEDBACK-LOOP PREPROCESS CONTEXT]",
        "Use this analyzed context to craft a minimal, valid patch that is accepted by the harness.",
        "",
        "Requirement refinement (must satisfy all):",
        f"- Fix failing target tests (FAIL_TO_PASS): {len(fail_to_pass)}",
        f"- Keep passing tests stable (PASS_TO_PASS): {len(pass_to_pass)}",
        "- Output only a valid unified git diff patch (no markdown wrappers).",
        f"- Validation command: {validation_command}",
        "",
        "Target tests to fix (priority):",
        _render_limited_bullets(fail_to_pass),
        "",
        "Regression guard tests (must remain passing):",
        _render_limited_bullets(pass_to_pass),
        "",
        "Program-analysis: related file paths (from problem + test nodeids):",
        _render_limited_bullets(related_paths),
        "",
        "Program-analysis: likely dependency roots / code areas:",
        _render_limited_bullets(dependency_roots),
        "",
        "Program-analysis: referenced symbols in requirement text:",
        _render_limited_bullets(statement_symbols),
    ]

    if hints_text:
        enriched_sections.extend(["", "Additional hints_text context:", hints_text])

    return base_task + "\n" + "\n".join(enriched_sections)


def process_instance(
    instance: dict,
    output_dir: Path,
    config: dict,
    progress_manager: RunBatchProgressManager,
    feedback_loop: bool = True,
    repeat_action_limit: int = 6,
    echo_action_limit: int = 10,
) -> None:
    """Process a single SWEBench instance."""
    instance_id = instance["instance_id"]
    instance_dir = output_dir / instance_id
    # avoid inconsistent state if something here fails and there's leftover previous files
    remove_from_preds_file(output_dir / "preds.json", instance_id)
    (instance_dir / f"{instance_id}.traj.json").unlink(missing_ok=True)
    model = get_model(config=config.get("model", {}))
    task = instance["problem_statement"]

    progress_manager.on_instance_start(instance_id)
    progress_manager.update_instance_status(instance_id, "Pulling/starting environment")

    agent = None
    exit_status = None
    result = None
    extra_info = {}

    try:
        fail_to_pass = _normalize_required_tests(instance.get("FAIL_TO_PASS"))
        pass_to_pass = _normalize_required_tests(instance.get("PASS_TO_PASS"))
        validation_command = build_validation_command(instance)

        # Preprocess and enrich context only when feedback loop is enabled.
        if feedback_loop:
            task = build_feedback_enriched_task(instance, validation_command)

        env = get_sb_environment(
            config,
            instance,
            fail_to_pass=fail_to_pass,
            pass_to_pass=pass_to_pass,
            validation_command=validation_command,
            feedback_loop=feedback_loop,
        )
        agent = ProgressTrackingAgent(
            model,
            env,
            progress_manager=progress_manager,
            instance_id=instance_id,
            repeat_action_limit=repeat_action_limit,
            echo_action_limit=echo_action_limit,
            **config.get("agent", {}),
        )
        # Pass test information to agent for validation
        info = agent.run(
            task,
            test_cmd=instance.get("test_cmd", ""),
            validation_command=validation_command,
            fail_to_pass=fail_to_pass,
            pass_to_pass=pass_to_pass,
        )
        exit_status = info.get("exit_status")
        submission = info.get("submission") or ""
        if feedback_loop:
            submission = sanitize_feedback_submission(submission)
        result = normalize_patch_for_harness(submission)
    except Exception as e:
        logger.error(f"Error processing instance {instance_id}: {e}", exc_info=True)
        exit_status, result = type(e).__name__, ""
        extra_info = {"traceback": traceback.format_exc(), "exception_str": str(e)}
    finally:
        if agent is not None:
            traj_path = instance_dir / f"{instance_id}.traj.json"
            agent.save(
                traj_path,
                {
                    "info": {
                        "exit_status": exit_status,
                        "submission": result,
                        **extra_info,
                    },
                    "instance_id": instance_id,
                },
            )
            logger.info(f"Saved trajectory to '{traj_path}'")
        update_preds_file(output_dir / "preds.json", instance_id, model.config.model_name, result)
        progress_manager.on_instance_end(instance_id, exit_status)


def filter_instances(
    instances: list[dict], *, filter_spec: str, slice_spec: str = "", shuffle: bool = False
) -> list[dict]:
    """Filter and slice a list of SWEBench instances."""
    if shuffle:
        instances = sorted(instances.copy(), key=lambda x: x["instance_id"])
        random.seed(42)
        random.shuffle(instances)
    before_filter = len(instances)
    instances = [instance for instance in instances if re.match(filter_spec, instance["instance_id"])]
    if (after_filter := len(instances)) != before_filter:
        logger.info(f"Instance filter: {before_filter} -> {after_filter} instances")
    if slice_spec:
        values = [int(x) if x else None for x in slice_spec.split(":")]
        instances = instances[slice(*values)]
        if (after_slice := len(instances)) != before_filter:
            logger.info(f"Instance slice: {before_filter} -> {after_slice} instances")
    return instances


# fmt: off
@app.command(help=_HELP_TEXT)
def main(
    subset: str = typer.Option("lite", "--subset", help="SWEBench subset to use or path to a dataset", rich_help_panel="Data selection"),
    split: str = typer.Option("dev", "--split", help="Dataset split", rich_help_panel="Data selection"),
    slice_spec: str = typer.Option("", "--slice", help="Slice specification (e.g., '0:5' for first 5 instances)", rich_help_panel="Data selection"),
    filter_spec: str = typer.Option("", "--filter", help="Filter instance IDs by regex", rich_help_panel="Data selection"),
    shuffle: bool = typer.Option(False, "--shuffle", help="Shuffle instances", rich_help_panel="Data selection"),
    output: str = typer.Option("", "-o", "--output", help="Output directory", rich_help_panel="Basic"),
    workers: int = typer.Option(1, "-w", "--workers", help="Number of worker threads for parallel processing", rich_help_panel="Basic"),
    model: str | None = typer.Option(None, "-m", "--model", help="Model to use", rich_help_panel="Basic"),
    model_class: str | None = typer.Option(None, "--model-class", help="Model class to use (e.g., 'anthropic' or 'minisweagent.models.anthropic.AnthropicModel')", rich_help_panel="Advanced"),
    redo_existing: bool = typer.Option(False, "--redo-existing", help="Redo existing instances", rich_help_panel="Data selection"),
    config_spec: list[str] = typer.Option([str(DEFAULT_CONFIG_FILE)], "-c", "--config", help=_CONFIG_SPEC_HELP_TEXT, rich_help_panel="Basic"),
    environment_class: str | None = typer.Option(None, "--environment-class", help="Environment type to use. Recommended are docker or singularity", rich_help_panel="Advanced"),
    feedback_loop: bool = typer.Option(True, "--feedback-loop/--no-feedback-loop", help="Enable hard validation gate (feedback-in-loop). Use --no-feedback-loop for baseline mini-swe-agent behavior.", rich_help_panel="Basic"),
    repeat_action_limit: int = typer.Option(12, "--repeat-action-limit", help="Abort an instance when the exact same action set repeats this many consecutive steps.", rich_help_panel="Advanced"),
    echo_action_limit: int = typer.Option(18, "--echo-action-limit", help="Abort an instance when echo/printf-only actions repeat this many consecutive steps.", rich_help_panel="Advanced"),
    model_timeout_seconds: int = typer.Option(180, "--model-timeout-seconds", help="Per-request model timeout in seconds (applied if model.model_kwargs.timeout is not explicitly set).", rich_help_panel="Advanced"),
    model_retry_attempts: int = typer.Option(2, "--model-retry-attempts", help="Max model retry attempts for transient API errors. Lower values fail fast and continue to next instance.", rich_help_panel="Advanced"),
    sync_model_name_to_config: bool = typer.Option(False, "--sync-model-name-to-config/--no-sync-model-name-to-config", help="Persist effective model.model_name back into the first YAML config file when a CLI model override is used.", rich_help_panel="Advanced"),
) -> None:
    # fmt: on
    from datasets import load_dataset, load_from_disk
    from pathlib import Path

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {output_path}")
    add_file_handler(output_path / "minisweagent.log")

    dataset_path = DATASET_MAPPING.get(subset, subset)
    logger.info(f"Loading dataset {dataset_path}, split {split}...")
    
    # Handle both remote (HuggingFace) and local dataset paths
    if Path(dataset_path).exists():
        dataset = load_from_disk(dataset_path)
        instances = list(dataset[split])
    else:
        instances = list(load_dataset(dataset_path, split=split))

    instances = filter_instances(instances, filter_spec=filter_spec, slice_spec=slice_spec, shuffle=shuffle)
    if not redo_existing and (output_path / "preds.json").exists():
        existing_instances = list(json.loads((output_path / "preds.json").read_text()).keys())
        logger.info(f"Skipping {len(existing_instances)} existing instances")
        instances = [instance for instance in instances if instance["instance_id"] not in existing_instances]
    logger.info(f"Running on {len(instances)} instances...")

    logger.info(f"Building agent config from specs: {config_spec}")
    configs = [get_config_from_spec(spec) for spec in config_spec]
    configs.append({
        "environment": {"environment_class": environment_class or UNSET},
        "model": {"model_name": model or UNSET, "model_class": model_class or UNSET},
    })
    config = recursive_merge(*configs)
    normalize_model_provider_prefix(config)
    maybe_sync_model_name_to_config(
        config_spec,
        model_option=model,
        merged_config=config,
        sync_enabled=sync_model_name_to_config,
    )

    # Fail fast on stuck model generations and continue with remaining instances.
    os.environ["MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT"] = str(max(1, model_retry_attempts))
    model_config = config.setdefault("model", {})
    model_kwargs = model_config.setdefault("model_kwargs", {})
    if model_timeout_seconds > 0 and "timeout" not in model_kwargs:
        model_kwargs["timeout"] = model_timeout_seconds
    logger.info(
        "Model safeguards enabled: timeout=%ss, retry_attempts=%s",
        model_kwargs.get("timeout", "unset"),
        os.environ["MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT"],
    )

    progress_manager = RunBatchProgressManager(len(instances), output_path / f"exit_statuses_{time.time()}.yaml")

    def process_futures(futures: dict[concurrent.futures.Future, str]):
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except concurrent.futures.CancelledError:
                pass
            except Exception as e:
                instance_id = futures[future]
                logger.error(f"Error in future for instance {instance_id}: {e}", exc_info=True)
                progress_manager.on_uncaught_exception(instance_id, e)

    with Live(progress_manager.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    process_instance,
                    instance,
                    output_path,
                    config,
                    progress_manager,
                    feedback_loop,
                    repeat_action_limit,
                    echo_action_limit,
                ): instance[
                    "instance_id"
                ]
                for instance in instances
            }
            try:
                process_futures(futures)
            except KeyboardInterrupt:
                logger.info("Cancelling all pending jobs. Press ^C again to exit immediately.")
                for future in futures:
                    if not future.running() and not future.done():
                        future.cancel()
                process_futures(futures)


if __name__ == "__main__":
    app()
