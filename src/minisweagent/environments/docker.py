import logging
import os
import platform
import shlex
import subprocess
import uuid
from typing import Any

from pydantic import BaseModel

from minisweagent.exceptions import Submitted
from minisweagent.utils.serialize import recursive_merge


class DockerEnvironmentConfig(BaseModel):
    image: str
    cwd: str = "/"
    """Working directory in which to execute commands."""
    env: dict[str, str] = {}
    """Environment variables to set in the container."""
    forward_env: list[str] = []
    """Environment variables to forward to the container.
    Variables are only forwarded if they are set in the host environment.
    In case of conflict with `env`, the `env` variables take precedence.
    """
    timeout: int = 30
    """Timeout for executing commands in the container."""
    executable: str = os.getenv("MSWEA_DOCKER_EXECUTABLE", "docker")
    """Path to the docker/container executable."""
    run_args: list[str] = ["--rm"]
    """Additional arguments to pass to the docker/container executable.
    Default is ["--rm"], which removes the container after it exits.
    """
    container_timeout: str = "2h"
    """Max duration to keep container running. Uses the same format as the sleep command."""
    pull_timeout: int = 120
    """Timeout in seconds for pulling images."""
    interpreter: list[str] = ["bash", "-lc"]
    """Interpreter to use to execute commands. Default is ["bash", "-lc"].
    The actual command will be appended as argument to this. Override this to e.g., modify shell flags
    (e.g., to remove the `-l` flag to disable login shell) or to use python instead of bash to interpret commands.
    """
    required_fail_to_pass: list[str] = []
    """Tests that must pass before submission is allowed."""
    required_pass_to_pass: list[str] = []
    """Tests that must not regress before submission is allowed."""
    required_validation_command: str = ""
    """Preferred full validation command that should be run successfully before submission."""
    enforce_non_empty_submission: bool = False
    """When True, block COMPLETE_TASK submission if no patch content is provided."""
    enforce_patch_sanity: bool = False
    """When True, require submission content to look like a unified diff patch."""


class DockerEnvironment:
    def __init__(
        self,
        *,
        config_class: type = DockerEnvironmentConfig,
        logger: logging.Logger | None = None,
        **kwargs,
    ):
        """This class executes bash commands in a Docker container using direct docker commands.
        See `DockerEnvironmentConfig` for keyword arguments.
        """
        self.logger = logger or logging.getLogger("minisweagent.environment")
        self.container_id: str | None = None
        self.config = config_class(**kwargs)
        # Submission is allowed only after at least one successful validation command.
        self._validation_passed = False
        self._required_fail_to_pass = list(dict.fromkeys(self.config.required_fail_to_pass))
        self._required_pass_to_pass = list(dict.fromkeys(self.config.required_pass_to_pass))
        self._required_tests = list(dict.fromkeys(self._required_fail_to_pass + self._required_pass_to_pass))
        self._required_validation_command = " ".join(self.config.required_validation_command.split())
        self._enforce_non_empty_submission = bool(self.config.enforce_non_empty_submission)
        self._enforce_patch_sanity = bool(self.config.enforce_patch_sanity)
        self._start_container()

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return recursive_merge(self.config.model_dump(), platform.uname()._asdict(), kwargs)

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(mode="json"),
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }

    def _start_container(self):
        """Start the Docker container and return the container ID."""
        container_name = f"minisweagent-{uuid.uuid4().hex[:8]}"
        cmd = [
            self.config.executable,
            "run",
            "-d",
            "--name",
            container_name,
            "-w",
            self.config.cwd,
            *self.config.run_args,
            self.config.image,
            "sleep",
            self.config.container_timeout,
        ]
        self.logger.debug(f"Starting container with command: {shlex.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.config.pull_timeout,  # docker pull might take a while
            check=True,
        )
        self.logger.info(f"Started container {container_name} with ID {result.stdout.strip()}")
        self.container_id = result.stdout.strip()

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Execute a command in the Docker container and return the result as a dict."""
        command = action.get("command", "")
        cwd = cwd or self.config.cwd
        assert self.container_id, "Container not started"

        cmd = [self.config.executable, "exec", "-w", cwd]
        for key in self.config.forward_env:
            if (value := os.getenv(key)) is not None:
                cmd.extend(["-e", f"{key}={value}"])
        for key, value in self.config.env.items():
            cmd.extend(["-e", f"{key}={value}"])
        cmd.extend([self.container_id, *self.config.interpreter, command])

        try:
            result = subprocess.run(
                cmd,
                text=True,
                timeout=timeout or self.config.timeout,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            output = {"output": result.stdout, "returncode": result.returncode, "exception_info": ""}
        except Exception as e:
            raw_output = getattr(e, "output", None)
            raw_output = (
                raw_output.decode("utf-8", errors="replace") if isinstance(raw_output, bytes) else (raw_output or "")
            )
            output = {
                "output": raw_output,
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {e}",
                "extra": {"exception_type": type(e).__name__, "exception": str(e)},
            }
        if output["returncode"] == 0 and self._is_validation_command(command) and self._covers_required_tests(command):
            self._validation_passed = True
        self._check_finished(output)
        return output

    def _covers_required_tests(self, command: str) -> bool:
        """Return True only if command covers required tests for this instance."""
        if not self._required_tests:
            return True
        normalized = " ".join(command.strip().split())
        if self._required_validation_command and normalized == self._required_validation_command:
            return True
        return all(test_id in normalized for test_id in self._required_tests)

    def _is_validation_command(self, command: str) -> bool:
        """Heuristic check for common test/validation commands across ecosystems."""
        normalized = command.strip().lower()
        if not normalized:
            return False
        validation_patterns = (
            "pytest",
            "python -m pytest",
            "npm test",
            "pnpm test",
            "yarn test",
            "mvn test",
            "./mvnw test",
            "gradle test",
            "./gradlew test",
            "go test",
            "cargo test",
        )
        return any(pattern in normalized for pattern in validation_patterns)

    @staticmethod
    def _looks_like_unified_diff(submission: str) -> bool:
        """Best-effort validation that a submission contains a unified diff patch."""
        if not submission:
            return False
        normalized = submission.strip()
        has_diff_header = "diff --git " in normalized
        has_file_markers = "\n--- " in f"\n{normalized}" and "\n+++ " in f"\n{normalized}"
        has_hunk = "\n@@ " in f"\n{normalized}" or "\n@@" in f"\n{normalized}"
        return (has_diff_header and has_file_markers) or (has_file_markers and has_hunk)

    @staticmethod
    def _starts_with_patch_marker(submission: str) -> bool:
        """Require raw patch content to start immediately, without narrative prefix."""
        lines = submission.lstrip().splitlines()
        if not lines:
            return False
        first = lines[0]
        return first.startswith("diff --git ") or first.startswith("--- ") or first.startswith("Index: ")

    def _check_finished(self, output: dict):
        """Raises Submitted if the output indicates task completion."""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" and output["returncode"] == 0:
            if not self._validation_passed:
                output["returncode"] = 1
                gate_msg = (
                    "\n[VALIDATION_GATE] Submission blocked: required tests were not successfully validated.\n"
                )
                if self._required_tests:
                    tests_summary = "\n".join(f"- {test_id}" for test_id in self._required_tests)
                    gate_msg += (
                        "Required tests for this instance:\n"
                        f"{tests_summary}\n"
                        "Run a successful test command that includes these tests, then submit again.\n"
                    )
                    if self._required_validation_command:
                        gate_msg += f"Recommended command: {self._required_validation_command}\n"
                else:
                    gate_msg += "Run relevant tests (e.g. pytest / npm test / mvn test), fix failures, then submit again.\n"
                output["output"] = output.get("output", "") + gate_msg
                return
            submission = "".join(lines[1:])
            if self._enforce_non_empty_submission and not submission.strip():
                output["returncode"] = 1
                output["output"] = (
                    output.get("output", "")
                    + "\n[VALIDATION_GATE] Submission blocked: empty patch payload.\n"
                    + "Submit with COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT followed by unified diff content.\n"
                )
                return
            if self._enforce_patch_sanity and submission.strip() and not self._looks_like_unified_diff(submission):
                output["returncode"] = 1
                output["output"] = (
                    output.get("output", "")
                    + "\n[VALIDATION_GATE] Submission blocked: payload does not look like a valid unified diff patch.\n"
                    + "Include raw git diff output only (no markdown fences or narrative text).\n"
                )
                return
            if self._enforce_patch_sanity and submission.strip() and "```" in submission:
                output["returncode"] = 1
                output["output"] = (
                    output.get("output", "")
                    + "\n[VALIDATION_GATE] Submission blocked: markdown code fences are not allowed.\n"
                    + "Submit raw diff text only.\n"
                )
                return
            if self._enforce_patch_sanity and submission.strip() and not self._starts_with_patch_marker(submission):
                output["returncode"] = 1
                output["output"] = (
                    output.get("output", "")
                    + "\n[VALIDATION_GATE] Submission blocked: patch must start at the first non-whitespace line.\n"
                    + "Remove summary/explanatory text and submit raw diff only.\n"
                )
                return
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )

    def cleanup(self):
        """Stop and remove the Docker container."""
        if getattr(self, "container_id", None) is not None:  # if init fails early, container_id might not be set
            cmd = f"(timeout 60 {self.config.executable} stop {self.container_id} || {self.config.executable} rm -f {self.container_id}) >/dev/null 2>&1 &"
            subprocess.Popen(cmd, shell=True)

    def __del__(self):
        """Cleanup container when object is destroyed."""
        self.cleanup()
