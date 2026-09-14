"""CLI integration tests — init / list-projects / show-project / install-skills."""
from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest
from click.testing import CliRunner

from zenith_harness.assets import parse_frontmatter
from zenith_harness.cli import cli


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def env(harness_home: Path, workspace: Path, monkeypatch) -> dict[str, str]:
    monkeypatch.setenv("ZENITH_HOME", str(harness_home))
    monkeypatch.chdir(workspace)
    return {"ZENITH_HOME": str(harness_home)}


def _expected_mcp_server_args() -> list[str]:
    zenith_root = Path(__file__).resolve().parents[1]
    return [
        "run",
        "--project",
        str(zenith_root),
        "zenith-server",
        "--mode",
        "orchestrator",
    ]


BUNDLED_DIR = Path(__file__).resolve().parents[1] / "src" / "zenith_harness" / "bundled"
OPENCODE_AGENT_NAMES = (
    "contract-review",
    "feature-reviewer",
    "flow-validator",
    "investigator",
)


def _bundled_skill_names() -> set[str]:
    return {
        path.name
        for path in (BUNDLED_DIR / "skills").iterdir()
        if path.is_dir() and (path / "SKILL.md").exists()
    }


def _installed_skill_names(skill_root: Path) -> list[str]:
    return sorted(
        path.name
        for path in skill_root.iterdir()
        if path.is_dir() and (path / "SKILL.md").exists()
    )


def _read_opencode_config(workspace: Path) -> dict[str, object]:
    return json.loads((workspace / ".opencode" / "opencode.json").read_text(encoding="utf-8"))


class TestInit:
    def test_stages_host_agent_surface_only(
        self, runner: CliRunner, workspace: Path, env: dict[str, str]
    ) -> None:
        """`zenith init` writes MCP config + provider agents + orchestrator prompt
        but does NOT create the project bucket or workspace shims — those are
        created by `start_project` at the first MCP call."""
        result = runner.invoke(
            cli, ["init", "--workspace-dir", str(workspace), "--agent", "claude"]
        )
        assert result.exit_code == 0, result.output
        # Workspace stays clean of .zenith/ — bucket lives under ZENITH_HOME.
        assert not (workspace / ".zenith").exists()
        # No symlink shims either — start_project handles them.
        assert not (workspace / "AGENTS.md").exists()
        # MCP config + .claude/agents/ are written.
        assert (workspace / ".mcp.json").exists()
        mcp = json.loads((workspace / ".mcp.json").read_text())
        assert "zenith" in mcp["mcpServers"]
        server = mcp["mcpServers"]["zenith"]
        assert server["command"] == "uv"
        assert server["args"] == _expected_mcp_server_args()

    def test_init_does_not_touch_gitignore(
        self, runner: CliRunner, workspace: Path, env: dict[str, str]
    ) -> None:
        gitignore = workspace / ".gitignore"
        gitignore.write_text("node_modules/\n")
        original = gitignore.read_text()
        r = runner.invoke(
            cli, ["init", "--workspace-dir", str(workspace), "--agent", "claude"]
        )
        assert r.exit_code == 0, r.output
        assert gitignore.read_text() == original

    def test_idempotent(
        self, runner: CliRunner, workspace: Path, env: dict[str, str]
    ) -> None:
        for _ in range(2):
            r = runner.invoke(
                cli, ["init", "--workspace-dir", str(workspace), "--agent", "claude"]
            )
            assert r.exit_code == 0, r.output
        # .mcp.json preserved across reruns.
        assert (workspace / ".mcp.json").exists()

    def test_codex_writes_codex_config(
        self, runner: CliRunner, workspace: Path, env: dict[str, str]
    ) -> None:
        r = runner.invoke(cli, ["init", "--workspace-dir", str(workspace), "--agent", "codex"])
        assert r.exit_code == 0, r.output
        config_path = workspace / ".codex" / "config.toml"
        assert config_path.exists()
        config = tomllib.loads(config_path.read_text(encoding="utf-8"))
        server = config["mcp_servers"]["zenith"]
        assert server["command"] == "uv"
        assert server["args"] == _expected_mcp_server_args()
        assert f"Initialized v5 project workspace at {workspace}" in r.output
        assert "Start your agent from the initialized project workspace" in r.output
        assert (
            "First read .codex/orchestrator_prompt.md and treat it as your primary role, "
            "then use Zenith to run this mission." in r.output
        )

    def test_claude_init_writes_reasoning_effort_env(
        self,
        runner: CliRunner,
        workspace: Path,
        env: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("ZENITH_WORKER_REASONING_EFFORT", "high")
        monkeypatch.setenv("ZENITH_VALIDATOR_REASONING_EFFORT", "medium")
        monkeypatch.setenv("ZENITH_TERMINAL_REVIEWER_REASONING_EFFORT", "low")

        r = runner.invoke(cli, ["init", "--workspace-dir", str(workspace), "--agent", "claude"])
        assert r.exit_code == 0, r.output

        mcp = json.loads((workspace / ".mcp.json").read_text(encoding="utf-8"))
        server_env = mcp["mcpServers"]["zenith"]["env"]
        assert server_env["ZENITH_WORKER_REASONING_EFFORT"] == "high"
        assert server_env["ZENITH_VALIDATOR_REASONING_EFFORT"] == "medium"
        assert server_env["ZENITH_TERMINAL_REVIEWER_REASONING_EFFORT"] == "low"

    def test_codex_init_writes_reasoning_effort_env(
        self,
        runner: CliRunner,
        workspace: Path,
        env: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("ZENITH_WORKER_REASONING_EFFORT", "high")
        monkeypatch.setenv("ZENITH_VALIDATOR_REASONING_EFFORT", "medium")
        monkeypatch.setenv("ZENITH_TERMINAL_REVIEWER_REASONING_EFFORT", "low")

        r = runner.invoke(cli, ["init", "--workspace-dir", str(workspace), "--agent", "codex"])
        assert r.exit_code == 0, r.output

        config = tomllib.loads(
            (workspace / ".codex" / "config.toml").read_text(encoding="utf-8")
        )
        server_env = config["mcp_servers"]["zenith"]["env"]
        assert server_env["ZENITH_WORKER_REASONING_EFFORT"] == "high"
        assert server_env["ZENITH_VALIDATOR_REASONING_EFFORT"] == "medium"
        assert server_env["ZENITH_TERMINAL_REVIEWER_REASONING_EFFORT"] == "low"

    def test_codex_init_escapes_quoted_acp_commands(
        self, runner: CliRunner, workspace: Path, env: dict[str, str]
    ) -> None:
        """Quoted ACP commands must survive into config.toml as valid TOML.

        `-c key="value"` is the supported splice shape for codex config, so
        a role's command can carry double quotes. Interpolating them raw
        terminates the TOML string early and corrupts the managed block.
        The two commands differ so `ProviderSelection.env()` emits both —
        it suppresses a role whose resolved command matches the previous
        role's.
        """
        worker_cmd = 'codex-acp -c model="gpt-5.6-luna"'
        validator_cmd = 'codex-acp -c model="gpt-5.6-terra"'

        r = runner.invoke(
            cli,
            [
                "init",
                "--workspace-dir",
                str(workspace),
                "--agent",
                "codex",
                "--worker-acp-command",
                worker_cmd,
                "--validator-acp-command",
                validator_cmd,
            ],
        )
        assert r.exit_code == 0, r.output

        # Parsing at all is the regression guard — this raises before the fix.
        config = tomllib.loads(
            (workspace / ".codex" / "config.toml").read_text(encoding="utf-8")
        )
        server_env = config["mcp_servers"]["zenith"]["env"]
        assert server_env["ZENITH_WORKER_ACP_COMMAND"] == worker_cmd
        assert server_env["ZENITH_VALIDATOR_ACP_COMMAND"] == validator_cmd

    def test_init_reasoning_effort_flags_override_env(
        self,
        runner: CliRunner,
        workspace: Path,
        env: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("ZENITH_WORKER_REASONING_EFFORT", "xhigh")

        r = runner.invoke(
            cli,
            [
                "init",
                "--workspace-dir",
                str(workspace),
                "--agent",
                "claude",
                "--worker-reasoning-effort",
                "max",
                "--validator-reasoning-effort",
                "medium",
            ],
        )
        assert r.exit_code == 0, r.output

        mcp = json.loads((workspace / ".mcp.json").read_text(encoding="utf-8"))
        server_env = mcp["mcpServers"]["zenith"]["env"]
        # Flag beats the inherited shell env.
        assert server_env["ZENITH_WORKER_REASONING_EFFORT"] == "max"
        assert server_env["ZENITH_VALIDATOR_REASONING_EFFORT"] == "medium"
        assert "ZENITH_TERMINAL_REVIEWER_REASONING_EFFORT" not in server_env

    def test_init_invalid_inherited_effort_env_fails_despite_flag(
        self,
        runner: CliRunner,
        workspace: Path,
        env: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Flags override valid inherited settings; a broken env var is still a
        # hard error — the same validation would raise at server launch, so
        # masking it at init would only defer the failure.
        monkeypatch.setenv("ZENITH_WORKER_REASONING_EFFORT", "turbo")

        r = runner.invoke(
            cli,
            [
                "init",
                "--workspace-dir",
                str(workspace),
                "--agent",
                "claude",
                "--worker-reasoning-effort",
                "max",
            ],
        )
        assert r.exit_code != 0
        assert isinstance(r.exception, ValueError)
        assert "ZENITH_WORKER_REASONING_EFFORT" in str(r.exception)

    def test_claude_init_writes_runtime_validator_env_names(
        self, runner: CliRunner, workspace: Path, env: dict[str, str]
    ) -> None:
        r = runner.invoke(
            cli,
            [
                "init",
                "--workspace-dir",
                str(workspace),
                "--agent",
                "claude",
                "--validator-provider",
                "codex",
                "--validator-acp-command",
                "custom-validator-acp",
            ],
        )
        assert r.exit_code == 0, r.output

        mcp = json.loads((workspace / ".mcp.json").read_text())
        mcp_env = mcp["mcpServers"]["zenith"]["env"]
        assert mcp_env["ZENITH_VALIDATOR_PROVIDER"] == "codex"
        assert mcp_env["ZENITH_VALIDATOR_ACP_COMMAND"] == "custom-validator-acp"
        assert "ZENITH_VALIDATION_WORKER_PROVIDER" not in mcp_env
        assert "ZENITH_VALIDATION_WORKER_ACP_COMMAND" not in mcp_env

    def test_claude_init_forwards_only_allowed_model_env(
        self,
        runner: CliRunner,
        workspace: Path,
        env: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.z.ai/api/anthropic")
        monkeypatch.setenv("ANTHROPIC_MODEL", "glm-5.2[1m]")
        monkeypatch.setenv("ZAI_API_KEY", "zai-test-key")
        monkeypatch.setenv("DATABASE_URL", "postgres://should-not-forward")

        r = runner.invoke(cli, ["init", "--workspace-dir", str(workspace), "--agent", "claude"])
        assert r.exit_code == 0, r.output

        mcp = json.loads((workspace / ".mcp.json").read_text())
        mcp_env = mcp["mcpServers"]["zenith"]["env"]
        assert mcp_env["ANTHROPIC_BASE_URL"] == "https://api.z.ai/api/anthropic"
        assert mcp_env["ANTHROPIC_MODEL"] == "glm-5.2[1m]"
        assert mcp_env["ZAI_API_KEY"] == "zai-test-key"
        assert "DATABASE_URL" not in mcp_env

    def test_claude_init_writes_terminal_reviewer_env_names(
        self, runner: CliRunner, workspace: Path, env: dict[str, str]
    ) -> None:
        r = runner.invoke(
            cli,
            [
                "init",
                "--workspace-dir",
                str(workspace),
                "--agent",
                "claude",
                "--terminal-reviewer-provider",
                "codex",
                "--terminal-reviewer-acp-command",
                "custom-tr-acp",
            ],
        )
        assert r.exit_code == 0, r.output

        mcp = json.loads((workspace / ".mcp.json").read_text())
        mcp_env = mcp["mcpServers"]["zenith"]["env"]
        assert mcp_env["ZENITH_TERMINAL_REVIEWER_PROVIDER"] == "codex"
        assert mcp_env["ZENITH_TERMINAL_REVIEWER_ACP_COMMAND"] == "custom-tr-acp"

    def test_claude_init_omits_terminal_reviewer_env_when_unset(
        self, runner: CliRunner, workspace: Path, env: dict[str, str]
    ) -> None:
        r = runner.invoke(
            cli, ["init", "--workspace-dir", str(workspace), "--agent", "claude"]
        )
        assert r.exit_code == 0, r.output

        mcp = json.loads((workspace / ".mcp.json").read_text())
        mcp_env = mcp["mcpServers"]["zenith"]["env"]
        assert "ZENITH_TERMINAL_REVIEWER_PROVIDER" not in mcp_env
        assert "ZENITH_TERMINAL_REVIEWER_ACP_COMMAND" not in mcp_env

    def test_three_distinct_providers_all_env_written(
        self, runner: CliRunner, workspace: Path, env: dict[str, str]
    ) -> None:
        r = runner.invoke(
            cli,
            [
                "init",
                "--workspace-dir",
                str(workspace),
                "--agent",
                "claude",
                "--validator-provider",
                "codex",
                "--terminal-reviewer-provider",
                "hermes",
            ],
        )
        assert r.exit_code == 0, r.output

        mcp = json.loads((workspace / ".mcp.json").read_text())
        mcp_env = mcp["mcpServers"]["zenith"]["env"]
        assert mcp_env["ZENITH_ORCHESTRATOR_PROVIDER"] == "claude"
        assert mcp_env["ZENITH_WORKER_PROVIDER"] == "claude"
        assert mcp_env["ZENITH_VALIDATOR_PROVIDER"] == "codex"
        assert mcp_env["ZENITH_TERMINAL_REVIEWER_PROVIDER"] == "hermes"

    def test_opencode_agent_selector_writes_native_host_config(
        self,
        runner: CliRunner,
        workspace: Path,
        env: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "should-not-persist-for-opencode")

        r = runner.invoke(
            cli, ["init", "--workspace-dir", str(workspace), "--agent", "opencode"]
        )
        assert r.exit_code == 0, r.output

        config_path = workspace / ".opencode" / "opencode.json"
        assert config_path.exists()
        config = _read_opencode_config(workspace)
        assert config["$schema"] == "https://opencode.ai/config.json"
        server = config["mcp"]["zenith"]
        assert server["type"] == "local"
        assert set(server) == {
            "type",
            "command",
            "enabled",
            "timeout",
            "environment",
        }
        assert server["command"] == ["uv", *_expected_mcp_server_args()]
        assert server["enabled"] is True
        assert server["timeout"] == 1000000

        server_env = server["environment"]
        assert server_env["ZENITH_ORCHESTRATOR_PROVIDER"] == "opencode"
        assert server_env["ZENITH_WORKER_PROVIDER"] == "opencode"
        assert server_env["ZENITH_WORKER_ACP_COMMAND"] == "opencode acp"
        assert "ANTHROPIC_API_KEY" not in server_env
        assert "model" not in config
        assert "provider" not in config
        assert "permission" not in config
        assert "plugin" not in config

        assert f"Initialized v5 project workspace at {workspace}" in r.output
        assert "orchestrator=opencode, worker=opencode, validator=opencode." in r.output
        assert "First read .opencode/orchestrator_prompt.md" in r.output

    def test_opencode_init_stdout_includes_default_role_diagnostics(
        self,
        runner: CliRunner,
        workspace: Path,
        env: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("OPENCODE_MODEL", "openai/ambient-expensive")
        monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")

        r = runner.invoke(
            cli, ["init", "--workspace-dir", str(workspace), "--agent", "opencode"]
        )
        assert r.exit_code == 0, r.output

        config_path = workspace / ".opencode" / "opencode.json"
        assert "OpenCode diagnostics:" in r.output
        assert f"host: opencode ({config_path})" in r.output
        assert (
            "roles: orchestrator=opencode, worker=opencode, "
            "validator=opencode, terminal reviewer=opencode"
        ) in r.output
        assert (
            "ACP commands: worker=opencode acp, validator=opencode acp, "
            "terminal reviewer=opencode acp"
        ) in r.output
        assert (
            "model pins: worker=absent, validator=absent, "
            "terminal reviewer=absent"
        ) in r.output
        assert "provider defaults preserved; no default model selected" in r.output
        assert "openai/ambient-expensive" not in r.output
        assert "openai-secret" not in r.output
        assert "OPENAI_API_KEY" not in r.output

    def test_opencode_init_preserves_unrelated_strict_json_settings(
        self, runner: CliRunner, workspace: Path, env: dict[str, str]
    ) -> None:
        config_path = workspace / ".opencode" / "opencode.json"
        config_path.parent.mkdir()
        before = {
            "$schema": "https://example.invalid/custom-schema.json",
            "theme": "system",
            "path": '/tmp/project "quoted"/space dir',
            "mcp": {
                "existing": {
                    "type": "local",
                    "command": ["node", "-e", 'console.log("kept")'],
                    "environment": {"TOKEN": 'value with "quotes" and spaces'},
                    "enabled": False,
                    "timeout": 1234,
                },
                "zenith": {
                    "type": "local",
                    "command": ["old-zenith"],
                    "environment": {"OLD": "1"},
                    "enabled": False,
                    "timeout": 1,
                },
            },
        }
        config_path.write_text(json.dumps(before, indent=2) + "\n", encoding="utf-8")

        r = runner.invoke(
            cli,
            [
                "init",
                "--workspace-dir",
                str(workspace),
                "--agent",
                "opencode",
                "--worker-acp-command",
                'opencode acp --flag="quoted value"',
                "--validator-provider",
                "codex",
                "--validator-acp-command",
                'codex-acp -c model="gpt-5.6"',
            ],
        )
        assert r.exit_code == 0, r.output

        after = _read_opencode_config(workspace)
        assert after["$schema"] == before["$schema"]
        assert after["theme"] == before["theme"]
        assert after["path"] == before["path"]
        mcp = after["mcp"]
        assert mcp["existing"] == before["mcp"]["existing"]
        assert mcp["zenith"] != before["mcp"]["zenith"]
        server_env = mcp["zenith"]["environment"]
        assert server_env["ZENITH_WORKER_ACP_COMMAND"] == 'opencode acp --flag="quoted value"'
        assert server_env["ZENITH_VALIDATOR_PROVIDER"] == "codex"
        assert server_env["ZENITH_VALIDATOR_ACP_COMMAND"] == 'codex-acp -c model="gpt-5.6"'

    @pytest.mark.parametrize(
        ("contents", "expected"),
        [
            ("{not-json", "expected strict JSON object"),
            ("[]\n", "expected strict JSON object"),
            ('{"mcp": []}\n', "mcp must be an object"),
        ],
    )
    def test_opencode_init_rejects_unsafe_json_without_mutation(
        self,
        runner: CliRunner,
        workspace: Path,
        env: dict[str, str],
        contents: str,
        expected: str,
    ) -> None:
        config_path = workspace / ".opencode" / "opencode.json"
        config_path.parent.mkdir()
        config_path.write_text(contents, encoding="utf-8")

        r = runner.invoke(
            cli, ["init", "--workspace-dir", str(workspace), "--agent", "opencode"]
        )

        assert r.exit_code != 0
        assert expected in r.output
        assert config_path.read_text(encoding="utf-8") == contents
        assert not (workspace / ".opencode" / ".opencode.json.tmp").exists()

    def test_opencode_init_second_run_is_byte_idempotent(
        self, runner: CliRunner, workspace: Path, env: dict[str, str]
    ) -> None:
        config_path = workspace / ".opencode" / "opencode.json"
        config_path.parent.mkdir()
        config_path.write_text(
            json.dumps(
                {
                    "theme": "dark",
                    "mcp": {
                        "existing": {
                            "type": "local",
                            "command": ["node", "server.js"],
                            "environment": {"A": "B"},
                            "enabled": True,
                            "timeout": 1000,
                        }
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        r = runner.invoke(
            cli, ["init", "--workspace-dir", str(workspace), "--agent", "opencode"]
        )
        assert r.exit_code == 0, r.output
        first = config_path.read_bytes()

        r = runner.invoke(
            cli, ["init", "--workspace-dir", str(workspace), "--agent", "opencode"]
        )
        assert r.exit_code == 0, r.output
        assert config_path.read_bytes() == first
        assert _read_opencode_config(workspace)["mcp"]["existing"]["command"] == [
            "node",
            "server.js",
        ]

    def test_opencode_jsonc_coexists_without_mutation_when_not_shadowing_zenith(
        self, runner: CliRunner, workspace: Path, env: dict[str, str]
    ) -> None:
        jsonc_path = workspace / ".opencode" / "opencode.jsonc"
        jsonc_path.parent.mkdir()
        jsonc = """{
  // user comments must stay byte-for-byte
  "mcp": {
    "user-server": {"type": "local", "command": ["node", "server.js"]}
  },
  "theme": "dark",
}
"""
        jsonc_path.write_text(jsonc, encoding="utf-8")

        r = runner.invoke(
            cli, ["init", "--workspace-dir", str(workspace), "--agent", "opencode"]
        )

        assert r.exit_code == 0, r.output
        assert jsonc_path.read_text(encoding="utf-8") == jsonc
        assert _read_opencode_config(workspace)["mcp"]["zenith"]["type"] == "local"

    def test_opencode_jsonc_shadowing_zenith_fails_without_mutation(
        self, runner: CliRunner, workspace: Path, env: dict[str, str]
    ) -> None:
        config_path = workspace / ".opencode" / "opencode.json"
        config_path.parent.mkdir()
        original_json = '{"theme": "dark"}\n'
        config_path.write_text(original_json, encoding="utf-8")
        jsonc_path = workspace / ".opencode" / "opencode.jsonc"
        jsonc = """{
  // Loaded after opencode.json, so this would shadow Zenith's managed entry.
  "mcp": {
    "zenith": {"type": "local", "command": ["shadowed"]}
  }
}
"""
        jsonc_path.write_text(jsonc, encoding="utf-8")

        r = runner.invoke(
            cli, ["init", "--workspace-dir", str(workspace), "--agent", "opencode"]
        )

        assert r.exit_code != 0
        assert "opencode.jsonc" in r.output
        assert "mcp.zenith" in r.output
        assert "remove that jsonc entry" in r.output.lower()
        assert config_path.read_text(encoding="utf-8") == original_json
        assert jsonc_path.read_text(encoding="utf-8") == jsonc

    def test_opencode_root_jsonc_shadowing_zenith_fails_without_mutation(
        self, runner: CliRunner, workspace: Path, env: dict[str, str]
    ) -> None:
        config_path = workspace / ".opencode" / "opencode.json"
        config_path.parent.mkdir()
        original_json = '{"theme": "system"}\n'
        config_path.write_text(original_json, encoding="utf-8")
        jsonc_path = workspace / "opencode.jsonc"
        jsonc = """{
  // Project-root config also loads after the managed .opencode JSON file.
  "mcp": {
    "zenith": {"type": "local", "command": ["shadowed"]}
  }
}
"""
        jsonc_path.write_text(jsonc, encoding="utf-8")

        r = runner.invoke(
            cli, ["init", "--workspace-dir", str(workspace), "--agent", "opencode"]
        )

        assert r.exit_code != 0
        assert "opencode.jsonc" in r.output
        assert "mcp.zenith" in r.output
        assert config_path.read_text(encoding="utf-8") == original_json
        assert jsonc_path.read_text(encoding="utf-8") == jsonc

    def test_opencode_root_strict_json_shadowing_zenith_fails_without_mutation(
        self, runner: CliRunner, workspace: Path, env: dict[str, str]
    ) -> None:
        config_path = workspace / ".opencode" / "opencode.json"
        config_path.parent.mkdir()
        original_managed = '{"theme": "system"}\n'
        config_path.write_text(original_managed, encoding="utf-8")
        root_config_path = workspace / "opencode.json"
        root_config = json.dumps(
            {
                "theme": "root-user-config",
                "mcp": {
                    "zenith": {
                        "type": "local",
                        "command": ["node", "shadowed.js"],
                        "environment": {"SHADOW": "root-strict-json"},
                        "enabled": True,
                        "timeout": 42,
                    }
                },
            },
            indent=2,
        ) + "\n"
        root_config_path.write_text(root_config, encoding="utf-8")

        r = runner.invoke(
            cli, ["init", "--workspace-dir", str(workspace), "--agent", "opencode"]
        )

        assert r.exit_code != 0
        assert "opencode.json" in r.output
        assert "mcp.zenith" in r.output
        assert "remove that root entry" in r.output.lower()
        assert "SHADOW" not in r.output
        assert config_path.read_text(encoding="utf-8") == original_managed
        assert root_config_path.read_text(encoding="utf-8") == root_config

    def test_opencode_init_does_not_persist_ambient_defaults_or_provider_secrets(
        self,
        runner: CliRunner,
        workspace: Path,
        env: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("OPENCODE_MODEL", "openai/expensive")
        monkeypatch.setenv("OPENCODE_PERMISSION", "allow-all")
        monkeypatch.setenv("OPENCODE_PLUGIN", "ambient-plugin")
        monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-secret")
        monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret")

        r = runner.invoke(
            cli, ["init", "--workspace-dir", str(workspace), "--agent", "opencode"]
        )
        assert r.exit_code == 0, r.output

        config_text = (workspace / ".opencode" / "opencode.json").read_text(
            encoding="utf-8"
        )
        config = json.loads(config_text)
        assert "model" not in config
        assert "small_model" not in config
        assert "provider" not in config
        assert "permission" not in config
        assert "plugin" not in config
        assert "openai-secret" not in config_text
        assert "anthropic-secret" not in config_text
        assert "gemini-secret" not in config_text
        server_env = config["mcp"]["zenith"]["environment"]
        assert all("API_KEY" not in key for key in server_env)
        assert all("AUTH_TOKEN" not in key for key in server_env)

    def test_opencode_init_writes_mixed_provider_routing_values(
        self, runner: CliRunner, workspace: Path, env: dict[str, str]
    ) -> None:
        r = runner.invoke(
            cli,
            [
                "init",
                "--workspace-dir",
                str(workspace),
                "--orchestrator-provider",
                "opencode",
                "--worker-provider",
                "opencode",
                "--validator-provider",
                "claude",
                "--terminal-reviewer-provider",
                "codex",
            ],
        )
        assert r.exit_code == 0, r.output

        server_env = _read_opencode_config(workspace)["mcp"]["zenith"]["environment"]
        assert server_env["ZENITH_ORCHESTRATOR_PROVIDER"] == "opencode"
        assert server_env["ZENITH_WORKER_PROVIDER"] == "opencode"
        assert server_env["ZENITH_WORKER_ACP_COMMAND"] == "opencode acp"
        assert server_env["ZENITH_VALIDATOR_PROVIDER"] == "claude"
        assert server_env["ZENITH_VALIDATOR_ACP_COMMAND"] == "claude-agent-acp"
        assert server_env["ZENITH_TERMINAL_REVIEWER_PROVIDER"] == "codex"
        assert server_env["ZENITH_TERMINAL_REVIEWER_ACP_COMMAND"] == "codex-acp"

        config_path = workspace / ".opencode" / "opencode.json"
        assert "OpenCode diagnostics:" in r.output
        assert f"host: opencode ({config_path})" in r.output
        assert (
            "roles: orchestrator=opencode, worker=opencode, "
            "validator=claude, terminal reviewer=codex"
        ) in r.output
        assert (
            "ACP commands: worker=opencode acp, validator=claude-agent-acp, "
            "terminal reviewer=codex-acp"
        ) in r.output
        assert (
            "model pins: worker=absent, validator=absent, "
            "terminal reviewer=absent"
        ) in r.output
        assert "provider defaults preserved; no default model selected" in r.output

    def test_opencode_init_installs_prompt_agents_and_skill_surfaces(
        self, runner: CliRunner, workspace: Path, env: dict[str, str]
    ) -> None:
        r = runner.invoke(
            cli, ["init", "--workspace-dir", str(workspace), "--agent", "opencode"]
        )
        assert r.exit_code == 0, r.output

        prompt_path = workspace / ".opencode" / "orchestrator_prompt.md"
        expected_prompt = (
            BUNDLED_DIR / "prompts" / "orchestrator" / "system_prompt.md"
        ).read_text(encoding="utf-8")
        assert prompt_path.read_text(encoding="utf-8") == expected_prompt
        assert "Created " + str(prompt_path) in r.output

        agents_dir = workspace / ".opencode" / "agents"
        for agent_name in OPENCODE_AGENT_NAMES:
            installed = agents_dir / f"{agent_name}.md"
            assert installed.exists()
            frontmatter, _ = parse_frontmatter(installed.read_text(encoding="utf-8"))
            permission = frontmatter.get("permission")
            assert frontmatter["mode"] == "subagent"
            assert "model" not in frontmatter
            assert isinstance(permission, dict)
            assert permission["edit"] == "deny"
            assert permission["task"] == "deny"

        bundled_skill_names = _bundled_skill_names()
        for rel in (".opencode/skills", ".agents/skills"):
            skill_root = workspace / rel
            skill_names = _installed_skill_names(skill_root)
            assert set(skill_names) == bundled_skill_names
            assert len(skill_names) == len(bundled_skill_names)
            assert f"Installed bundled skills to {skill_root}" in r.output

        assert f"Installed opencode subagents to {agents_dir}" in r.output
        assert not (workspace / ".codex" / "orchestrator_prompt.md").exists()

    def test_opencode_init_preserves_custom_orchestrator_prompt_on_reinit(
        self, runner: CliRunner, workspace: Path, env: dict[str, str]
    ) -> None:
        first = runner.invoke(
            cli, ["init", "--workspace-dir", str(workspace), "--agent", "opencode"]
        )
        assert first.exit_code == 0, first.output

        prompt_path = workspace / ".opencode" / "orchestrator_prompt.md"
        custom_prompt = "# Custom OpenCode Orchestrator\n\nDo not replace me.\n"
        prompt_path.write_text(custom_prompt, encoding="utf-8")

        second = runner.invoke(
            cli, ["init", "--workspace-dir", str(workspace), "--agent", "opencode"]
        )
        assert second.exit_code == 0, second.output
        assert prompt_path.read_text(encoding="utf-8") == custom_prompt
        assert f"Created {prompt_path}" not in second.output

    def test_opencode_init_preserves_existing_skills_and_deduplicates(
        self, runner: CliRunner, workspace: Path, env: dict[str, str]
    ) -> None:
        custom_same_name = (
            "---\n"
            "name: scrutiny-validator\n"
            "description: local replacement\n"
            "---\n\n"
            "Keep this local skill.\n"
        )
        for rel in (".opencode/skills", ".agents/skills"):
            skill = workspace / rel / "scrutiny-validator" / "SKILL.md"
            skill.parent.mkdir(parents=True, exist_ok=True)
            skill.write_text(custom_same_name, encoding="utf-8")

        local_skill = workspace / ".agents" / "skills" / "local-only" / "SKILL.md"
        local_skill.parent.mkdir(parents=True, exist_ok=True)
        local_body = "---\nname: local-only\ndescription: user skill\n---\n\nLocal.\n"
        local_skill.write_text(local_body, encoding="utf-8")

        for _ in range(2):
            r = runner.invoke(
                cli, ["init", "--workspace-dir", str(workspace), "--agent", "opencode"]
            )
            assert r.exit_code == 0, r.output

        for rel in (".opencode/skills", ".agents/skills"):
            skill_root = workspace / rel
            skill = skill_root / "scrutiny-validator" / "SKILL.md"
            skill_names = _installed_skill_names(skill_root)
            assert skill.read_text(encoding="utf-8") == custom_same_name
            assert len(skill_names) == len(set(skill_names))
            assert (skill_root / "engineering-mission-playbook" / "SKILL.md").exists()

        assert local_skill.read_text(encoding="utf-8") == local_body

    def test_codex_init_asset_paths_still_install_codex_and_shared_surfaces(
        self, runner: CliRunner, workspace: Path, env: dict[str, str]
    ) -> None:
        r = runner.invoke(
            cli, ["init", "--workspace-dir", str(workspace), "--agent", "codex"]
        )
        assert r.exit_code == 0, r.output

        assert (workspace / ".codex" / "agents" / "investigator.toml").exists()
        assert (workspace / ".codex" / "orchestrator_prompt.md").exists()
        assert (workspace / ".codex" / "skills" / "scrutiny-validator" / "SKILL.md").exists()
        assert (workspace / ".agents" / "skills" / "scrutiny-validator" / "SKILL.md").exists()
        assert not (workspace / ".opencode").exists()

    def test_opencode_explicit_provider_selectors_are_accepted(
        self, runner: CliRunner, workspace: Path, env: dict[str, str]
    ) -> None:
        r = runner.invoke(
            cli,
            [
                "init",
                "--workspace-dir",
                str(workspace),
                "--orchestrator-provider",
                "opencode",
                "--worker-provider",
                "opencode",
                "--validator-provider",
                "opencode",
                "--terminal-reviewer-provider",
                "opencode",
            ],
        )
        assert r.exit_code == 0, r.output
        assert (workspace / ".opencode" / "opencode.json").exists()



class TestListProjects:
    def test_empty(self, runner: CliRunner, env: dict[str, str]) -> None:
        r = runner.invoke(cli, ["list-projects"])
        assert r.exit_code == 0
        assert "No projects" in r.output

    def test_after_creation(
        self, runner: CliRunner, workspace: Path, harness_home: Path, env: dict[str, str]
    ) -> None:
        from zenith_harness.config import HarnessConfig
        from zenith_harness.storage import ProjectStore

        ProjectStore(HarnessConfig.discover()).create_project(
            "brief", workspace, project_id="proj-x"
        )
        r = runner.invoke(cli, ["list-projects"])
        assert "proj-x" in r.output


class TestShowProject:
    def test_unknown_id(self, runner: CliRunner, env: dict[str, str]) -> None:
        r = runner.invoke(cli, ["show-project", "ghost"])
        assert r.exit_code != 0
        assert "not found" in r.output.lower()
