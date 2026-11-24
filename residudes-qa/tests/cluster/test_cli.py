"""Tests for CLI module."""

import subprocess
import sys
import pytest


class TestCLIHelp:
    """Verify all CLI subcommands are accessible."""

    @pytest.mark.parametrize("cmd", [
        "run", "analyze", "report", "heatmap",
        "comprehensive", "network", "consensus", "mpnn-score",
    ])
    def test_subcommand_help(self, cmd):
        result = subprocess.run(
            [sys.executable, "-m", "residudes_qa.cluster.cli", cmd, "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert "--help" not in result.stderr

    def test_main_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "residudes_qa.cluster.cli", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert "comprehensive" in result.stdout
        assert "network" in result.stdout

    def test_version(self):
        result = subprocess.run(
            [sys.executable, "-m", "residudes_qa.cluster.cli", "--version"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0


class TestCLIFlags:
    """Verify key CLI flags exist."""

    def test_comprehensive_flags(self):
        result = subprocess.run(
            [sys.executable, "-m", "residudes_qa.cluster.cli", "comprehensive", "--help"],
            capture_output=True, text=True,
        )
        for flag in ["--matrix", "--names", "--output", "--method", "--k-max",
                      "--no-adaptive-k", "--scores", "--pdbs", "--ranking"]:
            assert flag in result.stdout, f"Missing flag: {flag}"

    def test_network_flags(self):
        result = subprocess.run(
            [sys.executable, "-m", "residudes_qa.cluster.cli", "network", "--help"],
            capture_output=True, text=True,
        )
        for flag in ["--community-method", "--resolution", "--resolution-scan", "--visualize"]:
            assert flag in result.stdout, f"Missing flag: {flag}"
