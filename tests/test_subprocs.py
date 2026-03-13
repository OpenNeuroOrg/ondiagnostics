from pathlib import Path

import pytest

from ondiagnostics.subprocs import git


async def test_git_success(git_repo_with_tag: tuple[Path, str]) -> None:
    """Test git command execution with successful result."""
    repo_path, commit_sha = git_repo_with_tag

    result = await git("ls-remote", str(repo_path))

    expected_output = f"{commit_sha}\trefs/tags/1.0.0\n".encode()
    assert result.returncode == 0
    assert expected_output in result.stdout
    assert result.stderr == b""


async def test_git_failure() -> None:
    """Test git command execution with failure."""
    # Use a command that will fail
    result = await git("ls-remote", "/nonexistent/repo.git")

    assert result.returncode != 0
    assert result.stderr != b""


async def test_git_multiple_args(git_repo_with_tag: tuple[Path, str]) -> None:
    """Test git command with multiple arguments."""
    repo_path, commit_sha = git_repo_with_tag

    result = await git("ls-remote", "--exit-code", str(repo_path), "1.0.0")

    assert result.returncode == 0
    assert result.stdout == f"{commit_sha}\trefs/tags/1.0.0\n".encode()
