import asyncio
from asyncio.subprocess import PIPE
from dataclasses import dataclass, field


@dataclass
class SubprocessResult:
    args: tuple[str, ...]
    returncode: int
    stdout: bytes = field(repr=False)
    stderr: bytes = field(repr=False)


async def git(*args: str) -> SubprocessResult:
    """Run a git command and return the exit code, stdout, and stderr."""
    args = ("git", *args)

    proc = await asyncio.create_subprocess_exec(*args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = await proc.communicate()
    assert proc.returncode is not None
    return SubprocessResult(
        args=args, returncode=proc.returncode, stdout=stdout, stderr=stderr
    )
