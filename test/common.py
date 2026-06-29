import os
from pathlib import Path
import shlex
import subprocess


def runcmd(cmd, wd=None, env=None):
    """
    Run a command, print stderr to stdout and optionally run in working directory
    """
    cwd = Path.cwd() if wd is None else wd
    local_env = os.environ.copy()
    if env is not None:
        local_env.update(env)
    subprocess.run(
        shlex.split(cmd), stderr=subprocess.STDOUT, cwd=cwd, env=local_env, check=True
    )


def make_nc(tmp_path, cdl_file):
    nc_filename = Path(cdl_file).with_suffix(".nc").name
    filepath = f"{tmp_path}/{nc_filename}"
    cmd = f"ncgen -o {filepath}  {cdl_file}"

    runcmd(cmd)

    return filepath
