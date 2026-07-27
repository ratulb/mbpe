#!/usr/bin/env python3
"""Print benchmark environment hardware/tool info as markdown."""

import datetime
import os
import platform
import subprocess
import sys


def run(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else "N/A"
    except Exception:
        return "N/A"


def cpu_model():
    out = run(["lscpu"])
    for line in out.split("\n"):
        if "Model name" in line:
            return line.split(":", 1)[-1].strip()
    # fallback
    out = run(["cat", "/proc/cpuinfo"])
    for line in out.split("\n"):
        if line.startswith("model name"):
            return line.split(":", 1)[-1].strip()
    return platform.processor() or "N/A"


def cpu_cores():
    out = run(["nproc"])
    try:
        return int(out.strip())
    except (ValueError, TypeError):
        return "N/A"


def ram_gb():
    out = run(["free", "-h"])
    for line in out.split("\n"):
        if line.startswith("Mem:"):
            parts = line.split()
            if len(parts) >= 2:
                return parts[1]
    return "N/A"


def os_info():
    out = run(["cat", "/etc/os-release"])
    name = ""
    version = ""
    for line in out.split("\n"):
        if line.startswith("PRETTY_NAME="):
            return line.split('=', 1)[-1].strip().strip('"')
        if line.startswith("NAME="):
            name = line.split('=', 1)[-1].strip().strip('"')
        if line.startswith("VERSION_ID="):
            version = line.split('=', 1)[-1].strip().strip('"')
    if name:
        return f"{name} {version}".strip()
    return platform.system() + " " + platform.release()


def pixi_python_version():
    out = run(["pixi", "run", "python", "--version"])
    if out and out != "N/A":
        return out
    return run(["python3", "--version"]) or sys.version.split()[0]


def mojo_version():
    return run(["mojo", "--version"])


def rust_version():
    return run(["rustc", "--version"])


def tiktoken_version():
    try:
        import tiktoken
        return tiktoken.__version__
    except ImportError:
        return "N/A"


def main():
    print("## Benchmark Environment")
    print()
    print(f"| Property | Value |")
    print(f"|----------|-------|")
    print(f"| Date | {datetime.date.today()} |")
    print(f"| CPU | {cpu_model()} |")
    cores = cpu_cores()
    print(f"| Cores | {cores} (logical) |")
    print(f"| RAM | {ram_gb()} |")
    print(f"| OS | {os_info()} |")
    print(f"| Mojo | {mojo_version()} |")
    print(f"| Python (pixi) | {pixi_python_version()} |")
    print(f"| Rust | {rust_version()} |")
    print(f"| tiktoken | {tiktoken_version()} |")
    print()


if __name__ == "__main__":
    main()
