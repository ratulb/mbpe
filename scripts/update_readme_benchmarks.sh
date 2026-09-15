#!/usr/bin/env bash
# Update the *Environment:* line in README.md from the current toolchain.
# The benchmark chart itself is owned by scripts/generate_benchmark_chart.py
# (run via bash readme_bench_update.sh); this script only refreshes the
# environment caption so a standalone re-run stays consistent.
# Usage:  bash scripts/update_readme_benchmarks.sh
set -euo pipefail
cd "$(dirname "$0")/.."

RESULTS="benchmarks/results"
README="README.md"

echo "  Generating environment line..." >&2

python3 -c "
import platform, subprocess, sys

def run(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else 'N/A'
    except Exception:
        return 'N/A'

def cpu_model():
    out = run(['lscpu'])
    for line in out.split('\n'):
        if 'Model name' in line:
            return line.split(':', 1)[-1].strip()
    return platform.processor() or 'N/A'

def cpu_cores():
    return run(['nproc'])

def ram():
    out = run(['free', '-h'])
    for line in out.split('\n'):
        if line.startswith('Mem:'):
            parts = line.split()
            if len(parts) >= 2:
                return parts[1]
    return 'N/A'

def os_info():
    out = run(['cat', '/etc/os-release'])
    for line in out.split('\n'):
        if line.startswith('PRETTY_NAME='):
            return line.split('=', 1)[-1].strip().strip('\"')
    return platform.system()

def mojo_ver():
    # Try pixi first (mojo not on PATH in pixi-managed envs)
    v = run(['pixi', 'run', 'mojo', '--version'])
    if v != 'N/A':
        # Extract version number from 'Mojo 1.0.0 (...)' or similar
        parts = v.split()
        if len(parts) >= 2:
            return parts[1]
        return v
    v = run(['mojo', '--version'])
    return v.split()[1] if v != 'N/A' and len(v.split()) >= 2 else 'N/A'

def python_ver():
    v = run(['pixi', 'run', 'python', '--version'])
    if v != 'N/A':
        parts = v.split()
        return parts[1] if len(parts) >= 2 else v
    return sys.version.split()[0]

def rust_ver():
    import os
    # Try system rustc first
    v = run(['rustc', '--version'])
    if v != 'N/A':
        parts = v.split()
        return parts[1] if len(parts) >= 2 else v
    # Try benchmark-installed rustup at /tmp/mbpe-rust
    rustup = '/tmp/mbpe-rust/cargo/bin/rustup'
    if os.path.isfile(rustup):
        env = os.environ.copy()
        env['RUSTUP_HOME'] = '/tmp/mbpe-rust/rustup'
        env['CARGO_HOME'] = '/tmp/mbpe-rust/cargo'
        try:
            r = subprocess.run(
                [rustup, 'run', 'stable', 'rustc', '--version'],
                capture_output=True, text=True, timeout=5, env=env,
            )
            if r.returncode == 0 and r.stdout.strip():
                parts = r.stdout.strip().split()
                return parts[1] if len(parts) >= 2 else r.stdout.strip()
        except Exception:
            pass
    return 'N/A'

def tiktoken_ver():
    v = run(['pixi', 'run', '-e', 'dev', 'python', '-c',
             'import tiktoken; print(tiktoken.__version__)'])
    return v if v != 'N/A' else 'N/A'

cpu = cpu_model()
cores = cpu_cores()
mem = ram()
os_name = os_info()
mojo = mojo_ver()
py = python_ver()
rust = rust_ver()
tt = tiktoken_ver()

# Match the existing README format:
# *Environment: Intel Xeon @ 3.10 GHz, 8 cores, 31 Gi RAM, Ubuntu 24.04. Mojo 1.0.0, Python 3.14.7, Rust 1.97.1, tiktoken 0.14.0.*
parts = [f'{cpu}, {cores} cores, {mem} RAM, {os_name}.']
tools = []
if mojo != 'N/A':   tools.append(f'Mojo {mojo}')
if py != 'N/A':     tools.append(f'Python {py}')
if rust != 'N/A':   tools.append(f'Rust {rust}')
if tt != 'N/A':     tools.append(f'tiktoken {tt}')
parts.append(', '.join(tools))
print(f'*Environment: {\" \".join(parts)}.*')
" > /tmp/rb_environment_line.txt

echo "  Updating $README..." >&2

python3 -c "
import re

with open('$README') as f:
    content = f.read()

env_line = open('/tmp/rb_environment_line.txt').read().strip()

# Replace environment line
content = re.sub(
    r'\*Environment:.*\*',
    env_line,
    content,
)

with open('$README', 'w') as f:
    f.write(content)

print('Done.')
"
