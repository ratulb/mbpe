"""> pixi run benchmark (GPreTokenizer)."""
from benchmarks.benchmark import run
from pretokenizer import GPreTokenizer

def main() raises:
    run[GPreTokenizer]("default")
