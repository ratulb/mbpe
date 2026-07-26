"""> pixi run benchmark-gpt2 (GPT2Pretokenizer)."""
from benchmarks.benchmark import run
from pretokenizer import GPT2Pretokenizer

def main() raises:
    run[GPT2Pretokenizer]("gpt2")
