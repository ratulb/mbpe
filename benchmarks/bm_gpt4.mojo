"""> pixi run benchmark-gpt4 (GPT4Pretokenizer)."""
from benchmarks.benchmark import run
from pretokenizer import GPT4Pretokenizer

def main() raises:
    run[GPT4Pretokenizer]("gpt4")
