from std.sys.defines import get_defined_int
from benchmark import run
from pretokenizer import GPreTokenizer, GPT2Pretokenizer, GPT4Pretokenizer

comptime bpe_pt = get_defined_int["BPE_PT", 0]()

def main() raises:
    comptime if bpe_pt == 1:
        run[GPT2Pretokenizer]("BPETokenizer[GPT2Pretokenizer]")
    elif bpe_pt == 2:
        run[GPT4Pretokenizer]("BPETokenizer[GPT4Pretokenizer]")
    else:
        run[GPreTokenizer]("BPETokenizer[GPreTokenizer]")
