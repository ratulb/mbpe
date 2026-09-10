trait Tokenizer:

    def decode(self, token_ids: List[Int]) raises -> String:
        ...

    def encode(self, text: String) raises -> List[Int]:
        ...

