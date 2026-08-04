"""Python bindings for mbpe — all 3 BPETokenizer variants.

Build: mojo build python-binding/mbpe.mojo -I . --emit shared-lib -o python-binding/mbpe/_mbpe.so
Use:   PYTHONPATH=python-binding python -c "import mbpe; tok = mbpe.GPT2Tokenizer()"
"""

from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from bpe.tokenizer import BPETokenizer
from bpe.pretokenizer import (
    GPT2Pretokenizer,
    GPT4Pretokenizer,
    ByteMapping,
)


# ── Type aliases ─────────────────────────────────────────────────

comptime GPT2TK = BPETokenizer[GPT2Pretokenizer]
comptime GPT4TK = BPETokenizer[GPT4Pretokenizer[ByteMapping.SEQUENTIAL]]
comptime GPT4oTK = BPETokenizer[GPT4Pretokenizer[ByteMapping.SHUFFLED]]


# ── Helpers ──────────────────────────────────────────────────────

def _list_of_int_to_py(ids: List[Int]) raises -> PythonObject:
    var n = len(ids)
    ref cpy = Python().cpython()
    var list_ptr = cpy.PyList_New(n)
    for i in range(n):
        _ = cpy.PyList_SetItem(list_ptr, i, cpy.PyLong_FromSsize_t(ids[i]))
    return PythonObject(from_owned=list_ptr)


def _py_dict_to_mojo(py_dict: PythonObject) raises -> Dict[String, Int]:
    var result = Dict[String, Int]()
    for key in py_dict:
        result[String(key)] = Int(py=py_dict[key])
    return result^


def _uint8_list_to_py_bytes(data: List[UInt8]) raises -> PythonObject:
    var _b = Python.import_module("builtins")
    var py_vals = List[PythonObject](capacity=len(data))
    for i in range(len(data)):
        py_vals.append(Python.int(data[i]))
    return _b.bytes(Python.list(Span[PythonObject](py_vals)))


def _uint8_list_list_to_py_list(data: List[List[UInt8]]) raises -> PythonObject:
    var py_vals = List[PythonObject](capacity=len(data))
    for i in range(len(data)):
        py_vals.append(_uint8_list_to_py_bytes(data[i]))
    return Python.list(Span[PythonObject](py_vals))


def _py_ids_to_mojo(py_ids: PythonObject) raises -> List[Int]:
    var n = len(py_ids)
    ref cpy = Python().cpython()
    var list_ptr = py_ids.steal_data()
    var ids = List[Int](capacity=n)
    for i in range(n):
        ids.append(Int(cpy.PyLong_AsSsize_t(cpy.PyList_GetItem(list_ptr, i))))
    return ids^


# ── Parameterized binding helpers (instantiated per-type) ──────
# Each method is duplicated for all 3 types to work around Mojo's
# trait constraint limitation on downcast_value_ptr.

def _name_gpt2(mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    _ = args; return PythonObject(self.downcast_value_ptr[GPT2TK]()[].name())

def _name_gpt4(mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    _ = args; return PythonObject(self.downcast_value_ptr[GPT4TK]()[].name())

def _name_gpt4o(mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    _ = args; return PythonObject(self.downcast_value_ptr[GPT4oTK]()[].name())

def _decode_bytes_gpt2(mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    var ptr = self.downcast_value_ptr[GPT2TK]()
    return _uint8_list_to_py_bytes(ptr[].decode_bytes(Span[Int](_py_ids_to_mojo(args[0]))))

def _decode_bytes_gpt4(mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    var ptr = self.downcast_value_ptr[GPT4TK]()
    return _uint8_list_to_py_bytes(ptr[].decode_bytes(Span[Int](_py_ids_to_mojo(args[0]))))

def _decode_bytes_gpt4o(mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    var ptr = self.downcast_value_ptr[GPT4oTK]()
    return _uint8_list_to_py_bytes(ptr[].decode_bytes(Span[Int](_py_ids_to_mojo(args[0]))))

def _encode_single_token_gpt2(mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    var ptr = self.downcast_value_ptr[GPT2TK]()
    return PythonObject(ptr[].encode_single_token(Python().as_string_slice(args[0])))

def _encode_single_token_gpt4(mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    var ptr = self.downcast_value_ptr[GPT4TK]()
    return PythonObject(ptr[].encode_single_token(Python().as_string_slice(args[0])))

def _encode_single_token_gpt4o(mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    var ptr = self.downcast_value_ptr[GPT4oTK]()
    return PythonObject(ptr[].encode_single_token(Python().as_string_slice(args[0])))

def _token_byte_values_gpt2(mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    _ = args
    return _uint8_list_list_to_py_list(self.downcast_value_ptr[GPT2TK]()[].token_byte_values())

def _token_byte_values_gpt4(mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    _ = args
    return _uint8_list_list_to_py_list(self.downcast_value_ptr[GPT4TK]()[].token_byte_values())

def _token_byte_values_gpt4o(mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    _ = args
    return _uint8_list_list_to_py_list(self.downcast_value_ptr[GPT4oTK]()[].token_byte_values())

def _decode_single_token_bytes_gpt2(mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    var ptr = self.downcast_value_ptr[GPT2TK]()
    return _uint8_list_to_py_bytes(ptr[].decode_single_token_bytes(Int(py=args[0])))

def _decode_single_token_bytes_gpt4(mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    var ptr = self.downcast_value_ptr[GPT4TK]()
    return _uint8_list_to_py_bytes(ptr[].decode_single_token_bytes(Int(py=args[0])))

def _decode_single_token_bytes_gpt4o(mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    var ptr = self.downcast_value_ptr[GPT4oTK]()
    return _uint8_list_to_py_bytes(ptr[].decode_single_token_bytes(Int(py=args[0])))

def _decode_with_offsets_gpt2(mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    var ptr = self.downcast_value_ptr[GPT2TK]()
    var ids = _py_ids_to_mojo(args[0])
    var starts = List[Int]()
    var ends = List[Int]()
    var text = ptr[].decode_with_offsets(Span[Int](ids), starts, ends)
    var _b = Python.import_module("builtins")
    ref cpy = Python().cpython()
    var py_vals = List[PythonObject](capacity=len(starts))
    for i in range(len(starts)):
        var tup = cpy.PyTuple_New(2)
        _ = cpy.PyTuple_SetItem(tup, 0, cpy.PyLong_FromSsize_t(starts[i]))
        _ = cpy.PyTuple_SetItem(tup, 1, cpy.PyLong_FromSsize_t(ends[i]))
        py_vals.append(PythonObject(from_owned=tup))
    return _b.eval("lambda t, o: (t, o)")(PythonObject(text), Python.list(Span[PythonObject](py_vals)))

def _decode_with_offsets_gpt4(mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    var ptr = self.downcast_value_ptr[GPT4TK]()
    var ids = _py_ids_to_mojo(args[0])
    var starts = List[Int]()
    var ends = List[Int]()
    var text = ptr[].decode_with_offsets(Span[Int](ids), starts, ends)
    var _b = Python.import_module("builtins")
    ref cpy = Python().cpython()
    var py_vals = List[PythonObject](capacity=len(starts))
    for i in range(len(starts)):
        var tup = cpy.PyTuple_New(2)
        _ = cpy.PyTuple_SetItem(tup, 0, cpy.PyLong_FromSsize_t(starts[i]))
        _ = cpy.PyTuple_SetItem(tup, 1, cpy.PyLong_FromSsize_t(ends[i]))
        py_vals.append(PythonObject(from_owned=tup))
    return _b.eval("lambda t, o: (t, o)")(PythonObject(text), Python.list(Span[PythonObject](py_vals)))

def _decode_with_offsets_gpt4o(mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    var ptr = self.downcast_value_ptr[GPT4oTK]()
    var ids = _py_ids_to_mojo(args[0])
    var starts = List[Int]()
    var ends = List[Int]()
    var text = ptr[].decode_with_offsets(Span[Int](ids), starts, ends)
    var _b = Python.import_module("builtins")
    ref cpy = Python().cpython()
    var py_vals = List[PythonObject](capacity=len(starts))
    for i in range(len(starts)):
        var tup = cpy.PyTuple_New(2)
        _ = cpy.PyTuple_SetItem(tup, 0, cpy.PyLong_FromSsize_t(starts[i]))
        _ = cpy.PyTuple_SetItem(tup, 1, cpy.PyLong_FromSsize_t(ends[i]))
        py_vals.append(PythonObject(from_owned=tup))
    return _b.eval("lambda t, o: (t, o)")(PythonObject(text), Python.list(Span[PythonObject](py_vals)))


# ═════════════════════════════════════════════════════════════════
# GPT2Tokenizer
# ═════════════════════════════════════════════════════════════════

def _init_gpt2(args: PythonObject, kwargs: PythonObject) raises -> GPT2TK:
    return GPT2TK()

def _train_gpt2(mut self: PythonObject, mut args: PythonObject, mut kwargs: PythonObject) raises -> PythonObject:
    var ptr = self.downcast_value_ptr[GPT2TK]()
    var corpus_py = args[0]
    var vocab_size = Int(py=args[1]) if len(args) >= 2 else Int(py=kwargs["vocab_size"])
    var corpus = List[String]()
    for i in range(len(corpus_py)): corpus.append(String(corpus_py[i]))
    ptr[].train(Span[String](corpus), vocab_size)
    return PythonObject(None)

def _encode_gpt2(mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    var ptr = self.downcast_value_ptr[GPT2TK]()
    return _list_of_int_to_py(ptr[].encode(Python().as_string_slice(args[0])))

def _encode_ordinary_gpt2(mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    var ptr = self.downcast_value_ptr[GPT2TK]()
    return _list_of_int_to_py(ptr[].encode_ordinary(Python().as_string_slice(args[0])))

def _decode_gpt2(mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    var ptr = self.downcast_value_ptr[GPT2TK]()
    var py_ids = args[0]
    var n = len(py_ids)
    ref cpy = Python().cpython()
    var list_ptr = py_ids.steal_data()
    var ids = List[Int](capacity=n)
    for i in range(n):
        ids.append(Int(cpy.PyLong_AsSsize_t(cpy.PyList_GetItem(list_ptr, i))))
    return PythonObject(ptr[].decode(Span[Int](ids)))

def _n_vocab_gpt2(mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    _ = args; return PythonObject(len(self.downcast_value_ptr[GPT2TK]()[]))

def _save_tiktok_gpt2(mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    self.downcast_value_ptr[GPT2TK]()[].save_tiktoken(String(args[0]))
    return PythonObject(None)

def _load_tiktok_gpt2(mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    self.downcast_value_ptr[GPT2TK]()[].load_tiktoken(String(args[0]))
    return PythonObject(None)

def _reg_special_gpt2(mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    var ptr = self.downcast_value_ptr[GPT2TK]()
    var tokens = _py_dict_to_mojo(args[0])
    ptr[].register_special_tokens(tokens^)
    return PythonObject(None)

def _get_specials_gpt2(mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    _ = args
    var ptr = self.downcast_value_ptr[GPT2TK]()
    var result = Python.evaluate("{}")
    for item in ptr[].special_bytes.items():
        result[PythonObject(item.key)] = Python.int(item.value)
    return result

def _set_specials_gpt2(mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    var ptr = self.downcast_value_ptr[GPT2TK]()
    var tokens = _py_dict_to_mojo(args[0])
    ptr[].special_bytes = Dict[String, Int]()
    ptr[].inverse_special = Dict[Int, String]()
    ptr[].register_special_tokens(tokens^)
    return PythonObject(None)


# ═════════════════════════════════════════════════════════════════
# GPT4Tokenizer (cl100k / ByteMapping.SEQUENTIAL)
# ═════════════════════════════════════════════════════════════════

def _init_gpt4(args: PythonObject, kwargs: PythonObject) raises -> GPT4TK:
    return GPT4TK()

def _train_gpt4(mut self: PythonObject, mut args: PythonObject, mut kwargs: PythonObject) raises -> PythonObject:
    var ptr = self.downcast_value_ptr[GPT4TK]()
    var corpus_py = args[0]
    var vocab_size = Int(py=args[1]) if len(args) >= 2 else Int(py=kwargs["vocab_size"])
    var corpus = List[String]()
    for i in range(len(corpus_py)): corpus.append(String(corpus_py[i]))
    ptr[].train(Span[String](corpus), vocab_size)
    return PythonObject(None)

def _encode_gpt4(mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    var ptr = self.downcast_value_ptr[GPT4TK]()
    return _list_of_int_to_py(ptr[].encode(Python().as_string_slice(args[0])))

def _encode_ordinary_gpt4(mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    var ptr = self.downcast_value_ptr[GPT4TK]()
    return _list_of_int_to_py(ptr[].encode_ordinary(Python().as_string_slice(args[0])))

def _decode_gpt4(mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    var ptr = self.downcast_value_ptr[GPT4TK]()
    var py_ids = args[0]
    var n = len(py_ids)
    ref cpy = Python().cpython()
    var list_ptr = py_ids.steal_data()
    var ids = List[Int](capacity=n)
    for i in range(n):
        ids.append(Int(cpy.PyLong_AsSsize_t(cpy.PyList_GetItem(list_ptr, i))))
    return PythonObject(ptr[].decode(Span[Int](ids)))

def _n_vocab_gpt4(mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    _ = args; return PythonObject(len(self.downcast_value_ptr[GPT4TK]()[]))

def _save_tiktok_gpt4(mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    self.downcast_value_ptr[GPT4TK]()[].save_tiktoken(String(args[0]))
    return PythonObject(None)

def _load_tiktok_gpt4(mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    self.downcast_value_ptr[GPT4TK]()[].load_tiktoken(String(args[0]))
    return PythonObject(None)

def _reg_special_gpt4(mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    var ptr = self.downcast_value_ptr[GPT4TK]()
    var tokens = _py_dict_to_mojo(args[0])
    ptr[].register_special_tokens(tokens^)
    return PythonObject(None)

def _get_specials_gpt4(mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    _ = args
    var ptr = self.downcast_value_ptr[GPT4TK]()
    var result = Python.evaluate("{}")
    for item in ptr[].special_bytes.items():
        result[PythonObject(item.key)] = Python.int(item.value)
    return result

def _set_specials_gpt4(mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    var ptr = self.downcast_value_ptr[GPT4TK]()
    var tokens = _py_dict_to_mojo(args[0])
    ptr[].special_bytes = Dict[String, Int]()
    ptr[].inverse_special = Dict[Int, String]()
    ptr[].register_special_tokens(tokens^)
    return PythonObject(None)


# ═════════════════════════════════════════════════════════════════
# GPT4oTokenizer (o200k / ByteMapping.SHUFFLED)
# ═════════════════════════════════════════════════════════════════

def _init_gpt4o(args: PythonObject, kwargs: PythonObject) raises -> GPT4oTK:
    return GPT4oTK()

def _train_gpt4o(mut self: PythonObject, mut args: PythonObject, mut kwargs: PythonObject) raises -> PythonObject:
    var ptr = self.downcast_value_ptr[GPT4oTK]()
    var corpus_py = args[0]
    var vocab_size = Int(py=args[1]) if len(args) >= 2 else Int(py=kwargs["vocab_size"])
    var corpus = List[String]()
    for i in range(len(corpus_py)): corpus.append(String(corpus_py[i]))
    ptr[].train(Span[String](corpus), vocab_size)
    return PythonObject(None)

def _encode_gpt4o(mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    var ptr = self.downcast_value_ptr[GPT4oTK]()
    return _list_of_int_to_py(ptr[].encode(Python().as_string_slice(args[0])))

def _encode_ordinary_gpt4o(mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    var ptr = self.downcast_value_ptr[GPT4oTK]()
    return _list_of_int_to_py(ptr[].encode_ordinary(Python().as_string_slice(args[0])))

def _decode_gpt4o(mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    var ptr = self.downcast_value_ptr[GPT4oTK]()
    var py_ids = args[0]
    var n = len(py_ids)
    ref cpy = Python().cpython()
    var list_ptr = py_ids.steal_data()
    var ids = List[Int](capacity=n)
    for i in range(n):
        ids.append(Int(cpy.PyLong_AsSsize_t(cpy.PyList_GetItem(list_ptr, i))))
    return PythonObject(ptr[].decode(Span[Int](ids)))

def _n_vocab_gpt4o(mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    _ = args; return PythonObject(len(self.downcast_value_ptr[GPT4oTK]()[]))

def _save_tiktok_gpt4o(mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    self.downcast_value_ptr[GPT4oTK]()[].save_tiktoken(String(args[0]))
    return PythonObject(None)

def _load_tiktok_gpt4o(mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    self.downcast_value_ptr[GPT4oTK]()[].load_tiktoken(String(args[0]))
    return PythonObject(None)

def _reg_special_gpt4o(mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    var ptr = self.downcast_value_ptr[GPT4oTK]()
    var tokens = _py_dict_to_mojo(args[0])
    ptr[].register_special_tokens(tokens^)
    return PythonObject(None)

def _get_specials_gpt4o(mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    _ = args
    var ptr = self.downcast_value_ptr[GPT4oTK]()
    var result = Python.evaluate("{}")
    for item in ptr[].special_bytes.items():
        result[PythonObject(item.key)] = Python.int(item.value)
    return result

def _set_specials_gpt4o(mut self: PythonObject, mut args: PythonObject) raises -> PythonObject:
    var ptr = self.downcast_value_ptr[GPT4oTK]()
    var tokens = _py_dict_to_mojo(args[0])
    ptr[].special_bytes = Dict[String, Int]()
    ptr[].inverse_special = Dict[Int, String]()
    ptr[].register_special_tokens(tokens^)
    return PythonObject(None)


# ── Module-level factory ─────────────────────────────────────────

def _find_data_dir() raises -> String:
    var sys = Python.import_module("sys")
    var os = Python.import_module("os")
    var mod = sys.modules["mbpe"]
    var so_path = String(mod.__file__)
    return String(os.path.dirname(os.path.realpath(so_path)))

def py_get_encoding(name: PythonObject) raises -> PythonObject:
    var enc_name = String(name)
    var data_dir = _find_data_dir()
    if enc_name == "gpt2" or enc_name == "r50k_base":
        var tok = GPT2TK()
        tok.load_tiktoken(data_dir + "/gpt2.tiktoken")
        return PythonObject(alloc=tok^)
    elif enc_name == "p50k_base":
        var tok = GPT2TK()
        tok.load_tiktoken(data_dir + "/p50k_base.tiktoken")
        return PythonObject(alloc=tok^)
    elif enc_name == "p50k_edit":
        var tok = GPT2TK()
        tok.load_tiktoken(data_dir + "/p50k_edit.tiktoken")
        return PythonObject(alloc=tok^)
    elif enc_name == "cl100k":
        var tok = GPT4TK()
        tok.load_tiktoken(data_dir + "/cl100k.tiktoken")
        return PythonObject(alloc=tok^)
    elif enc_name == "o200k":
        var tok = GPT4oTK()
        tok.load_tiktoken(data_dir + "/o200k.tiktoken")
        return PythonObject(alloc=tok^)
    elif enc_name == "o200k_harmony":
        var tok = GPT4oTK()
        tok.load_tiktoken(data_dir + "/o200k_harmony.tiktoken")
        return PythonObject(alloc=tok^)
    else:
        raise Error("unknown encoding: " + enc_name)


def py_train(
    texts: PythonObject,
    vocab_size: PythonObject,
    pretokenizer: PythonObject,
) raises -> PythonObject:
    var pt_name = String(pretokenizer)
    var vocab_sz = Int(py=vocab_size)
    var corpus = List[String]()
    for i in range(len(texts)):
        corpus.append(String(texts[i]))

    if pt_name == "gpt2":
        var tok = GPT2TK()
        tok.train(Span[String](corpus), vocab_sz)
        return PythonObject(alloc=tok^)
    elif pt_name == "gpt4":
        var tok = GPT4TK()
        tok.train(Span[String](corpus), vocab_sz)
        return PythonObject(alloc=tok^)
    else:
        raise Error("unknown pretokenizer: '" + pt_name + "'; use 'gpt2' or 'gpt4'")


# ── Module entry point ───────────────────────────────────────────

@export("PyInit__mbpe")
def PyInit_mbpe() abi("C") -> PythonObject:
    try:
        var mb = PythonModuleBuilder("mbpe")

        _ = mb.add_type[GPT2TK]("GPT2Tokenizer") \
            .def_py_init[_init_gpt2]() \
            .def_py_method[_train_gpt2]("train") \
            .def_py_method[_encode_gpt2]("encode") \
            .def_py_method[_encode_ordinary_gpt2]("encode_ordinary") \
            .def_py_method[_decode_gpt2]("decode") \
            .def_py_method[_n_vocab_gpt2]("n_vocab") \
            .def_py_method[_save_tiktok_gpt2]("save_tiktoken") \
            .def_py_method[_load_tiktok_gpt2]("load_tiktoken") \
            .def_py_method[_reg_special_gpt2]("register_special_tokens") \
            .def_py_method[_get_specials_gpt2]("_get_special_bytes") \
            .def_py_method[_set_specials_gpt2]("_set_special_bytes") \
            .def_py_method[_name_gpt2]("name") \
            .def_py_method[_decode_bytes_gpt2]("decode_bytes") \
            .def_py_method[_encode_single_token_gpt2]("encode_single_token") \
            .def_py_method[_token_byte_values_gpt2]("token_byte_values") \
            .def_py_method[_decode_single_token_bytes_gpt2]("decode_single_token_bytes") \
            .def_py_method[_decode_with_offsets_gpt2]("decode_with_offsets")

        _ = mb.add_type[GPT4TK]("GPT4Tokenizer") \
            .def_py_init[_init_gpt4]() \
            .def_py_method[_train_gpt4]("train") \
            .def_py_method[_encode_gpt4]("encode") \
            .def_py_method[_encode_ordinary_gpt4]("encode_ordinary") \
            .def_py_method[_decode_gpt4]("decode") \
            .def_py_method[_n_vocab_gpt4]("n_vocab") \
            .def_py_method[_save_tiktok_gpt4]("save_tiktoken") \
            .def_py_method[_load_tiktok_gpt4]("load_tiktoken") \
            .def_py_method[_reg_special_gpt4]("register_special_tokens") \
            .def_py_method[_get_specials_gpt4]("_get_special_bytes") \
            .def_py_method[_set_specials_gpt4]("_set_special_bytes") \
            .def_py_method[_name_gpt4]("name") \
            .def_py_method[_decode_bytes_gpt4]("decode_bytes") \
            .def_py_method[_encode_single_token_gpt4]("encode_single_token") \
            .def_py_method[_token_byte_values_gpt4]("token_byte_values") \
            .def_py_method[_decode_single_token_bytes_gpt4]("decode_single_token_bytes") \
            .def_py_method[_decode_with_offsets_gpt4]("decode_with_offsets")

        _ = mb.add_type[GPT4oTK]("GPT4oTokenizer") \
            .def_py_init[_init_gpt4o]() \
            .def_py_method[_train_gpt4o]("train") \
            .def_py_method[_encode_gpt4o]("encode") \
            .def_py_method[_encode_ordinary_gpt4o]("encode_ordinary") \
            .def_py_method[_decode_gpt4o]("decode") \
            .def_py_method[_n_vocab_gpt4o]("n_vocab") \
            .def_py_method[_save_tiktok_gpt4o]("save_tiktoken") \
            .def_py_method[_load_tiktok_gpt4o]("load_tiktoken") \
            .def_py_method[_reg_special_gpt4o]("register_special_tokens") \
            .def_py_method[_get_specials_gpt4o]("_get_special_bytes") \
            .def_py_method[_set_specials_gpt4o]("_set_special_bytes") \
            .def_py_method[_name_gpt4o]("name") \
            .def_py_method[_decode_bytes_gpt4o]("decode_bytes") \
            .def_py_method[_encode_single_token_gpt4o]("encode_single_token") \
            .def_py_method[_token_byte_values_gpt4o]("token_byte_values") \
            .def_py_method[_decode_single_token_bytes_gpt4o]("decode_single_token_bytes") \
            .def_py_method[_decode_with_offsets_gpt4o]("decode_with_offsets")

        mb.def_function[py_get_encoding]("get_encoding")
        mb.def_function[py_train]("_train_impl")

        return mb.finalize()
    except:
        return PythonObject(None)
