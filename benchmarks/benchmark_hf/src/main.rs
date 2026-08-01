use std::str::FromStr;
use std::time::{SystemTime, UNIX_EPOCH};
use tokenizers::tokenizer::Tokenizer;

fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

fn bench_encoding(
    name: &str,
    tok: &Tokenizer,
    text: &str,
    n_bytes: usize,
    n_iters: u32,
) -> String {
    // First encode
    let t0 = now_ns();
    let encoding = tok.encode(text, false).unwrap();
    let t1 = now_ns();
    let num_tokens = encoding.get_ids().len();
    let first_encode_ms = (t1 - t0) as f64 / 1_000_000.0;
    let ids = encoding.get_ids().to_vec();

    // Encode benchmark (best of n_iters)
    let _ = tok.encode(text, false).unwrap();
    let mut encode_times: Vec<u64> = Vec::with_capacity(n_iters as usize);
    for _ in 0..n_iters {
        let t0 = now_ns();
        let _ = tok.encode(text, false).unwrap();
        let t1 = now_ns();
        encode_times.push(t1 - t0);
    }
    let enc_best_ns = *encode_times.iter().min().unwrap();
    let enc_best_ms = enc_best_ns as f64 / 1_000_000.0;
    let enc_mtok_s = (num_tokens as f64) / (enc_best_ns as f64 / 1_000_000_000.0) / 1_000_000.0;

    // Decode benchmark (best of n_iters)
    let _ = tok.decode(&ids, false).unwrap();
    let mut decode_times: Vec<u64> = Vec::with_capacity(n_iters as usize);
    for _ in 0..n_iters {
        let t0 = now_ns();
        let _ = tok.decode(&ids, false).unwrap();
        let t1 = now_ns();
        decode_times.push(t1 - t0);
    }
    let dec_best_ns = *decode_times.iter().min().unwrap();
    let dec_best_ms = dec_best_ns as f64 / 1_000_000.0;
    let dec_mtok_s = (num_tokens as f64) / (dec_best_ns as f64 / 1_000_000_000.0) / 1_000_000.0;

    format!(
        r#"{{"impl":"hf_tokenizers","encoding":"{}","corpus_bytes":{},"n_tokens":{},"first_encode_ms":{:.2},"encode_ms":{:.2},"encode_mtok_s":{:.2},"decode_ms":{:.2},"decode_mtok_s":{:.2}}}"#,
        name, n_bytes, num_tokens, first_encode_ms, enc_best_ms, enc_mtok_s, dec_best_ms, dec_mtok_s
    )
}

fn main() {
    let corpus_path = std::env::var("BPE_CORPUS").unwrap_or_else(|_| {
        std::env::args().nth(1).unwrap_or_else(|| {
            let mut p = std::env::current_dir().unwrap();
            p.push("benchmarks/corpus.txt");
            p.to_string_lossy().to_string()
        })
    });
    let text = std::fs::read_to_string(&corpus_path).expect("Failed to read corpus");
    let n_bytes = text.len();
    let n_iters = 3;

    for (name, path) in [
        ("gpt2", "benchmarks/hf_data/gpt2.json"),
        ("cl100k", "benchmarks/hf_data/cl100k.json"),
        ("o200k", "benchmarks/hf_data/o200k.json"),
    ] {
        let json = std::fs::read_to_string(path).expect("Failed to read tokenizer.json");
        let tok = Tokenizer::from_str(&json).expect("Failed to parse tokenizer.json");
        println!("{}", bench_encoding(name, &tok, &text, n_bytes, n_iters));
    }
}
