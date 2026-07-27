use std::time::{SystemTime, UNIX_EPOCH};

fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

fn bench_encoding(
    name: &str,
    bpe: &tiktoken_rs::CoreBPE,
    text: &str,
    n_bytes: usize,
    n_iters: u32,
) -> String {
    // First encode
    let t0 = now_ns();
    let tokens: Vec<u32> = bpe.encode_with_special_tokens(text);
    let t1 = now_ns();
    let num_tokens = tokens.len();
    let first_encode_ms = (t1 - t0) as f64 / 1_000_000.0;

    // Encode benchmark (best of n_iters)
    let _ = bpe.encode_with_special_tokens(text);
    let mut encode_times: Vec<u64> = Vec::with_capacity(n_iters as usize);
    for _ in 0..n_iters {
        let t0 = now_ns();
        let _ = bpe.encode_with_special_tokens(text);
        let t1 = now_ns();
        encode_times.push(t1 - t0);
    }
    let enc_best_ns = *encode_times.iter().min().unwrap();
    let enc_best_ms = enc_best_ns as f64 / 1_000_000.0;
    let enc_mtok_s = (num_tokens as f64) / (enc_best_ns as f64 / 1_000_000_000.0) / 1_000_000.0;

    // Decode benchmark (best of n_iters)
    let _ = bpe.decode(&tokens);
    let mut decode_times: Vec<u64> = Vec::with_capacity(n_iters as usize);
    for _ in 0..n_iters {
        let t0 = now_ns();
        let _ = bpe.decode(&tokens);
        let t1 = now_ns();
        decode_times.push(t1 - t0);
    }
    let dec_best_ns = *decode_times.iter().min().unwrap();
    let dec_best_ms = dec_best_ns as f64 / 1_000_000.0;
    let dec_mtok_s = (num_tokens as f64) / (dec_best_ns as f64 / 1_000_000_000.0) / 1_000_000.0;

    format!(
        r#"{{"impl":"tiktoken_rs","encoding":"{}","corpus_bytes":{},"n_tokens":{},"first_encode_ms":{:.2},"encode_ms":{:.2},"encode_mtok_s":{:.2},"decode_ms":{:.2},"decode_mtok_s":{:.2}}}"#,
        name, n_bytes, num_tokens, first_encode_ms, enc_best_ms, enc_mtok_s, dec_best_ms, dec_mtok_s
    )
}

fn main() {
    // Read corpus: BPE_CORPUS env var → CLI arg → default
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

    // p50k_base (GPT-2)
    let bpe_p50k = tiktoken_rs::p50k_base().unwrap();
    println!("{}", bench_encoding("gpt2", &bpe_p50k, &text, n_bytes, n_iters));

    // cl100k_base (GPT-4)
    let bpe_cl100k = tiktoken_rs::cl100k_base().unwrap();
    println!("{}", bench_encoding("cl100k", &bpe_cl100k, &text, n_bytes, n_iters));

    // o200k_base (GPT-4o)
    let bpe_o200k = tiktoken_rs::o200k_base().unwrap();
    println!("{}", bench_encoding("o200k", &bpe_o200k, &text, n_bytes, n_iters));
}
