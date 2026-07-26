use std::time::{SystemTime, UNIX_EPOCH};

fn fmt_ns(ns: u64) -> String {
    if ns < 1000 {
        format!("{} ns", ns)
    } else if ns < 1_000_000 {
        format!("{:.1} us", ns as f64 / 1000.0)
    } else if ns < 1_000_000_000 {
        format!("{:.1} ms", ns as f64 / 1_000_000.0)
    } else {
        format!("{:.2} s", ns as f64 / 1_000_000_000.0)
    }
}

fn fmt_tok_s(tokens: usize, ns: u64) -> String {
    if ns == 0 {
        return "N/A".into();
    }
    let per_sec = (tokens as u64) * 1_000_000_000 / ns;
    if per_sec >= 1_000_000 {
        format!("{:.1} M tok/s", per_sec as f64 / 1_000_000.0)
    } else if per_sec >= 1000 {
        format!("{:.1} K tok/s", per_sec as f64 / 1000.0)
    } else {
        format!("{} tok/s", per_sec)
    }
}

fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

fn main() {
    println!("{}", "=".repeat(60));
    println!("  tiktoken-rs (Rust)  — GPT-2 BPE encode/decode benchmark");
    println!("{}", "=".repeat(60));

    // Read corpus
    let corpus_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| {
            let mut p = std::env::current_dir().unwrap();
            p.push("benchmarks/corpus.txt");
            p.to_string_lossy().to_string()
        });
    let text = std::fs::read_to_string(&corpus_path)
        .expect("Failed to read corpus.txt");
    let n_bytes = text.len();
    println!("\nCorpus: {} bytes", n_bytes);

    // Load GPT-2 tokenizer
    let bpe = tiktoken_rs::p50k_base().unwrap();

    // First encode to get token count
    let t0 = now_ns();
    let tokens: Vec<u32> = bpe.encode_with_special_tokens(&text);
    let t1 = now_ns();
    let num_tokens = tokens.len();
    println!("  tokens: {}  (first encode: {})", num_tokens, fmt_ns(t1 - t0));

    // Encode benchmark
    println!("\n── encode ──");
    let n_iters = 20;
    let _ = bpe.encode_with_special_tokens(&text); // warmup
    let mut encode_times: Vec<u64> = Vec::with_capacity(n_iters);
    for _ in 0..n_iters {
        let t0 = now_ns();
        let _ = bpe.encode_with_special_tokens(&text);
        let t1 = now_ns();
        encode_times.push(t1 - t0);
    }

    let enc_best = *encode_times.iter().min().unwrap();
    println!("  best: {}  {}", fmt_ns(enc_best), fmt_tok_s(num_tokens, enc_best));

    // Decode benchmark
    println!("\n── decode ──");
    let _ = bpe.decode(&tokens); // warmup
    let mut decode_times: Vec<u64> = Vec::with_capacity(n_iters);
    for _ in 0..n_iters {
        let t0 = now_ns();
        let _ = bpe.decode(&tokens);
        let t1 = now_ns();
        decode_times.push(t1 - t0);
    }

    let dec_best = *decode_times.iter().min().unwrap();
    println!("  best: {}  {}", fmt_ns(dec_best), fmt_tok_s(num_tokens, dec_best));

    println!("\n{}", "=".repeat(60));
}
