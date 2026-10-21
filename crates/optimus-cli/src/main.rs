use optimus_core::backend::{CpuReference, MatmulBackend};
use optimus_core::{memory::MemoryBudget, Tensor};
use std::time::Instant;

fn main() {
    let result = match std::env::args().nth(1).as_deref() {
        Some("doctor") => doctor(),
        Some("smoke") => smoke(),
        Some("bench") => bench(),
        _ => {
            eprintln!("Usage: optimus-cli <doctor|smoke|bench>");
            Err("invalid command".into())
        }
    };
    if let Err(e) = result {
        eprintln!("optimus: {e}");
        std::process::exit(1);
    }
}

fn doctor() -> Result<(), Box<dyn std::error::Error>> {
    println!("Optimus v0.1 scaffold");
    println!("CPU f32 reference operations: implemented");
    println!("Intel NPU: planned, not detected or implemented");
    println!("NVIDIA GPU: planned, not detected or implemented");
    println!("Model loading and token generation: not implemented");
    Ok(())
}

fn smoke() -> Result<(), Box<dyn std::error::Error>> {
    let a = Tensor::from_vec(vec![1, 2], vec![2., 3.])?;
    let b = Tensor::from_vec(vec![2, 1], vec![4., 5.])?;
    let mut budget = MemoryBudget::new(1024);
    budget.reserve((a.as_slice().len() + b.as_slice().len()) * 4)?;
    let result = CpuReference.matmul(&a, &b)?;
    if result.as_slice() != [23.] { return Err("incorrect output".into()); }
    println!("PASS: CPU matmul 2*4 + 3*5 = 23");
    println!("Logical reservations: {} bytes, not measured process RAM", budget.used_bytes());
    Ok(())
}

fn bench() -> Result<(), Box<dyn std::error::Error>> {
    const N: usize = 128;
    const ITERS: usize = 3;
    let a = Tensor::from_vec(vec![N, N], vec![0.25; N * N])?;
    let b = Tensor::from_vec(vec![N, N], vec![0.5; N * N])?;
    let begin = Instant::now();
    let mut checksum = 0.0f32;
    for _ in 0..ITERS {
        let out = CpuReference.matmul(&a, &b)?;
        checksum += std::hint::black_box(out.as_slice()[0]);
    }
    println!("CPU reference matmul {N}x{N} x{ITERS}: {:.3} ms; checksum {checksum}", begin.elapsed().as_secs_f64() * 1000.);
    println!("Not a model benchmark, and not evidence of a speedup over Ollama.");
    Ok(())
}
