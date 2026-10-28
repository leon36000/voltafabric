//! Actual R13 GPU executor of the existing bounded authenticated rank session.
mod identity;
mod resident;
use std::collections::BTreeMap;
use std::io::{self, Write};
use voltafabric::{
    session::{
        self,
        mxfp4::{Artifact, Executor, Producer},
    },
    Error, Result,
};

// Audit is an explicit fallible boundary. Diagnostic loss must never unwind owners.
fn write_audit(writer: &mut impl Write, value: &serde_json::Value) -> Result<()> {
    writeln!(writer, "{value}")
        .map_err(|error| Error::new("GPU_AUDIT_IO", format!("stderr audit write: {error}")))
}
fn audit_line(value: &serde_json::Value) -> Result<()> {
    write_audit(&mut io::stderr().lock(), value)
}
fn diagnostic(message: std::fmt::Arguments<'_>) {
    // The primary error is carried by the existing exit status; there is no second
    // diagnostic channel if stderr itself is lost. Never retry native cleanup.
    let _write_error = writeln!(io::stderr().lock(), "{message}");
}

struct Gpu {
    identity: identity::Identity,
    owners: BTreeMap<u32, resident::Owner>,
    binding: Option<session::mxfp4::Binding>,
    fatal: bool,
    audit: bool,
}
impl Gpu {
    fn status(&mut self, code: resident::Status, operation: &str) -> Error {
        // Only numerical refusal is a documented safe post-start recoverable result.
        // All other unexpected native statuses terminate this GPU worker.
        if code.0 != -9 {
            self.fatal = true;
        }
        Error::new(
            if code.0 == -9 {
                "GPU_NUMERIC"
            } else {
                "GPU_WORKER_FATAL"
            },
            format!(
                "{operation}: native status {}; no replay or CPU fallback",
                code.0
            ),
        )
    }
    fn check_identity(&mut self) -> Result<()> {
        if let Err(error) = self.identity.unchanged() {
            self.fatal = true;
            return Err(error);
        }
        if self.fatal {
            return Err(Error::new("GPU_WORKER_FATAL", "worker is terminal"));
        }
        Ok(())
    }
    fn close_staged(&mut self, staged: &mut BTreeMap<u32, resident::Owner>) -> Result<()> {
        for owner in staged.values_mut() {
            if let Err(code) = owner.close() {
                self.fatal = true;
                // No destructor may repeat cleanup or touch an unrelated context.
                std::mem::forget(std::mem::take(staged));
                return Err(Error::new(
                    "GPU_WORKER_FATAL",
                    format!("LOAD rollback status {}; stop process", code.0),
                ));
            }
        }
        staged.clear();
        Ok(())
    }
}
impl Executor for Gpu {
    fn producer(&self) -> Producer {
        Producer::GpuR13ResidentFp32V1 {
            library_sha256: self.identity.sha256.clone(),
        }
    }
    fn load(&mut self, artifact: &Artifact) -> Result<()> {
        self.check_identity()?;
        if !self.owners.is_empty() || self.binding.is_some() {
            return Err(Error::new("ARTIFACT_LOADED", "evict before native LOAD"));
        }
        // R13 qualified lab owner bound, not a limit on the GPU fleet or service.
        if artifact.binding().experts.len() > 4 {
            return Err(Error::new(
                "GPU_ADMISSION",
                "qualified native ABI admits at most four expert owners per worker",
            ));
        }
        for entry in &artifact.binding().experts {
            let e = artifact.expert(entry.expert)?;
            let (h, m) = (e.w1().cols(), e.w1().rows());
            if !(32..=5120).contains(&h)
                || !(32..=5120).contains(&m)
                || h % 32 != 0
                || m % 32 != 0
                || h * m * 3 > 48_000_000
            {
                return Err(Error::new(
                    "GPU_ADMISSION",
                    "shape exceeds the qualified R13 domain",
                ));
            }
        }
        // All shapes and planes are admitted by the immutable CPU Artifact first.
        // Native numeric admission remains authoritative before each create.
        let mut staged = BTreeMap::new();
        for entry in &artifact.binding().experts {
            let expert = artifact.expert(entry.expert)?;
            let planes = [
                expert.w1().compressed_planes(),
                expert.w3().compressed_planes(),
                expert.w2().compressed_planes(),
            ];
            let made = resident::Owner::new(
                expert.w1().cols() as u32,
                expert.w1().rows() as u32,
                [
                    planes[0].0,
                    planes[0].1,
                    planes[1].0,
                    planes[1].1,
                    planes[2].0,
                    planes[2].1,
                ],
            );
            match made {
                Ok(owner) => {
                    staged.insert(entry.expert, owner);
                }
                Err(code) => {
                    // Native pre-CUDA admission failures can safely refuse this LOAD.
                    // CUDA failure may have executed cleanup; never repeat it.
                    // create's NUMERIC is a synchronous HOST weight-admission refusal,
                    // before CUDA. It differs from an uncertain CUDA allocation error.
                    let known = matches!(code.0, -1 | -2 | -3 | -4 | -5 | -9 | -11);
                    if !known
                        || resident::lifecycle().1 != 0
                        || resident::first_cleanup_failure().is_some()
                    {
                        self.fatal = true;
                        std::mem::forget(staged);
                        return Err(self.status(code, "LOAD"));
                    }
                    let rollback = self.close_staged(&mut staged);
                    rollback?;
                    return Err(if known {
                        Error::new("GPU_ADMISSION", format!("native LOAD status {}", code.0))
                    } else {
                        self.status(code, "LOAD")
                    });
                }
            }
        }
        self.owners = staged;
        self.binding = Some(artifact.binding().clone());
        Ok(())
    }
    fn forward(&mut self, expert: u32, input: &[f32], coefficient: f32) -> Result<Vec<f32>> {
        self.check_identity()?;
        let owner = self
            .owners
            .get_mut(&expert)
            .ok_or_else(|| Error::new("GPU_ADMISSION", "expert is not resident"))?;
        let bits: Vec<_> = input.iter().map(|v| v.to_bits()).collect();
        match owner.execute(&bits, coefficient.to_bits()) {
            Ok(output) => Ok(output.into_iter().map(f32::from_bits).collect()),
            Err(code) => Err(self.status(code, "EXEC")),
        }
    }
    fn evict(&mut self) -> Result<()> {
        self.check_identity()?;
        let mut counters = Vec::new();
        for (&expert, owner) in &mut self.owners {
            if self.audit {
                match owner.stats() {
                    Ok(s)=>counters.push(serde_json::json!({"expert":expert,
                        "weight_upload_bytes":s.weight_upload_bytes,"weight_upload_copies":s.weight_upload_copies,
                        "resident_weight_bytes":s.resident_weight_bytes,"exec_calls":s.exec_calls,
                        "published":s.published,"rejected":s.rejected,"kernel_launches":s.kernel_launches,
                        "down_launches":s.down_launches,"device_allocations":s.device_allocations,
                        "input_h2d_bytes":s.input_h2d_bytes,"output_d2h_bytes":s.output_d2h_bytes,
                        "status_d2h_bytes":s.status_d2h_bytes,"scratch_device_bytes":s.scratch_device_bytes,
                        "poisoned":s.poisoned})),
                    Err(code)=>{self.fatal=true;return Err(Error::new("GPU_WORKER_FATAL",format!("stats status {}",code.0)));},
                }
            }
            if let Err(code) = owner.close() {
                self.fatal = true;
                return Err(Error::new(
                    "GPU_WORKER_FATAL",
                    format!("close status {}; no retry", code.0),
                ));
            }
        }
        self.owners.clear();
        let audit = (self.audit && self.binding.is_some()).then(|| {
            serde_json::json!({"gpu_audit":"evicted","producer":self.producer(),
                "binding":self.binding,"experts":counters,"lifecycle":resident::lifecycle(),
                "cuda_api_calls":resident::api_calls()})
        });
        self.binding = None;
        if resident::lifecycle() != (0, 0, 0) || resident::first_cleanup_failure().is_some() {
            self.fatal = true;
            return Err(Error::new(
                "GPU_WORKER_FATAL",
                "native cleanup counters are not clear",
            ));
        }
        if let Some(value) = audit {
            audit_line(&value)?;
        }
        Ok(())
    }
    fn fatal(&self) -> bool {
        self.fatal
    }
}
fn main() {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let audit = match args.as_slice() {
        [] => false,
        [flag] if flag == "--audit-counters" => true,
        _ => {
            diagnostic(format_args!(
                "USAGE: vf-mxfp4-gpu-worker [--audit-counters]"
            ));
            std::process::exit(2);
        }
    };
    let identity = match identity::Identity::resolve() {
        Ok(value) => value,
        Err(error) => {
            diagnostic(format_args!("{error}"));
            std::process::exit(2);
        }
    };
    if audit {
        if let Err(error) = audit_line(
            &serde_json::json!({"gpu_audit":"startup", "library_sha256":identity.sha256,
            "lifecycle":resident::lifecycle(),"cuda_api_calls":resident::api_calls()}),
        ) {
            diagnostic(format_args!("{error}"));
            std::process::exit(2);
        }
    }
    let mut gpu = Gpu {
        identity,
        owners: BTreeMap::new(),
        binding: None,
        fatal: false,
        audit,
    };
    let served = session::session_main_with_mxfp4_executor(&mut gpu);
    // Fatal CUDA/cleanup must end the process without another Drop cleanup attempt.
    if gpu.fatal {
        diagnostic(format_args!("GPU_WORKER_FATAL: stop process; no retry"));
        std::process::exit(2);
    }
    let cleanup = gpu.evict();
    if let Err(error) = served.and(cleanup) {
        diagnostic(format_args!("{error}"));
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Broken;
    impl Write for Broken {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            Err(io::ErrorKind::BrokenPipe.into())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    #[test]
    fn closed_pipe_is_typed_error() {
        assert_eq!(
            write_audit(&mut Broken, &serde_json::json!({"x":1}))
                .unwrap_err()
                .code,
            "GPU_AUDIT_IO"
        );
    }
    #[test]
    fn exact_line() {
        let mut b = Vec::new();
        write_audit(&mut b, &serde_json::json!({"x":1})).unwrap();
        assert_eq!(b, b"{\"x\":1}\n");
    }
    struct Short {
        prefix: Vec<u8>,
    }
    impl Write for Short {
        fn write(&mut self, b: &[u8]) -> io::Result<usize> {
            if self.prefix.len() == 2 {
                return Err(io::ErrorKind::BrokenPipe.into());
            }
            let n = b.len().min(2 - self.prefix.len());
            self.prefix.extend_from_slice(&b[..n]);
            Ok(n)
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    #[test]
    fn partial_write_never_attests_success() {
        let mut w = Short { prefix: vec![] };
        assert_eq!(
            write_audit(&mut w, &serde_json::json!({"x":1}))
                .unwrap_err()
                .code,
            "GPU_AUDIT_IO"
        );
        assert_eq!(w.prefix, b"{\"");
    }
    struct Interrupted {
        first: bool,
        bytes: Vec<u8>,
    }
    impl Write for Interrupted {
        fn write(&mut self, b: &[u8]) -> io::Result<usize> {
            if self.first {
                self.first = false;
                return Err(io::ErrorKind::Interrupted.into());
            }
            self.bytes.extend_from_slice(b);
            Ok(b.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    #[test]
    fn interrupted_then_complete() {
        let mut w = Interrupted {
            first: true,
            bytes: vec![],
        };
        write_audit(&mut w, &serde_json::json!({"x":1})).unwrap();
        assert_eq!(w.bytes, b"{\"x\":1}\n");
    }
}
