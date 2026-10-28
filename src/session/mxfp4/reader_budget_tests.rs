use super::*;
use std::io::{self, Cursor};
use std::sync::atomic::{AtomicUsize, Ordering};

const RAW: &[u8] = include_bytes!("../../../fixtures/preweighted_mxfp4_tiny.safetensors");

fn binding() -> Binding {
    Binding {
        artifact_sha256: crate::sha256::hex(RAW),
        model_revision: "r21-reader-budget".into(),
        rank: 2,
        layer: 0,
        epoch: 31,
        experts: vec![
            Expert {
                expert: 241,
                local: 9,
            },
            Expert {
                expert: 17,
                local: 7,
            },
            Expert {
                expert: 93,
                local: 1,
            },
        ],
    }
}

#[derive(Default)]
struct Counts {
    consumed: AtomicUsize,
    calls: AtomicUsize,
}

struct Counted {
    source: Cursor<&'static [u8]>,
    counts: Arc<Counts>,
    chunk: usize,
}

impl Read for Counted {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        self.counts.calls.fetch_add(1, Ordering::Relaxed);
        let requested = buffer.len().min(self.chunk);
        let n = self.source.read(&mut buffer[..requested])?;
        self.counts.consumed.fetch_add(n, Ordering::Relaxed);
        Ok(n)
    }
}

#[test]
fn client_reader_exact_domain_and_sentinel_counts() {
    let mut budgets: Vec<usize> = (0..=RAW.len() + 1).collect();
    budgets.extend([
        MAX_ARTIFACT_BYTES - 1,
        MAX_ARTIFACT_BYTES,
        MAX_ARTIFACT_BYTES + 1,
        usize::MAX,
    ]);
    for budget in budgets {
        let counts = Arc::new(Counts::default());
        let reader = Counted {
            source: Cursor::new(RAW),
            counts: Arc::clone(&counts),
            chunk: usize::MAX,
        };
        let result = Artifact::from_client_bounded_reader(reader, binding(), budget);
        if budget < RAW.len() {
            assert_eq!(
                result.err().expect("under-budget input was accepted").code,
                "ARTIFACT_BUDGET"
            );
            let expected = if budget == 0 { 0 } else { budget + 1 };
            assert_eq!(counts.consumed.load(Ordering::Relaxed), expected);
            if budget == 0 {
                assert_eq!(counts.calls.load(Ordering::Relaxed), 0);
            }
        } else {
            let artifact = result.expect("exact fit must succeed");
            assert_eq!(artifact.bytes(), RAW);
            assert_eq!(counts.consumed.load(Ordering::Relaxed), RAW.len());
        }
    }
}

#[test]
fn client_reader_preserves_short_reads_and_digest_failure() {
    let counts = Arc::new(Counts::default());
    let reader = Counted {
        source: Cursor::new(RAW),
        counts: Arc::clone(&counts),
        chunk: 7,
    };
    let artifact = Artifact::from_client_bounded_reader(reader, binding(), RAW.len()).unwrap();
    assert_eq!(artifact.bytes(), RAW);
    assert_eq!(counts.consumed.load(Ordering::Relaxed), RAW.len());
    assert!(counts.calls.load(Ordering::Relaxed) > RAW.len() / 7);
    let mut bad = binding();
    bad.artifact_sha256 = "0".repeat(64);
    let error = Artifact::from_client_bounded_reader(Cursor::new(RAW), bad, RAW.len() + 1)
        .err()
        .unwrap();
    let public_error = Artifact::from_bounded_reader(Cursor::new(RAW), {
        let mut expected = binding();
        expected.artifact_sha256 = "0".repeat(64);
        expected
    })
    .err()
    .unwrap();
    assert_eq!(error.code, public_error.code);
    assert_eq!(error.code, "ARTIFACT_SHA256_MISMATCH");
    assert!(!error.message.contains("byte budget"));
}

struct Broken {
    calls: Arc<AtomicUsize>,
}
impl Read for Broken {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if self.calls.fetch_add(1, Ordering::Relaxed) == 0 {
            buffer[..4].copy_from_slice(b"data");
            Ok(4)
        } else {
            Err(io::Error::new(
                io::ErrorKind::ConnectionReset,
                "r21 injected error",
            ))
        }
    }
}

#[test]
fn client_reader_does_not_relabel_or_retry_io_errors() {
    let calls = Arc::new(AtomicUsize::new(0));
    let error = Artifact::from_client_bounded_reader(
        Broken {
            calls: Arc::clone(&calls),
        },
        binding(),
        1024,
    )
    .err()
    .unwrap();
    assert_eq!(error.code, "SHARD_READ_ERROR");
    assert_eq!(calls.load(Ordering::Relaxed), 2);
}
