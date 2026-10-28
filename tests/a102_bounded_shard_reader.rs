//! Bounded input-read contract for a compressed expert shard.
use safetensors::tensor::{serialize, Dtype, TensorView};
use std::io::{self, Cursor, Read};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use voltafabric::{deepseek_v41::Mxfp4ExpertShard, sha256};

fn tiny_shard() -> Vec<u8> {
    let codes = [0u8; 32 * 16];
    let scales = [127u8; 32];
    let mut tensors = Vec::with_capacity(6);
    for projection in ["w1", "w2", "w3"] {
        tensors.push((
            format!("layers.0.ffn.experts.0.{projection}.weight"),
            TensorView::new(Dtype::I8, vec![32, 16], &codes).unwrap(),
        ));
        tensors.push((
            format!("layers.0.ffn.experts.0.{projection}.scale"),
            TensorView::new(Dtype::F8_E8M0, vec![32, 1], &scales).unwrap(),
        ));
    }
    serialize(tensors, None).unwrap()
}

#[derive(Clone)]
struct CountingReader {
    inner: Cursor<Vec<u8>>,
    consumed: Arc<AtomicUsize>,
}

impl CountingReader {
    fn new(bytes: Vec<u8>) -> Self {
        Self {
            inner: Cursor::new(bytes),
            consumed: Arc::new(AtomicUsize::new(0)),
        }
    }
}

impl Read for CountingReader {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let read = self.inner.read(buffer)?;
        self.consumed.fetch_add(read, Ordering::Relaxed);
        Ok(read)
    }
}

struct FailingReader {
    calls: Arc<AtomicUsize>,
}

impl Read for FailingReader {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if self.calls.fetch_add(1, Ordering::Relaxed) == 0 {
            buffer[..4].copy_from_slice(b"data");
            Ok(4)
        } else {
            Err(io::Error::new(
                io::ErrorKind::ConnectionReset,
                "injected failure",
            ))
        }
    }
}

#[test]
fn reader_accepts_the_exact_byte_budget() {
    let bytes = tiny_shard();
    let digest = sha256::hex(&bytes);
    let shard =
        Mxfp4ExpertShard::from_bounded_reader(Cursor::new(bytes.clone()), &digest, bytes.len())
            .unwrap();
    assert_eq!(shard.backing_bytes(), bytes.len());
    assert_eq!(shard.expert_count(), 1);
    assert_eq!(shard.expert(0, 0).unwrap().compressed_bytes(), 3 * 544);
}

#[test]
fn reader_stops_after_the_single_over_budget_sentinel_byte() {
    let bytes = tiny_shard();
    let max_bytes = bytes.len();
    let mut over = bytes;
    over.push(0);
    let digest = sha256::hex(&over);
    let reader = CountingReader::new(over);
    let observed = Arc::clone(&reader.consumed);
    let error = Mxfp4ExpertShard::from_bounded_reader(reader, &digest, max_bytes).unwrap_err();
    assert!(error
        .message
        .contains("exceeds its caller-supplied byte budget"));
    assert_eq!(observed.load(Ordering::Relaxed), max_bytes + 1);
}

#[test]
fn reader_reports_truncation_and_io_failure_without_retry() {
    let mut truncated = tiny_shard();
    truncated.pop();
    let digest = sha256::hex(&truncated);
    assert!(Mxfp4ExpertShard::from_bounded_reader(
        Cursor::new(truncated.clone()),
        &digest,
        truncated.len(),
    )
    .is_err());

    let calls = Arc::new(AtomicUsize::new(0));
    let error = Mxfp4ExpertShard::from_bounded_reader(
        FailingReader {
            calls: Arc::clone(&calls),
        },
        &"0".repeat(64),
        1024,
    )
    .unwrap_err();
    assert_eq!(error.code, "SHARD_READ_ERROR");
    assert_eq!(calls.load(Ordering::Relaxed), 2);
}

#[test]
fn reader_refuses_an_overflowing_budget_before_reading() {
    let calls = Arc::new(AtomicUsize::new(0));
    let error = Mxfp4ExpertShard::from_bounded_reader(
        FailingReader {
            calls: Arc::clone(&calls),
        },
        &"0".repeat(64),
        usize::MAX,
    )
    .unwrap_err();
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert!(error.message.contains("byte budget overflow"));
}
