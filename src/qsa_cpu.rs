//! Short-context SC117 QSA CPU reference: all KV cells selected, FP16 cache.
//! Caller supplies already-normalized and RoPE-rotated Q/K and owns durable
//! reservation and whole-layer commit. This does not implement sparse selection.
use crate::{
    completion::{decode_fp16, encode_fp16},
    gdn_cpu::Identity,
    invalid, Result,
};
use std::sync::atomic::{AtomicU64, Ordering};
static NEXT_ID: AtomicU64 = AtomicU64::new(1);
pub const MAX_IDENTITY_SELECTION: usize = 2051;
#[derive(Clone)]
struct Cell {
    k: Vec<u16>,
    v: Vec<u16>,
}
pub struct State {
    id: u64,
    identity: Identity,
    q_heads: usize,
    kv_heads: usize,
    dim: usize,
    capacity: usize,
    cells: Vec<Cell>,
}
pub struct Proposal {
    owner: u64,
    position: usize,
    cell: Cell,
    output: Vec<f32>,
}
/// Provisional causal cache for a bounded group of positions. Outputs returned
/// by `prepare_next` are tentative; the enclosing layer must not publish them.
/// Dropping, failing or incompletely committing the transaction leaves the
/// original cache unchanged. Durable admission remains the caller's obligation.
pub struct Batch<'a> {
    original: &'a mut State,
    provisional: State,
    expected: usize,
    outputs: Vec<Vec<f32>>,
    failed: bool,
}
impl Proposal {
    pub fn output(&self) -> &[f32] {
        &self.output
    }
}
impl State {
    pub fn new(
        identity: Identity,
        q_heads: usize,
        kv_heads: usize,
        dim: usize,
        capacity: usize,
    ) -> Result<Self> {
        if identity.run_id.trim().is_empty()
            || identity.run_id.len() > 256
            || identity.model_revision.trim().is_empty()
            || identity.model_revision.len() > 256
            || q_heads == 0
            || q_heads > 128
            || kv_heads == 0
            || !q_heads.is_multiple_of(kv_heads)
            || dim == 0
            || dim > 512
            || capacity == 0
            || capacity > MAX_IDENTITY_SELECTION
            || kv_heads
                .checked_mul(dim)
                .and_then(|x| x.checked_mul(capacity))
                .is_none_or(|x| x > 16_777_216)
        {
            return Err(invalid("QSA identity or bounded geometry"));
        }
        let id = NEXT_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |x| x.checked_add(1))
            .map_err(|_| invalid("QSA state ID exhausted"))?;
        Ok(Self {
            id,
            identity,
            q_heads,
            kv_heads,
            dim,
            capacity,
            cells: Vec::new(),
        })
    }
    pub fn len(&self) -> usize {
        self.cells.len()
    }
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }
    /// Copy the committed cache under a fresh identity for a private transaction.
    pub(crate) fn fork(&self) -> Result<Self> {
        let mut provisional = State::new(
            self.identity.clone(),
            self.q_heads,
            self.kv_heads,
            self.dim,
            self.capacity,
        )?;
        provisional.cells = self.cells.clone();
        Ok(provisional)
    }
    pub(crate) fn batch_fits(&self, expected: usize) -> bool {
        expected > 0
            && expected <= 128
            && self
                .cells
                .len()
                .checked_add(expected)
                .is_some_and(|n| n <= self.capacity)
    }
    pub(crate) fn fork_batch(&self, expected: usize) -> Result<Self> {
        if !self.batch_fits(expected) {
            return Err(invalid("QSA batch size or capacity"));
        }
        self.fork()
    }
    /// Begin a transaction of at most 128 positions, using an independent
    /// cache identity so proposals cannot cross the provisional boundary.
    pub fn begin_batch(&mut self, expected: usize) -> Result<Batch<'_>> {
        let provisional = self.fork_batch(expected)?;
        Ok(Batch {
            original: self,
            provisional,
            expected,
            outputs: Vec::with_capacity(expected),
            failed: false,
        })
    }
    /// Computes a candidate without mutating cache. Q/K must already include the
    /// model's weighted RMS normalization and positional rotation.
    pub fn prepare(
        &self,
        identity: &Identity,
        position: usize,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        gate: &[f32],
    ) -> Result<Proposal> {
        let qn = self.q_heads * self.dim;
        let kn = self.kv_heads * self.dim;
        if identity != &self.identity
            || position != self.cells.len()
            || position >= self.capacity
            || q.len() != qn
            || gate.len() != qn
            || k.len() != kn
            || v.len() != kn
            || q.iter()
                .chain(k)
                .chain(v)
                .chain(gate)
                .any(|x| !x.is_finite())
        {
            return Err(invalid("QSA position, identity or input"));
        }
        let pack = |x: &[f32]| {
            x.iter()
                .map(|&v| encode_fp16(f64::from(v)))
                .collect::<Result<Vec<_>>>()
        };
        let cell = Cell {
            k: pack(k)?,
            v: pack(v)?,
        };
        let mut output = vec![0.0f32; qn];
        let ratio = self.q_heads / self.kv_heads;
        for h in 0..self.q_heads {
            let kh = h / ratio;
            let query = &q[h * self.dim..(h + 1) * self.dim];
            let mut scores = Vec::with_capacity(position + 1);
            for c in self.cells.iter().chain(std::iter::once(&cell)) {
                let mut sum = 0f64;
                for (d, &x) in query.iter().enumerate() {
                    sum += f64::from(x) * decode_fp16(c.k[kh * self.dim + d])?;
                }
                scores.push(sum / (self.dim as f64).sqrt());
            }
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<_> = scores.iter().map(|&x| (x - max).exp()).collect();
            let sum: f64 = weights.iter().sum();
            if !sum.is_finite() || sum <= 0.0 {
                return Err(invalid("QSA softmax overflow"));
            }
            for d in 0..self.dim {
                let mut x = 0f64;
                for (c, &w) in self
                    .cells
                    .iter()
                    .chain(std::iter::once(&cell))
                    .zip(&weights)
                {
                    x += (w / sum) * decode_fp16(c.v[kh * self.dim + d])?;
                }
                let g = f64::from(gate[h * self.dim + d]);
                let sigmoid = if g >= 0.0 {
                    1.0 / (1.0 + (-g).exp())
                } else {
                    let e = g.exp();
                    e / (1.0 + e)
                };
                let y = (x * sigmoid) as f32;
                if !y.is_finite() {
                    return Err(invalid("QSA output overflow"));
                }
                output[h * self.dim + d] = y;
            }
        }
        Ok(Proposal {
            owner: self.id,
            position,
            cell,
            output,
        })
    }
    /// Caller must complete the enclosing layer before committing this candidate.
    pub fn commit(&mut self, proposal: Proposal) -> Result<Vec<f32>> {
        if proposal.owner != self.id
            || proposal.position != self.cells.len()
            || self.cells.len() >= self.capacity
        {
            return Err(invalid("QSA stale candidate"));
        }
        self.cells.push(proposal.cell);
        Ok(proposal.output)
    }
}

impl Batch<'_> {
    /// Prepare the next causal position. The returned output is tentative.
    pub fn prepare_next(
        &mut self,
        identity: &Identity,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        gate: &[f32],
    ) -> Result<&[f32]> {
        if self.failed || self.outputs.len() >= self.expected {
            self.failed = true;
            return Err(invalid("QSA batch terminal or complete"));
        }
        let result = self
            .provisional
            .prepare(identity, self.provisional.len(), q, k, v, gate)
            .and_then(|proposal| self.provisional.commit(proposal));
        match result {
            Ok(output) => {
                self.outputs.push(output);
                Ok(self.outputs.last().expect("just pushed"))
            }
            Err(error) => {
                self.failed = true;
                Err(error)
            }
        }
    }

    /// Publish the cache only after every planned position is prepared and the
    /// enclosing layer has validated all other work.
    pub fn commit(self) -> Result<Vec<Vec<f32>>> {
        if self.failed || self.outputs.len() != self.expected {
            return Err(invalid("QSA batch incomplete or aborted"));
        }
        self.original.cells = self.provisional.cells;
        Ok(self.outputs)
    }
}
