//! Bounded CPU owner-row contract for immutable Engram shards and persistent workers.
//! Transport authentication is provided by the existing SSH session; hashes detect
//! corruption and identity mixups, not malicious peers. This is not a CUDA backend.
use crate::sha256;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

const ROW_BYTES: usize = 264;
// Keep one binary owner reply within the existing 16 MiB authenticated session frame.
pub const MAX_LOOKUPS: usize = 60_000;
const MAX_PAYLOAD_BYTES: usize = ROW_BYTES * MAX_LOOKUPS;
const MAX_WORLD_SIZE: u32 = 4_096;
const WIRE_MAGIC: &[u8; 8] = b"VFENGR01";
const MAX_IDENTITY_BYTES: usize = 1_024;
const MAX_HEADER_BYTES: usize =
    8 + 4 * (4 + MAX_IDENTITY_BYTES) + 20 + 2 * 64 + 20 + 28 + 2 * 64 + 8;
const MAX_FRAME_BYTES: usize = MAX_PAYLOAD_BYTES + MAX_HEADER_BYTES;

pub type Stamp = crate::moe::Stamp;

#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TableIdentity {
    pub model_revision: String,
    pub table_sha256: String,
    pub tokenizer_map_sha256: String,
    pub layer: u32,
    pub epoch: u64,
    pub num_embeddings: u64,
}

/// Bounded FIFO cache of immutable 264-byte rows. The key contains the full table identity,
/// including content digests, revision, layer, epoch and table size.
#[derive(Debug)]
pub struct EngramRowCache {
    capacity_bytes: usize,
    resident_bytes: usize,
    tables: HashMap<Arc<TableIdentity>, usize>,
    identity_slots: Vec<Option<Arc<TableIdentity>>>,
    identity_rows: Vec<usize>,
    free_identity_slots: Vec<usize>,
    rows: HashMap<(usize, u64), usize>,
    row_slots: Vec<Option<CachedRow>>,
    free_row_slots: Vec<usize>,
    insertion_order: VecDeque<usize>,
}

#[derive(Debug)]
struct CachedRow {
    key: (usize, u64),
    bytes: [u8; ROW_BYTES],
}

impl EngramRowCache {
    pub fn new(capacity_bytes: usize) -> Self {
        Self {
            capacity_bytes,
            resident_bytes: 0,
            tables: HashMap::new(),
            identity_slots: Vec::new(),
            identity_rows: Vec::new(),
            free_identity_slots: Vec::new(),
            rows: HashMap::new(),
            row_slots: Vec::new(),
            free_row_slots: Vec::new(),
            insertion_order: VecDeque::new(),
        }
    }
    pub fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }
    pub fn len(&self) -> usize {
        self.rows.len()
    }
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
    pub fn resident_table_count(&self) -> usize {
        self.tables.len()
    }
    pub fn get(&self, table: &TableIdentity, row_id: u64) -> Option<&[u8]> {
        let table_id = *self.tables.get(table)?;
        self.get_by_table_id(table_id, row_id)
    }
    fn get_by_table_id(&self, table_id: usize, row_id: u64) -> Option<&[u8]> {
        let slot = *self.rows.get(&(table_id, row_id))?;
        self.row_slots
            .get(slot)?
            .as_ref()
            .map(|row| row.bytes.as_slice())
    }
    /// Resolve the full table identity once for a batch and visit rows by compact interned ID
    /// without allocating a result vector. Row references remain tied to this cache borrow.
    pub fn for_each_lookup<E>(
        &self,
        table: &TableIdentity,
        row_ids: &[u64],
        mut visit: impl FnMut(u64, Option<&[u8]>) -> std::result::Result<(), E>,
    ) -> std::result::Result<(), E> {
        let table_id = self.tables.get(table).copied();
        for &row_id in row_ids {
            visit(
                row_id,
                table_id.and_then(|table_id| self.get_by_table_id(table_id, row_id)),
            )?;
        }
        Ok(())
    }
    /// Cache insertion is best-effort when capacity is smaller than one row. Call only after
    /// the full response has been validated; this method itself verifies identity and length.
    pub fn insert(&mut self, table: &TableIdentity, row_id: u64, bytes: &[u8]) -> Result<bool> {
        validate_table(table)?;
        if row_id >= table.num_embeddings || bytes.len() != ROW_BYTES {
            return Err(ExchangeError("invalid cache row identity or length"));
        }
        if self.capacity_bytes < ROW_BYTES {
            return Ok(false);
        }
        let row: [u8; ROW_BYTES] = bytes
            .try_into()
            .map_err(|_| ExchangeError("invalid cache row length"))?;
        let mut table_id_hint = self.tables.get(table).copied();
        self.insert_validated_row(table, row_id, row, &mut table_id_hint)
    }

    /// Insert contiguous rows after validating the complete batch. The row IDs and payload
    /// must match exactly; invalid inputs leave the cache untouched. This is called only after
    /// the complete remote response has been validated, so no uncertain worker result is cached.
    pub fn insert_many(
        &mut self,
        table: &TableIdentity,
        row_ids: &[u64],
        payload: &[u8],
    ) -> Result<()> {
        validate_table(table)?;
        let expected_len = row_ids
            .len()
            .checked_mul(ROW_BYTES)
            .ok_or(ExchangeError("cache batch byte count overflow"))?;
        if payload.len() != expected_len {
            return Err(ExchangeError("invalid cache batch payload length"));
        }
        if row_ids.iter().any(|&row_id| row_id >= table.num_embeddings) {
            return Err(ExchangeError("invalid cache row identity or length"));
        }
        if row_ids.is_empty() || self.capacity_bytes < ROW_BYTES {
            return Ok(());
        }

        let mut table_id_hint = self.tables.get(table).copied();
        for (index, &row_id) in row_ids.iter().enumerate() {
            let start = index * ROW_BYTES;
            let row: [u8; ROW_BYTES] = payload[start..start + ROW_BYTES]
                .try_into()
                .map_err(|_| ExchangeError("invalid cache batch row length"))?;
            self.insert_validated_row(table, row_id, row, &mut table_id_hint)?;
        }
        Ok(())
    }

    fn insert_validated_row(
        &mut self,
        table: &TableIdentity,
        row_id: u64,
        row: [u8; ROW_BYTES],
        table_id_hint: &mut Option<usize>,
    ) -> Result<bool> {
        if self.capacity_bytes < ROW_BYTES {
            return Ok(false);
        }
        let existing_table_id = *table_id_hint;
        if existing_table_id.is_some_and(|id| self.rows.contains_key(&(id, row_id))) {
            return Ok(true);
        }
        self.rows
            .try_reserve(1)
            .map_err(|_| ExchangeError("cache allocation failed"))?;
        self.insertion_order
            .try_reserve(1)
            .map_err(|_| ExchangeError("cache allocation failed"))?;
        self.free_row_slots
            .try_reserve(1)
            .map_err(|_| ExchangeError("cache allocation failed"))?;
        let will_evict = self.resident_bytes > self.capacity_bytes - ROW_BYTES;
        if self.free_row_slots.is_empty() && !will_evict {
            self.row_slots
                .try_reserve(1)
                .map_err(|_| ExchangeError("cache allocation failed"))?;
        }
        self.free_identity_slots
            .try_reserve(1)
            .map_err(|_| ExchangeError("cache allocation failed"))?;
        // An identity missing before eviction needs capacity reserved in advance. If eviction
        // removes the current identity, it supplies a reusable identity slot itself.
        if existing_table_id.is_none() {
            self.tables
                .try_reserve(1)
                .map_err(|_| ExchangeError("cache allocation failed"))?;
            if self.free_identity_slots.is_empty() {
                self.identity_slots
                    .try_reserve(1)
                    .map_err(|_| ExchangeError("cache allocation failed"))?;
                self.identity_rows
                    .try_reserve(1)
                    .map_err(|_| ExchangeError("cache allocation failed"))?;
            }
        }
        while self.resident_bytes > self.capacity_bytes - ROW_BYTES {
            let slot = self
                .insertion_order
                .pop_front()
                .ok_or(ExchangeError("cache accounting invariant failed"))?;
            let old_row = self
                .row_slots
                .get_mut(slot)
                .and_then(Option::take)
                .ok_or(ExchangeError("cache row slot accounting invariant failed"))?;
            if self.rows.remove(&old_row.key) != Some(slot) {
                return Err(ExchangeError("cache row index invariant failed"));
            }
            self.resident_bytes -= ROW_BYTES;
            self.identity_rows[old_row.key.0] -= 1;
            if self.identity_rows[old_row.key.0] == 0 {
                let old_key = self.identity_slots[old_row.key.0]
                    .take()
                    .ok_or(ExchangeError("cache identity accounting invariant failed"))?;
                self.tables.remove(&old_key);
                self.free_identity_slots.push(old_row.key.0);
            }
            self.free_row_slots.push(slot);
        }
        let table_id = match existing_table_id
            .filter(|&id| self.identity_slots.get(id).is_some_and(Option::is_some))
        {
            Some(id) => id,
            None => match self.tables.get(table).copied() {
                Some(id) => id,
                None => {
                    let id = self
                        .free_identity_slots
                        .pop()
                        .unwrap_or(self.identity_slots.len());
                    if id == self.identity_slots.len() {
                        self.identity_slots.push(None);
                        self.identity_rows.push(0);
                    }
                    let key = Arc::new(table.clone());
                    self.tables.insert(Arc::clone(&key), id);
                    self.identity_slots[id] = Some(key);
                    self.identity_rows[id] = 0;
                    id
                }
            },
        };
        debug_assert!(self.identity_slots[table_id]
            .as_ref()
            .is_some_and(|identity| identity.as_ref() == table));
        *table_id_hint = Some(table_id);
        let slot = self.free_row_slots.pop().unwrap_or_else(|| {
            self.row_slots.push(None);
            self.row_slots.len() - 1
        });
        self.rows.insert((table_id, row_id), slot);
        self.row_slots[slot] = Some(CachedRow {
            key: (table_id, row_id),
            bytes: row,
        });
        self.insertion_order.push_back(slot);
        self.identity_rows[table_id] += 1;
        self.resident_bytes += ROW_BYTES;
        Ok(true)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OwnerRows {
    pub stamp: Stamp,
    pub table: TableIdentity,
    pub sender_rank: u32,
    pub world_size: u32,
    pub first_row: u64,
    pub shard_rows: u64,
    pub row_count: u32,
    pub request_row_ids_sha256: String,
    pub payload_sha256: String,
    pub payload: Vec<u8>,
}

/// Immutable request binding built once per batch. Reusing it avoids re-hashing every row ID
/// and rescanning the whole request once for each worker response.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RequestBinding {
    stamp: Stamp,
    table: TableIdentity,
    world_size: u32,
    row_ids_sha256: String,
    owner_row_counts: Vec<u32>,
}

impl RequestBinding {
    pub fn new(
        stamp: &Stamp,
        table: &TableIdentity,
        world_size: u32,
        row_ids: &[u64],
    ) -> Result<Self> {
        validate_identity(stamp, table)?;
        checked_payload_len(row_ids.len())?;
        if world_size == 0 || world_size > MAX_WORLD_SIZE {
            return Err(ExchangeError("invalid rank or world size"));
        }
        let mut owner_row_counts = vec![0u32; world_size as usize];
        for &row_id in row_ids {
            let owner = owner_rank(table, world_size, row_id)? as usize;
            owner_row_counts[owner] = owner_row_counts[owner]
                .checked_add(1)
                .ok_or(ExchangeError("owner row count overflow"))?;
        }
        Ok(Self {
            stamp: stamp.clone(),
            table: table.clone(),
            world_size,
            row_ids_sha256: row_ids_sha256(row_ids),
            owner_row_counts,
        })
    }

    fn expected_rows(&self, rank: u32) -> Result<u32> {
        if rank >= self.world_size {
            return Err(ExchangeError("invalid rank or world size"));
        }
        Ok(self.owner_row_counts[rank as usize])
    }
}

/// Owner rows whose frame digest and exact request binding were validated during decode.
/// The contained bytes cannot be mutated through this type. The digest checks integrity only;
/// transport authentication still belongs to the enclosing SSH session.
pub struct VerifiedOwnerRows {
    rows: OwnerRows,
    request: RequestBinding,
    expected_sender_rank: u32,
}

impl VerifiedOwnerRows {
    pub fn as_owner_rows(&self) -> &OwnerRows {
        &self.rows
    }

    pub fn into_owner_rows(self) -> OwnerRows {
        self.rows
    }
}

/// An authenticated-session response frame retained without copying its payload out of the
/// frame allocation. Construction verifies the frame digest and request identity; the payload
/// remains a private slice until reconstruction or explicit conversion to mutable `OwnerRows`.
pub struct VerifiedOwnerFrame {
    rows: OwnerRows,
    frame: Vec<u8>,
    payload_range: std::ops::Range<usize>,
    request: RequestBinding,
    expected_sender_rank: u32,
}

impl VerifiedOwnerFrame {
    fn payload(&self) -> &[u8] {
        &self.frame[self.payload_range.clone()]
    }

    pub fn payload_len(&self) -> usize {
        self.payload_range.len()
    }

    pub fn into_owner_rows(self) -> OwnerRows {
        let payload = self.payload().to_vec();
        let mut rows = self.rows;
        rows.payload = payload;
        rows
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExchangeError(pub &'static str);

pub type Result<T> = std::result::Result<T, ExchangeError>;

/// Owned, immutable bytes for one rank's contiguous native Engram shard.
#[derive(Clone, Debug)]
pub struct EngramShard {
    table: TableIdentity,
    world_size: u32,
    rank: u32,
    bytes: Box<[u8]>,
    sha256: String,
}

impl EngramShard {
    pub fn new(table: TableIdentity, world_size: u32, rank: u32, bytes: Vec<u8>) -> Result<Self> {
        validate_table(&table)?;
        let (first, end) = shard_range(&table, world_size, rank)?;
        let expected = usize::try_from(end - first)
            .ok()
            .and_then(|rows| rows.checked_mul(ROW_BYTES))
            .ok_or(ExchangeError("shard byte length overflow"))?;
        if bytes.len() != expected || bytes.len() > MAX_PAYLOAD_BYTES {
            return Err(ExchangeError("local shard length mismatch"));
        }
        let sha256 = artifact_sha256(&table, world_size, rank, &bytes)?;
        Ok(Self {
            table,
            world_size,
            rank,
            bytes: bytes.into_boxed_slice(),
            sha256,
        })
    }

    pub fn table(&self) -> &TableIdentity {
        &self.table
    }
    pub fn world_size(&self) -> u32 {
        self.world_size
    }
    pub fn rank(&self) -> u32 {
        self.rank
    }
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    pub fn rows(&self, stamp: &Stamp, row_ids: &[u64]) -> Result<OwnerRows> {
        pack_owner_rows(
            stamp,
            &self.table,
            self.world_size,
            self.rank,
            row_ids,
            &self.bytes,
        )
    }

    /// Pack and encode rows directly from this immutable shard. The packer has just computed
    /// the payload digest, so this trusted path avoids hashing the same bytes a second time.
    pub fn encode_rows_frame(
        &self,
        stamp: &Stamp,
        row_ids: &[u64],
        request: &RequestBinding,
        expected_sender_rank: u32,
    ) -> Result<Vec<u8>> {
        let rows = self.rows(stamp, row_ids)?;
        validate_owner_frame_common(&rows, request, expected_sender_rank)?;
        encode_owner_frame_payload(&rows)
    }
}

/// Digest the exact resident identity and shard bytes, with unambiguous little-endian
/// lengths. Metadata carried in the JSON header is therefore covered by the artifact ID.
pub fn artifact_sha256(
    table: &TableIdentity,
    world_size: u32,
    rank: u32,
    bytes: &[u8],
) -> Result<String> {
    validate_table(table)?;
    if world_size == 0 || world_size > MAX_WORLD_SIZE || rank >= world_size {
        return Err(ExchangeError("invalid rank or world size"));
    }
    let revision_len = u32::try_from(table.model_revision.len())
        .map_err(|_| ExchangeError("identity length overflow"))?;
    let mut canonical = Vec::new();
    let header_len = 4usize
        .checked_add(table.model_revision.len())
        .and_then(|n| n.checked_add(64 + 64 + 4 + 8 + 8 + 4 + 4 + 8))
        .ok_or(ExchangeError("artifact identity length overflow"))?;
    canonical
        .try_reserve_exact(header_len.saturating_add(bytes.len()))
        .map_err(|_| ExchangeError("artifact digest allocation failed"))?;
    canonical.extend_from_slice(&revision_len.to_le_bytes());
    canonical.extend_from_slice(table.model_revision.as_bytes());
    canonical.extend_from_slice(table.table_sha256.as_bytes());
    canonical.extend_from_slice(table.tokenizer_map_sha256.as_bytes());
    canonical.extend_from_slice(&table.layer.to_le_bytes());
    canonical.extend_from_slice(&table.epoch.to_le_bytes());
    canonical.extend_from_slice(&table.num_embeddings.to_le_bytes());
    canonical.extend_from_slice(&world_size.to_le_bytes());
    canonical.extend_from_slice(&rank.to_le_bytes());
    canonical.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    canonical.extend_from_slice(bytes);
    Ok(sha256::hex(&canonical))
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_text(out: &mut Vec<u8>, value: &str) -> Result<()> {
    if value.is_empty() || value.len() > MAX_IDENTITY_BYTES {
        return Err(ExchangeError("identity string outside wire limit"));
    }
    push_u32(
        out,
        u32::try_from(value.len()).map_err(|_| ExchangeError("identity length overflow"))?,
    );
    out.extend_from_slice(value.as_bytes());
    Ok(())
}

/// Serialize one owner response. The outer SSH/session transport must authenticate the peer;
/// these unkeyed hashes detect corruption and identity mixups, not a malicious sender.
pub fn encode_owner_frame(
    frame: &OwnerRows,
    request: &RequestBinding,
    expected_sender_rank: u32,
) -> Result<Vec<u8>> {
    validate_owner_frame_common(frame, request, expected_sender_rank)?;
    if sha256::hex(&frame.payload) != frame.payload_sha256 {
        return Err(ExchangeError("owner payload digest mismatch"));
    }
    encode_owner_frame_payload(frame)
}

/// Validate identity and bounds before hashing or serializing a caller-owned payload.
fn validate_owner_frame_common(
    frame: &OwnerRows,
    request: &RequestBinding,
    expected_sender_rank: u32,
) -> Result<()> {
    validate_identity(&frame.stamp, &frame.table)?;
    let expected_rows = request.expected_rows(expected_sender_rank)?;
    if frame.stamp != request.stamp
        || frame.table != request.table
        || frame.world_size != request.world_size
        || frame.sender_rank != expected_sender_rank
    {
        return Err(ExchangeError("owner frame differs from requested identity"));
    }
    if frame.request_row_ids_sha256 != request.row_ids_sha256 {
        return Err(ExchangeError("requested row ID digest mismatch"));
    }
    if frame.row_count != expected_rows {
        return Err(ExchangeError("owner row count differs from request"));
    }
    validate_digest(&frame.request_row_ids_sha256)?;
    validate_digest(&frame.payload_sha256)?;
    if frame.table.num_embeddings == 0
        || frame.world_size == 0
        || frame.world_size > MAX_WORLD_SIZE
        || frame.sender_rank >= frame.world_size
        || frame.row_count as usize > MAX_LOOKUPS
        || frame.payload.len() != checked_payload_len(frame.row_count as usize)?
    {
        return Err(ExchangeError("owner frame dimensions outside wire limits"));
    }
    let (first_row, end_row) = shard_range(&frame.table, frame.world_size, frame.sender_rank)?;
    if frame.first_row != first_row || frame.shard_rows != end_row - first_row {
        return Err(ExchangeError("owner shard range mismatch"));
    }
    if frame.shard_rows == 0 && frame.row_count != 0 {
        return Err(ExchangeError("empty shard cannot contain owner rows"));
    }
    Ok(())
}

/// Serialize a frame whose identity, dimensions, and payload digest are already validated.
fn encode_owner_frame_payload(frame: &OwnerRows) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(frame.payload.len().saturating_add(512));
    out.extend_from_slice(WIRE_MAGIC);
    push_text(&mut out, &frame.stamp.run_id)?;
    push_text(&mut out, &frame.stamp.model_revision)?;
    push_text(&mut out, &frame.stamp.plan_id)?;
    push_u32(&mut out, frame.stamp.layer);
    push_u64(&mut out, frame.stamp.batch);
    push_u64(&mut out, frame.stamp.epoch);
    push_text(&mut out, &frame.table.model_revision)?;
    out.extend_from_slice(frame.table.table_sha256.as_bytes());
    out.extend_from_slice(frame.table.tokenizer_map_sha256.as_bytes());
    push_u32(&mut out, frame.table.layer);
    push_u64(&mut out, frame.table.epoch);
    push_u64(&mut out, frame.table.num_embeddings);
    push_u32(&mut out, frame.sender_rank);
    push_u32(&mut out, frame.world_size);
    push_u64(&mut out, frame.first_row);
    push_u64(&mut out, frame.shard_rows);
    push_u32(&mut out, frame.row_count);
    out.extend_from_slice(frame.request_row_ids_sha256.as_bytes());
    out.extend_from_slice(frame.payload_sha256.as_bytes());
    push_u64(
        &mut out,
        u64::try_from(frame.payload.len()).map_err(|_| ExchangeError("payload length overflow"))?,
    );
    out.extend_from_slice(&frame.payload);
    if out.len() > MAX_FRAME_BYTES {
        return Err(ExchangeError("encoded frame exceeds frame limit"));
    }
    Ok(out)
}

struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or(ExchangeError("wire offset overflow"))?;
        let part = self
            .bytes
            .get(self.offset..end)
            .ok_or(ExchangeError("truncated owner frame"))?;
        self.offset = end;
        Ok(part)
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(
            self.take(4)?
                .try_into()
                .map_err(|_| ExchangeError("truncated u32"))?,
        ))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(
            self.take(8)?
                .try_into()
                .map_err(|_| ExchangeError("truncated u64"))?,
        ))
    }

    fn text(&mut self) -> Result<String> {
        let len = self.u32()? as usize;
        if len == 0 || len > MAX_IDENTITY_BYTES {
            return Err(ExchangeError("identity string outside wire limit"));
        }
        String::from_utf8(self.take(len)?.to_vec())
            .map_err(|_| ExchangeError("identity string is not UTF-8"))
    }

    fn digest(&mut self) -> Result<String> {
        let value = String::from_utf8(self.take(64)?.to_vec())
            .map_err(|_| ExchangeError("digest is not ASCII"))?;
        validate_digest(&value)?;
        Ok(value)
    }
}

/// Parse a complete response against the exact request before copying its payload. This codec
/// alone does not authenticate remote workers.
pub fn decode_owner_frame(
    bytes: &[u8],
    request: &RequestBinding,
    expected_sender_rank: u32,
) -> Result<OwnerRows> {
    decode_owner_frame_inner(bytes, request, expected_sender_rank)
}

/// Decode a response while retaining its immutable proof of integrity and request identity.
/// Passing this result to `reconstruct_partitioned_rows_verified` avoids hashing the payload a
/// second time. Callers that extract mutable `OwnerRows` must use the defensive public path.
pub fn decode_owner_frame_verified(
    bytes: &[u8],
    request: &RequestBinding,
    expected_sender_rank: u32,
) -> Result<VerifiedOwnerRows> {
    let rows = decode_owner_frame_inner(bytes, request, expected_sender_rank)?;
    Ok(VerifiedOwnerRows {
        rows,
        request: request.clone(),
        expected_sender_rank,
    })
}

/// Decode a response received as an owned session frame and retain that allocation as the
/// immutable payload backing store. The caller must already have authenticated the transport;
/// this function validates framing, digest and request identity only.
pub fn decode_owner_frame_verified_owned(
    frame: Vec<u8>,
    request: &RequestBinding,
    expected_sender_rank: u32,
) -> Result<VerifiedOwnerFrame> {
    let (rows, payload_range) = decode_owner_frame_parts(&frame, request, expected_sender_rank)?;
    Ok(VerifiedOwnerFrame {
        rows,
        frame,
        payload_range,
        request: request.clone(),
        expected_sender_rank,
    })
}

fn decode_owner_frame_inner(
    bytes: &[u8],
    request: &RequestBinding,
    expected_sender_rank: u32,
) -> Result<OwnerRows> {
    let (mut rows, payload_range) = decode_owner_frame_parts(bytes, request, expected_sender_rank)?;
    rows.payload = bytes[payload_range].to_vec();
    Ok(rows)
}

fn decode_owner_frame_parts(
    bytes: &[u8],
    request: &RequestBinding,
    expected_sender_rank: u32,
) -> Result<(OwnerRows, std::ops::Range<usize>)> {
    if bytes.len() > MAX_FRAME_BYTES {
        return Err(ExchangeError("encoded frame exceeds frame limit"));
    }
    let mut c = Cursor { bytes, offset: 0 };
    if c.take(WIRE_MAGIC.len())? != WIRE_MAGIC {
        return Err(ExchangeError("owner frame magic or version mismatch"));
    }
    let stamp = Stamp {
        run_id: c.text()?,
        model_revision: c.text()?,
        plan_id: c.text()?,
        layer: c.u32()?,
        batch: c.u64()?,
        epoch: c.u64()?,
    };
    let model_revision = c.text()?;
    let table_sha256 = c.digest()?;
    let tokenizer_map_sha256 = c.digest()?;
    let table = TableIdentity {
        model_revision,
        table_sha256,
        tokenizer_map_sha256,
        layer: c.u32()?,
        epoch: c.u64()?,
        num_embeddings: c.u64()?,
    };
    let sender_rank = c.u32()?;
    let world_size = c.u32()?;
    let first_row = c.u64()?;
    let shard_rows = c.u64()?;
    let row_count = c.u32()?;
    let request_row_ids_sha256 = c.digest()?;
    let payload_sha256 = c.digest()?;
    let payload_len =
        usize::try_from(c.u64()?).map_err(|_| ExchangeError("payload length overflow"))?;
    let expected_len = checked_payload_len(row_count as usize)?;
    if payload_len != expected_len {
        return Err(ExchangeError("owner frame payload length mismatch"));
    }
    validate_identity(&stamp, &table)?;
    if table.num_embeddings == 0 {
        return Err(ExchangeError("empty table is outside the Engram domain"));
    }
    if world_size == 0 || world_size > MAX_WORLD_SIZE || sender_rank >= world_size {
        return Err(ExchangeError("owner frame dimensions outside wire limits"));
    }
    let expected_rows = request.expected_rows(expected_sender_rank)?;
    if stamp != request.stamp
        || table != request.table
        || world_size != request.world_size
        || sender_rank != expected_sender_rank
    {
        return Err(ExchangeError("owner frame differs from requested identity"));
    }
    if request_row_ids_sha256 != request.row_ids_sha256 {
        return Err(ExchangeError("requested row ID digest mismatch"));
    }
    if row_count != expected_rows {
        return Err(ExchangeError("owner row count differs from request"));
    }
    let (expected_first, expected_end) = shard_range(&table, world_size, sender_rank)?;
    if first_row != expected_first || shard_rows != expected_end - expected_first {
        return Err(ExchangeError("owner shard range mismatch"));
    }
    let payload_start = c.offset;
    let payload_slice = c.take(payload_len)?;
    let payload_end = c.offset;
    if c.offset != bytes.len() {
        return Err(ExchangeError("trailing bytes after owner frame"));
    }
    if sha256::hex(payload_slice) != payload_sha256 {
        return Err(ExchangeError("owner payload digest mismatch"));
    }
    Ok((
        OwnerRows {
            stamp,
            table,
            sender_rank,
            world_size,
            first_row,
            shard_rows,
            row_count,
            request_row_ids_sha256,
            payload_sha256,
            payload: Vec::new(),
        },
        payload_start..payload_end,
    ))
}

fn checked_payload_len(rows: usize) -> Result<usize> {
    if rows > MAX_LOOKUPS {
        return Err(ExchangeError("lookup count exceeds limit"));
    }
    let bytes = rows
        .checked_mul(ROW_BYTES)
        .ok_or(ExchangeError("payload size overflow"))?;
    if bytes > MAX_PAYLOAD_BYTES {
        return Err(ExchangeError("payload exceeds frame limit"));
    }
    Ok(bytes)
}

fn row_ids_sha256(row_ids: &[u64]) -> String {
    let mut canonical = Vec::with_capacity(std::mem::size_of_val(row_ids));
    for row_id in row_ids {
        canonical.extend_from_slice(&row_id.to_le_bytes());
    }
    sha256::hex(&canonical)
}

fn validate_digest(value: &str) -> Result<()> {
    if sha256::is_hex_digest(value) {
        Ok(())
    } else {
        Err(ExchangeError("invalid SHA-256 identity"))
    }
}

fn validate_identity(stamp: &Stamp, table: &TableIdentity) -> Result<()> {
    if stamp.run_id.is_empty() || stamp.plan_id.is_empty() || stamp.model_revision.is_empty() {
        return Err(ExchangeError("empty execution identity"));
    }
    if stamp.model_revision != table.model_revision
        || stamp.layer != table.layer
        || stamp.epoch != table.epoch
    {
        return Err(ExchangeError("stamp and table identity mismatch"));
    }
    validate_table(table)
}

fn validate_table(table: &TableIdentity) -> Result<()> {
    if table.model_revision.is_empty() || table.num_embeddings == 0 {
        return Err(ExchangeError("empty table identity or Engram domain"));
    }
    validate_digest(&table.table_sha256)?;
    validate_digest(&table.tokenizer_map_sha256)?;
    Ok(())
}

fn shard_range(table: &TableIdentity, world_size: u32, rank: u32) -> Result<(u64, u64)> {
    if world_size == 0 || world_size > MAX_WORLD_SIZE || rank >= world_size {
        return Err(ExchangeError("invalid rank or world size"));
    }
    let part = table
        .num_embeddings
        .checked_add(u64::from(world_size) - 1)
        .ok_or(ExchangeError("shard range overflow"))?
        / u64::from(world_size);
    let start = part
        .checked_mul(u64::from(rank))
        .ok_or(ExchangeError("shard start overflow"))?;
    let end = start
        .checked_add(part)
        .ok_or(ExchangeError("shard end overflow"))?
        .min(table.num_embeddings);
    Ok((start.min(table.num_embeddings), end))
}

fn owner_rank(table: &TableIdentity, world_size: u32, row_id: u64) -> Result<u32> {
    if world_size == 0 || world_size > MAX_WORLD_SIZE || row_id >= table.num_embeddings {
        return Err(ExchangeError("row ID or world size outside domain"));
    }
    let part = table
        .num_embeddings
        .checked_add(u64::from(world_size) - 1)
        .ok_or(ExchangeError("shard range overflow"))?
        / u64::from(world_size);
    if part == 0 {
        return Err(ExchangeError("row has no owner"));
    }
    let rank = u32::try_from(row_id / part).map_err(|_| ExchangeError("owner rank overflow"))?;
    if rank >= world_size {
        return Err(ExchangeError("row has no owner"));
    }
    Ok(rank)
}

/// Partition an ordered batch into owner-local ID lists. Each list preserves the original
/// relative order and duplicates, so workers never receive IDs for rows they cannot own.
pub fn partition_row_ids(
    table: &TableIdentity,
    world_size: u32,
    row_ids: &[u64],
) -> Result<Vec<Vec<u64>>> {
    validate_table(table)?;
    checked_payload_len(row_ids.len())?;
    if world_size == 0 || world_size > MAX_WORLD_SIZE {
        return Err(ExchangeError("invalid rank or world size"));
    }
    let mut partitions = Vec::new();
    partitions
        .try_reserve_exact(world_size as usize)
        .map_err(|_| ExchangeError("owner partition allocation failed"))?;
    partitions.resize_with(world_size as usize, Vec::new);
    for &row_id in row_ids {
        let owner = owner_rank(table, world_size, row_id)? as usize;
        partitions[owner]
            .try_reserve(1)
            .map_err(|_| ExchangeError("owner partition allocation failed"))?;
        partitions[owner].push(row_id);
    }
    Ok(partitions)
}

/// Partition an ordered row batch and retain only the first occurrence of each row ID.
/// Owners return a row once; the coordinator can expand duplicate lookup positions after
/// validating the unique owner frames.
pub fn partition_unique_row_ids(
    table: &TableIdentity,
    world_size: u32,
    row_ids: &[u64],
) -> Result<Vec<Vec<u64>>> {
    let mut partitions = partition_row_ids(table, world_size, row_ids)?;
    for ids in &mut partitions {
        let mut seen = HashSet::new();
        seen.try_reserve(ids.len())
            .map_err(|_| ExchangeError("owner deduplication allocation failed"))?;
        ids.retain(|id| seen.insert(*id));
    }
    Ok(partitions)
}

/// Pack only native rows selected by this owner, preserving hash lookup order.
pub fn pack_owner_rows(
    stamp: &Stamp,
    table: &TableIdentity,
    world_size: u32,
    sender_rank: u32,
    row_ids: &[u64],
    local_table_bytes: &[u8],
) -> Result<OwnerRows> {
    validate_identity(stamp, table)?;
    checked_payload_len(row_ids.len())?;
    let (first_row, end_row) = shard_range(table, world_size, sender_rank)?;
    let shard_rows = end_row - first_row;
    let expected_shard_bytes = usize::try_from(shard_rows)
        .ok()
        .and_then(|n| n.checked_mul(ROW_BYTES))
        .ok_or(ExchangeError("shard byte length overflow"))?;
    if local_table_bytes.len() != expected_shard_bytes {
        return Err(ExchangeError("local shard length mismatch"));
    }
    let mut payload = Vec::new();
    for &row_id in row_ids {
        if row_id >= table.num_embeddings {
            return Err(ExchangeError("row ID outside table"));
        }
        if (first_row..end_row).contains(&row_id) {
            let local = usize::try_from(row_id - first_row)
                .ok()
                .and_then(|n| n.checked_mul(ROW_BYTES))
                .ok_or(ExchangeError("local row offset overflow"))?;
            let slice = local_table_bytes
                .get(local..local + ROW_BYTES)
                .ok_or(ExchangeError("selected row outside local shard"))?;
            payload.extend_from_slice(slice);
        }
    }
    let row_count = u32::try_from(payload.len() / ROW_BYTES)
        .map_err(|_| ExchangeError("owner row count overflow"))?;
    let payload_sha256 = sha256::hex(&payload);
    Ok(OwnerRows {
        stamp: stamp.clone(),
        table: table.clone(),
        sender_rank,
        world_size,
        first_row,
        shard_rows,
        row_count,
        request_row_ids_sha256: row_ids_sha256(row_ids),
        payload_sha256,
        payload,
    })
}

/// Validate all owner frames and reconstruct exactly one row per requested ID.
pub fn reconstruct_rows(
    stamp: &Stamp,
    table: &TableIdentity,
    world_size: u32,
    row_ids: &[u64],
    owners: &[OwnerRows],
) -> Result<Vec<u8>> {
    validate_identity(stamp, table)?;
    checked_payload_len(row_ids.len())?;
    if world_size == 0 || world_size > MAX_WORLD_SIZE {
        return Err(ExchangeError("invalid rank or world size"));
    }
    if owners.len() != world_size as usize {
        return Err(ExchangeError("missing owner frame"));
    }
    let expected_request_digest = row_ids_sha256(row_ids);
    let output_len = checked_payload_len(row_ids.len())?;
    let mut by_rank: Vec<Option<&OwnerRows>> = vec![None; world_size as usize];
    for owner in owners {
        if owner.stamp != *stamp
            || owner.table != *table
            || owner.world_size != world_size
            || owner.request_row_ids_sha256 != expected_request_digest
        {
            return Err(ExchangeError("stale or mismatched owner identity"));
        }
        let rank = owner.sender_rank as usize;
        if rank >= by_rank.len() || by_rank[rank].replace(owner).is_some() {
            return Err(ExchangeError("invalid or duplicate owner rank"));
        }
    }
    let mut expected_counts = vec![0usize; world_size as usize];
    for &row_id in row_ids {
        if row_id >= table.num_embeddings {
            return Err(ExchangeError("row ID outside table"));
        }
        expected_counts[owner_rank(table, world_size, row_id)? as usize] += 1;
    }
    for (rank, maybe_owner) in by_rank.iter().enumerate() {
        let owner = maybe_owner.ok_or(ExchangeError("missing owner frame"))?;
        let (expected_start, expected_end) = shard_range(table, world_size, rank as u32)?;
        let expected_bytes = checked_payload_len(expected_counts[rank])?;
        if owner.first_row != expected_start
            || owner.shard_rows != expected_end - expected_start
            || owner.row_count as usize != expected_counts[rank]
            || owner.payload.len() != expected_bytes
        {
            return Err(ExchangeError(
                "owner row coverage or payload length mismatch",
            ));
        }
        if sha256::hex(&owner.payload) != owner.payload_sha256 {
            return Err(ExchangeError("owner payload digest mismatch"));
        }
    }

    let mut cursors = vec![0usize; world_size as usize];
    let mut output = vec![0u8; output_len];
    for (position, &row_id) in row_ids.iter().enumerate() {
        let owner_rank = owner_rank(table, world_size, row_id)? as usize;
        let owner = by_rank[owner_rank].ok_or(ExchangeError("missing owner frame"))?;
        let source = cursors[owner_rank];
        let destination = position * ROW_BYTES;
        output[destination..destination + ROW_BYTES]
            .copy_from_slice(&owner.payload[source..source + ROW_BYTES]);
        cursors[owner_rank] += ROW_BYTES;
    }
    Ok(output)
}

/// Reconstruct a complete request from only the owner frames that had selected rows.
/// Each frame is bound to that owner's filtered ID list; positions and duplicates in the
/// original request are restored here. A missing nonempty owner or an extra empty-owner
/// frame is rejected before any result is returned.
pub fn reconstruct_partitioned_rows(
    stamp: &Stamp,
    table: &TableIdentity,
    world_size: u32,
    row_ids: &[u64],
    owners: &[OwnerRows],
) -> Result<Vec<u8>> {
    reconstruct_partitioned_rows_core(stamp, table, world_size, row_ids, owners.len(), |index| {
        OwnerFrameRef::Unverified(&owners[index])
    })
}

/// Reconstruct rows from frames returned by `decode_owner_frame_verified` without hashing the
/// immutable payloads a second time. Their embedded RequestBinding is compared against this
/// exact reconstruction request, so a stale epoch, reordered IDs, or a frame from another batch
/// remains a hard refusal.
pub fn reconstruct_partitioned_rows_verified(
    stamp: &Stamp,
    table: &TableIdentity,
    world_size: u32,
    row_ids: &[u64],
    owners: &[VerifiedOwnerRows],
) -> Result<Vec<u8>> {
    reconstruct_partitioned_rows_core(stamp, table, world_size, row_ids, owners.len(), |index| {
        OwnerFrameRef::Verified(&owners[index])
    })
}

/// Reconstruct rows from consumed owner response frames. Payload bytes are read in place from
/// each immutable frame allocation; this saves one payload-sized allocation and copy per owner.
pub fn reconstruct_partitioned_rows_owned_verified(
    stamp: &Stamp,
    table: &TableIdentity,
    world_size: u32,
    row_ids: &[u64],
    owners: &[VerifiedOwnerFrame],
) -> Result<Vec<u8>> {
    reconstruct_partitioned_rows_core(stamp, table, world_size, row_ids, owners.len(), |index| {
        OwnerFrameRef::Owned(&owners[index])
    })
}

#[derive(Clone, Copy)]
enum OwnerFrameRef<'a> {
    Unverified(&'a OwnerRows),
    Verified(&'a VerifiedOwnerRows),
    Owned(&'a VerifiedOwnerFrame),
}

impl<'a> OwnerFrameRef<'a> {
    fn rows(self) -> &'a OwnerRows {
        match self {
            Self::Unverified(rows) => rows,
            Self::Verified(rows) => &rows.rows,
            Self::Owned(rows) => &rows.rows,
        }
    }

    fn payload(self) -> &'a [u8] {
        match self {
            Self::Unverified(rows) => &rows.payload,
            Self::Verified(rows) => &rows.rows.payload,
            Self::Owned(rows) => rows.payload(),
        }
    }

    fn request(self) -> Option<(&'a RequestBinding, u32)> {
        match self {
            Self::Unverified(_) => None,
            Self::Verified(rows) => Some((&rows.request, rows.expected_sender_rank)),
            Self::Owned(rows) => Some((&rows.request, rows.expected_sender_rank)),
        }
    }

    fn payload_digest_was_validated(self) -> bool {
        !matches!(self, Self::Unverified(_))
    }
}

fn reconstruct_partitioned_rows_core<'a>(
    stamp: &Stamp,
    table: &TableIdentity,
    world_size: u32,
    row_ids: &[u64],
    owner_count: usize,
    owner_at: impl Fn(usize) -> OwnerFrameRef<'a>,
) -> Result<Vec<u8>> {
    validate_identity(stamp, table)?;
    checked_payload_len(row_ids.len())?;
    let partitions = partition_unique_row_ids(table, world_size, row_ids)?;
    let expected_frames = partitions.iter().filter(|ids| !ids.is_empty()).count();
    if owner_count != expected_frames {
        return Err(ExchangeError("missing or unexpected owner frame"));
    }
    let mut by_rank: Vec<Option<OwnerFrameRef<'a>>> = vec![None; world_size as usize];
    for index in 0..owner_count {
        let frame = owner_at(index);
        let rank = frame.rows().sender_rank as usize;
        if rank >= by_rank.len() || by_rank[rank].replace(frame).is_some() {
            return Err(ExchangeError("invalid or duplicate owner rank"));
        }
    }
    for (rank, ids) in partitions.iter().enumerate() {
        match (ids.is_empty(), by_rank[rank]) {
            (true, None) => continue,
            (true, Some(_)) => return Err(ExchangeError("unexpected empty-owner frame")),
            (false, None) => return Err(ExchangeError("missing owner frame")),
            (false, Some(owner_frame)) => {
                let owner = owner_frame.rows();
                let binding = RequestBinding::new(stamp, table, world_size, ids)?;
                let (first_row, end_row) = shard_range(table, world_size, rank as u32)?;
                let expected_len = checked_payload_len(ids.len())?;
                let verified_request_mismatch =
                    owner_frame.request().is_some_and(|(request, sender_rank)| {
                        request != &binding || sender_rank != rank as u32
                    });
                let owner_payload = owner_frame.payload();
                let payload_digest_mismatch = !owner_frame.payload_digest_was_validated()
                    && sha256::hex(owner_payload) != owner.payload_sha256;
                if owner.stamp != *stamp
                    || owner.table != *table
                    || owner.sender_rank != rank as u32
                    || owner.world_size != world_size
                    || owner.first_row != first_row
                    || owner.shard_rows != end_row - first_row
                    || owner.row_count as usize != ids.len()
                    || owner.request_row_ids_sha256 != binding.row_ids_sha256
                    || owner_payload.len() != expected_len
                    || verified_request_mismatch
                    || payload_digest_mismatch
                {
                    return Err(ExchangeError("owner row coverage or identity mismatch"));
                }
            }
        }
    }
    let output_len = checked_payload_len(row_ids.len())?;
    let mut row_offsets: Vec<HashMap<u64, usize>> = Vec::new();
    row_offsets
        .try_reserve_exact(world_size as usize)
        .map_err(|_| ExchangeError("row index allocation failed"))?;
    for (rank, ids) in partitions.iter().enumerate() {
        let mut offsets = HashMap::new();
        offsets
            .try_reserve(ids.len())
            .map_err(|_| ExchangeError("row index allocation failed"))?;
        for (index, &id) in ids.iter().enumerate() {
            offsets.insert(id, index);
        }
        if !ids.is_empty() && by_rank[rank].is_none() {
            return Err(ExchangeError("missing owner frame"));
        }
        row_offsets.push(offsets);
    }
    let mut output = vec![0u8; output_len];
    for (position, &row_id) in row_ids.iter().enumerate() {
        let rank = owner_rank(table, world_size, row_id)? as usize;
        let owner_frame = by_rank[rank].ok_or(ExchangeError("missing owner frame"))?;
        let owner_payload = owner_frame.payload();
        let unique_index = *row_offsets[rank]
            .get(&row_id)
            .ok_or(ExchangeError("requested row is absent from owner frame"))?;
        let source = unique_index * ROW_BYTES;
        let destination = position * ROW_BYTES;
        output[destination..destination + ROW_BYTES]
            .copy_from_slice(&owner_payload[source..source + ROW_BYTES]);
    }
    Ok(output)
}

#[cfg(test)]
mod partition_tests {
    use super::*;

    fn fixture() -> (TableIdentity, Stamp, Vec<Vec<u8>>, Vec<EngramShard>) {
        let table = TableIdentity {
            model_revision: "partition-test-v1".into(),
            table_sha256: "11".repeat(32),
            tokenizer_map_sha256: "22".repeat(32),
            layer: 7,
            epoch: 3,
            num_embeddings: 8,
        };
        let stamp = Stamp {
            run_id: "partition-test".into(),
            model_revision: table.model_revision.clone(),
            plan_id: "owner-filtered".into(),
            layer: table.layer,
            batch: 1,
            epoch: table.epoch,
        };
        let rows: Vec<Vec<u8>> = (0..8)
            .map(|row| {
                (0..ROW_BYTES)
                    .map(|col| (row * 19 + col * 3) as u8)
                    .collect()
            })
            .collect();
        let shards = (0..2)
            .map(|rank| {
                let bytes = rows[rank * 4..rank * 4 + 4].concat();
                EngramShard::new(table.clone(), 2, rank as u32, bytes).unwrap()
            })
            .collect();
        (table, stamp, rows, shards)
    }

    #[test]
    fn owner_partition_preserves_input_order_duplicates_and_empty_owners() {
        let (table, _, _, _) = fixture();
        let parts = partition_row_ids(&table, 2, &[7, 0, 4, 7, 3, 2, 6, 1]).unwrap();
        assert_eq!(parts, [vec![0, 3, 2, 1], vec![7, 4, 7, 6]]);
        assert_eq!(
            partition_row_ids(&table, 2, &[0, 1, 1, 3]).unwrap(),
            [vec![0, 1, 1, 3], vec![]]
        );
        assert!(partition_row_ids(&table, 2, &[8]).is_err());
    }

    #[test]
    fn owner_query_deduplication_keeps_first_occurrence_order() {
        let (table, _, _, _) = fixture();
        let unique = partition_unique_row_ids(&table, 2, &[7, 0, 4, 7, 3, 2, 6, 1, 0]);
        assert_eq!(unique.unwrap(), [vec![0, 3, 2, 1], vec![7, 4, 6]]);
    }

    #[test]
    fn partitioned_reconstruction_accepts_sparse_frames_and_reorders_exactly() {
        let (table, stamp, rows, shards) = fixture();
        let ids = [7, 0, 4, 7, 3, 2, 6, 1];
        let parts = partition_unique_row_ids(&table, 2, &ids).unwrap();
        let mut responses: Vec<_> = shards
            .iter()
            .enumerate()
            .map(|(rank, shard)| shard.rows(&stamp, &parts[rank]).unwrap())
            .collect();
        assert_eq!(
            responses
                .iter()
                .map(|owner| owner.payload.len())
                .sum::<usize>(),
            7 * ROW_BYTES,
            "one duplicate lookup must not duplicate a transferred row"
        );
        responses.reverse();
        let expected: Vec<u8> = ids
            .iter()
            .flat_map(|&id| rows[id as usize].iter().copied())
            .collect();
        assert_eq!(
            reconstruct_partitioned_rows(&stamp, &table, 2, &ids, &responses).unwrap(),
            expected
        );
    }

    #[test]
    fn immutable_shard_frame_path_hashes_payload_once_and_matches_public_encoder() {
        let (table, stamp, _, shards) = fixture();
        let ids = [0u64, 2];
        let binding = RequestBinding::new(&stamp, &table, 2, &ids).unwrap();
        let rows = shards[0].rows(&stamp, &ids).unwrap();
        let expected = encode_owner_frame(&rows, &binding, 0).unwrap();

        crate::sha256::reset_test_hashed_bytes();
        let actual = shards[0]
            .encode_rows_frame(&stamp, &ids, &binding, 0)
            .unwrap();

        assert_eq!(
            actual, expected,
            "optimized bytes must match the public codec"
        );
        assert_eq!(
            crate::sha256::test_hashed_bytes(),
            rows.payload.len() + ids.len() * std::mem::size_of::<u64>(),
            "packing and encoding an immutable shard must hash the payload once"
        );
        let mut forged = rows;
        forged.payload[0] ^= 1;
        assert_eq!(
            encode_owner_frame(&forged, &binding, 0),
            Err(ExchangeError("owner payload digest mismatch")),
            "the public encoder still rejects caller-mutated payloads"
        );
    }

    #[test]
    fn decoded_owner_reconstruction_does_not_hash_payload_twice() {
        let (table, stamp, rows, shards) = fixture();
        let row_ids = [7u64, 0, 4, 7, 3, 2, 6, 1];
        let partitions = partition_unique_row_ids(&table, 2, &row_ids).unwrap();
        let mut bindings = Vec::new();
        let mut frames = Vec::new();
        for (rank, shard) in shards.iter().enumerate() {
            let binding = RequestBinding::new(&stamp, &table, 2, &partitions[rank]).unwrap();
            let packed = shard.rows(&stamp, &partitions[rank]).unwrap();
            frames.push(encode_owner_frame(&packed, &binding, rank as u32).unwrap());
            bindings.push(binding);
        }
        let payload_bytes = partitions
            .iter()
            .map(|ids| ids.len() * ROW_BYTES)
            .sum::<usize>();
        let request_digest_input_bytes = partitions
            .iter()
            .map(|ids| ids.len() * std::mem::size_of::<u64>())
            .sum::<usize>();

        crate::sha256::reset_test_hashed_bytes();
        let decoded: Vec<_> = frames
            .iter()
            .zip(&bindings)
            .enumerate()
            .map(|(rank, (frame, binding))| {
                decode_owner_frame_verified(frame, binding, rank as u32).unwrap()
            })
            .collect();
        let actual =
            reconstruct_partitioned_rows_verified(&stamp, &table, 2, &row_ids, &decoded).unwrap();
        let expected: Vec<u8> = row_ids
            .iter()
            .flat_map(|&id| rows[id as usize].iter().copied())
            .collect();
        assert_eq!(actual, expected);
        assert_eq!(
            crate::sha256::test_hashed_bytes(),
            payload_bytes + request_digest_input_bytes,
            "decode validates the payload digest; reconstruction must reuse that proof"
        );

        assert_eq!(
            reconstruct_partitioned_rows_verified(&stamp, &table, 2, &row_ids, &decoded),
            Ok(expected)
        );
        assert_eq!(
            reconstruct_partitioned_rows_verified(&stamp, &table, 2, &row_ids[..7], &decoded),
            Err(ExchangeError("owner row coverage or identity mismatch")),
            "a verified frame cannot be reused for a different batch"
        );
        let reordered_ids = [1u64, 6, 2, 3, 7, 4, 0, 7];
        assert_eq!(
            reconstruct_partitioned_rows_verified(&stamp, &table, 2, &reordered_ids, &decoded,),
            Err(ExchangeError("owner row coverage or identity mismatch")),
            "verified frames remain bound to each owner's exact ordered ID list"
        );
        let mut stale_stamp = stamp.clone();
        stale_stamp.epoch += 1;
        let mut stale_table = table.clone();
        stale_table.epoch += 1;
        assert_eq!(
            reconstruct_partitioned_rows_verified(
                &stale_stamp,
                &stale_table,
                2,
                &row_ids,
                &decoded,
            ),
            Err(ExchangeError("owner row coverage or identity mismatch")),
            "verified frames cannot cross ownership epochs"
        );
    }

    #[test]
    fn owned_verified_decode_reads_payload_from_consumed_frame_allocation() {
        let (table, stamp, rows, shards) = fixture();
        let row_ids = [7u64, 0, 4, 7, 3, 2, 6, 1];
        let partitions = partition_unique_row_ids(&table, 2, &row_ids).unwrap();
        let mut owners = Vec::new();
        for (rank, shard) in shards.iter().enumerate() {
            let ids = &partitions[rank];
            if ids.is_empty() {
                continue;
            }
            let binding = RequestBinding::new(&stamp, &table, 2, ids).unwrap();
            let packed = shard.rows(&stamp, ids).unwrap();
            let frame = encode_owner_frame(&packed, &binding, rank as u32).unwrap();
            let frame_start = frame.as_ptr() as usize;
            let frame_end = frame_start + frame.len();
            let verified = decode_owner_frame_verified_owned(frame, &binding, rank as u32).unwrap();
            let payload = verified.payload();
            let payload_start = payload.as_ptr() as usize;
            assert!(payload_start >= frame_start);
            assert!(payload_start + payload.len() <= frame_end);
            assert_eq!(payload, packed.payload);
            owners.push(verified);
        }

        let actual =
            reconstruct_partitioned_rows_owned_verified(&stamp, &table, 2, &row_ids, &owners)
                .unwrap();
        let expected: Vec<u8> = row_ids
            .iter()
            .flat_map(|&id| rows[id as usize].iter().copied())
            .collect();
        assert_eq!(actual, expected);

        let mut reordered_owners = owners;
        reordered_owners.reverse();
        assert_eq!(
            reconstruct_partitioned_rows_owned_verified(
                &stamp,
                &table,
                2,
                &row_ids,
                &reordered_owners,
            ),
            Ok(expected.clone()),
            "owned frames are placed by verified rank, not arrival order"
        );
        assert_eq!(
            reconstruct_partitioned_rows_owned_verified(
                &stamp,
                &table,
                2,
                &row_ids[..7],
                &reordered_owners,
            ),
            Err(ExchangeError("owner row coverage or identity mismatch")),
            "owned frames remain bound to the exact ordered owner-local IDs"
        );
        let mut stale_stamp = stamp.clone();
        stale_stamp.epoch += 1;
        let mut stale_table = table.clone();
        stale_table.epoch += 1;
        assert_eq!(
            reconstruct_partitioned_rows_owned_verified(
                &stale_stamp,
                &stale_table,
                2,
                &row_ids,
                &reordered_owners,
            ),
            Err(ExchangeError("owner row coverage or identity mismatch")),
            "an owned frame cannot cross the verified table epoch"
        );
        assert_eq!(
            reconstruct_partitioned_rows_owned_verified(
                &stamp,
                &table,
                2,
                &row_ids,
                &reordered_owners[..1],
            ),
            Err(ExchangeError("missing or unexpected owner frame")),
            "an incomplete owned-frame batch publishes no reconstructed result"
        );

        let binding = RequestBinding::new(&stamp, &table, 2, &partitions[0]).unwrap();
        let packed = shards[0].rows(&stamp, &partitions[0]).unwrap();
        let frame = encode_owner_frame(&packed, &binding, 0).unwrap();
        let mut wrong_order = partitions[0].clone();
        wrong_order.reverse();
        let wrong_binding = RequestBinding::new(&stamp, &table, 2, &wrong_order).unwrap();
        assert!(
            matches!(
                decode_owner_frame_verified_owned(frame.clone(), &wrong_binding, 0),
                Err(ExchangeError("requested row ID digest mismatch"))
            ),
            "owned decode rejects a frame requested with a different owner-local order"
        );
        let mut corrupt = frame;
        let last = corrupt.len() - 1;
        corrupt[last] ^= 1;
        assert!(matches!(
            decode_owner_frame_verified_owned(corrupt, &binding, 0),
            Err(ExchangeError("owner payload digest mismatch"))
        ));
    }

    #[test]
    fn mutable_owner_reconstruction_still_rehashes_and_rejects_mutation() {
        let (table, stamp, _, shards) = fixture();
        let row_ids = [0u64, 1, 4, 7];
        let partitions = partition_unique_row_ids(&table, 2, &row_ids).unwrap();
        let mut owners: Vec<_> = shards
            .iter()
            .enumerate()
            .map(|(rank, shard)| shard.rows(&stamp, &partitions[rank]).unwrap())
            .collect();
        let binding = RequestBinding::new(&stamp, &table, 2, &partitions[0]).unwrap();
        let frame = encode_owner_frame(&owners[0], &binding, 0).unwrap();
        owners[0] = decode_owner_frame(&frame, &binding, 0).unwrap();
        owners[0].payload[0] ^= 1;

        crate::sha256::reset_test_hashed_bytes();
        assert_eq!(
            reconstruct_partitioned_rows(&stamp, &table, 2, &row_ids, &owners),
            Err(ExchangeError("owner row coverage or identity mismatch"))
        );
        assert!(
            crate::sha256::test_hashed_bytes() >= owners[0].payload.len(),
            "the mutable compatibility path must revalidate caller-owned bytes"
        );
    }

    #[test]
    fn partitioned_reconstruction_refuses_missing_and_empty_owner_frames() {
        let (table, stamp, rows, shards) = fixture();
        let ids = [0, 1, 1, 3];
        let part = partition_unique_row_ids(&table, 2, &ids).unwrap();
        let owner0 = shards[0].rows(&stamp, &part[0]).unwrap();
        let empty_owner1 = shards[1].rows(&stamp, &part[1]).unwrap();
        let expected: Vec<u8> = ids
            .iter()
            .flat_map(|&id| rows[id as usize].iter().copied())
            .collect();
        assert_eq!(
            reconstruct_partitioned_rows(&stamp, &table, 2, &ids, std::slice::from_ref(&owner0))
                .unwrap(),
            expected
        );
        assert_eq!(
            reconstruct_partitioned_rows(&stamp, &table, 2, &ids, &[owner0.clone(), empty_owner1]),
            Err(ExchangeError("missing or unexpected owner frame"))
        );
        assert_eq!(
            reconstruct_partitioned_rows(&stamp, &table, 2, &ids, &[]),
            Err(ExchangeError("missing or unexpected owner frame"))
        );
    }

    #[test]
    fn bounded_row_cache_binds_full_identity_and_evicts_oldest_without_exceeding_capacity() {
        let (table, _, rows, _) = fixture();
        let mut cache = EngramRowCache::new(2 * ROW_BYTES);
        assert!(cache.insert(&table, 0, &rows[0]).unwrap());
        assert!(cache.insert(&table, 1, &rows[1]).unwrap());
        assert_eq!(cache.resident_bytes(), 2 * ROW_BYTES);

        let mut new_epoch = table.clone();
        new_epoch.epoch += 1;
        assert_eq!(cache.get(&new_epoch, 0), None);
        let mut new_revision = table.clone();
        new_revision.model_revision.push_str("-new");
        assert_eq!(cache.get(&new_revision, 0), None);
        let mut new_table_hash = table.clone();
        new_table_hash.table_sha256 = "33".repeat(32);
        assert_eq!(cache.get(&new_table_hash, 0), None);
        let mut new_tokenizer_hash = table.clone();
        new_tokenizer_hash.tokenizer_map_sha256 = "44".repeat(32);
        assert_eq!(cache.get(&new_tokenizer_hash, 0), None);
        let mut new_layer = table.clone();
        new_layer.layer += 1;
        assert_eq!(cache.get(&new_layer, 0), None);
        let mut new_size = table.clone();
        new_size.num_embeddings += 1;
        assert_eq!(cache.get(&new_size, 0), None);

        assert!(cache.insert(&table, 2, &rows[2]).unwrap());
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.resident_bytes(), 2 * ROW_BYTES);
        assert_eq!(cache.get(&table, 0), None, "FIFO evicts oldest row");
        assert_eq!(cache.get(&table, 1), Some(rows[1].as_slice()));
        assert_eq!(cache.get(&table, 2), Some(rows[2].as_slice()));
    }

    #[test]
    fn row_cache_refuses_bad_rows_and_treats_subrow_capacity_as_disabled() {
        let (table, _, rows, _) = fixture();
        let mut disabled = EngramRowCache::new(ROW_BYTES - 1);
        assert!(!disabled.insert(&table, 0, &rows[0]).unwrap());
        assert!(disabled.is_empty());
        assert!(disabled
            .insert(&table, table.num_embeddings, &rows[0])
            .is_err());
        assert!(disabled
            .insert(&table, 0, &rows[0][..ROW_BYTES - 1])
            .is_err());
        assert!(disabled.is_empty());
    }

    #[test]
    fn row_cache_interns_each_identity_once_and_releases_it_after_eviction() {
        let (table, _, rows, _) = fixture();
        let mut cache = EngramRowCache::new(ROW_BYTES);
        assert!(cache.insert(&table, 0, &rows[0]).unwrap());
        assert_eq!(cache.resident_table_count(), 1);
        let mut next_epoch = table.clone();
        next_epoch.epoch += 1;
        assert!(cache.insert(&next_epoch, 0, &rows[0]).unwrap());
        assert_eq!(cache.resident_table_count(), 1);
        assert_eq!(cache.get(&table, 0), None);
        assert_eq!(cache.get(&next_epoch, 0), Some(rows[0].as_slice()));
    }

    #[test]
    fn row_cache_retains_identity_when_fifo_evicts_its_last_row() {
        let (table, _, rows, _) = fixture();
        let mut cache = EngramRowCache::new(ROW_BYTES);
        assert!(cache.insert(&table, 0, &rows[0]).unwrap());
        assert!(cache.insert(&table, 1, &rows[1]).unwrap());
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.resident_table_count(), 1);
        assert_eq!(cache.get(&table, 0), None);
        assert_eq!(cache.get(&table, 1), Some(rows[1].as_slice()));
    }

    #[test]
    fn row_cache_batch_lookup_preserves_order_duplicates_and_identity() {
        let (table, _, rows, _) = fixture();
        let mut cache = EngramRowCache::new(2 * ROW_BYTES);
        cache.insert(&table, 0, &rows[0]).unwrap();
        cache.insert(&table, 1, &rows[1]).unwrap();
        let mut batch = Vec::new();
        cache
            .for_each_lookup(&table, &[1, 0, 1, 2], |id, row| {
                batch.push((id, row.map(<[u8]>::to_vec)));
                Ok::<(), ()>(())
            })
            .unwrap();
        assert_eq!(
            batch,
            [
                (1, Some(rows[1].clone())),
                (0, Some(rows[0].clone())),
                (1, Some(rows[1].clone())),
                (2, None)
            ]
        );
        let mut stale = table.clone();
        stale.epoch += 1;
        let mut stale_rows = Vec::new();
        cache
            .for_each_lookup(&stale, &[0, 1], |id, row| {
                stale_rows.push((id, row.is_some()));
                Ok::<(), ()>(())
            })
            .unwrap();
        assert_eq!(stale_rows, [(0, false), (1, false)]);
    }

    #[test]
    fn row_cache_batch_insert_matches_scalar_fifo_and_identity() {
        let (table, _, rows, _) = fixture();
        let ids = [0, 1, 0, 2, 3, 4, 5];
        let batch_rows: Vec<Vec<u8>> = ids
            .iter()
            .enumerate()
            .map(|(index, &id)| {
                if index == 2 {
                    vec![0xee; ROW_BYTES]
                } else {
                    rows[id as usize].clone()
                }
            })
            .collect();
        let payload: Vec<u8> = batch_rows.iter().flatten().copied().collect();

        for capacity_rows in [1, 2, ids.len()] {
            let capacity = capacity_rows * ROW_BYTES;
            let mut scalar = EngramRowCache::new(capacity);
            let mut batch = EngramRowCache::new(capacity);
            let mut fifo_oracle: VecDeque<(u64, Vec<u8>)> = VecDeque::new();
            for (&id, row) in ids.iter().zip(&batch_rows) {
                scalar.insert(&table, id, row).unwrap();
                if !fifo_oracle
                    .iter()
                    .any(|(resident_id, _)| *resident_id == id)
                {
                    if fifo_oracle.len() == capacity_rows {
                        fifo_oracle.pop_front();
                    }
                    fifo_oracle.push_back((id, row.clone()));
                }
            }
            batch.insert_many(&table, &ids, &payload).unwrap();
            assert_eq!(batch.len(), scalar.len());
            assert_eq!(batch.resident_bytes(), scalar.resident_bytes());
            assert_eq!(batch.resident_table_count(), scalar.resident_table_count());
            for id in 0..table.num_embeddings {
                let expected = fifo_oracle
                    .iter()
                    .find(|(resident_id, _)| *resident_id == id)
                    .map(|(_, row)| row.as_slice());
                assert_eq!(batch.get(&table, id), expected);
                assert_eq!(scalar.get(&table, id), expected);
            }

            let mut stale_epoch = table.clone();
            stale_epoch.epoch += 1;
            assert_eq!(batch.get(&stale_epoch, 0), None);
        }
    }

    #[test]
    fn row_cache_batch_insert_rejects_invalid_batch_before_mutation() {
        let (table, _, rows, _) = fixture();
        let mut cache = EngramRowCache::new(2 * ROW_BYTES);
        cache.insert(&table, 0, &rows[0]).unwrap();
        let before = cache.get(&table, 0).unwrap().to_vec();
        let before_len = cache.len();
        let before_bytes = cache.resident_bytes();

        let short_payload = vec![1; ROW_BYTES];
        assert!(cache.insert_many(&table, &[1, 2], &short_payload).is_err());
        let valid_payload = [rows[1].as_slice(), rows[2].as_slice()].concat();
        assert!(cache
            .insert_many(&table, &[1, table.num_embeddings], &valid_payload)
            .is_err());
        assert_eq!(cache.get(&table, 0), Some(before.as_slice()));
        assert_eq!(cache.get(&table, 1), None);
        assert_eq!(cache.len(), before_len);
        assert_eq!(cache.resident_bytes(), before_bytes);
    }

    fn current_rss_kib() -> u64 {
        let status = std::fs::read_to_string("/proc/self/status").unwrap();
        status
            .lines()
            .find(|line| line.starts_with("VmRSS:"))
            .and_then(|line| line.split_whitespace().nth(1))
            .unwrap()
            .parse()
            .unwrap()
    }

    /// Run this ignored test in a fresh process for each capacity. RSS includes allocator,
    /// hash-index and test-runtime overhead; resident_bytes is only the row payload bound.
    #[test]
    #[ignore = "explicit Linux RSS profile; run separately for each VF_CACHE_PROFILE_ROWS"]
    fn engram_cache_rss_profile() {
        let row_count: usize = std::env::var("VF_CACHE_PROFILE_ROWS")
            .expect("set VF_CACHE_PROFILE_ROWS")
            .parse()
            .unwrap();
        assert!(row_count > 0 && row_count <= u32::MAX as usize);
        let table = TableIdentity {
            model_revision: "a68-memory-profile-v1".into(),
            table_sha256: "51".repeat(32),
            tokenizer_map_sha256: "62".repeat(32),
            layer: 1,
            epoch: 1,
            num_embeddings: row_count as u64,
        };
        let row = [0x5a; ROW_BYTES];
        let before_kib = current_rss_kib();
        let mut cache = EngramRowCache::new(row_count * ROW_BYTES);
        let insert_start = std::time::Instant::now();
        for id in 0..row_count as u64 {
            assert!(cache.insert(&table, id, &row).unwrap());
        }
        let insert_ns_per_row = insert_start.elapsed().as_nanos() / row_count as u128;
        let lookup_repetitions = (2_097_152 / row_count).max(1);
        let mut lookup_rounds = Vec::with_capacity(7);
        let mut lookup_checksum = 0u64;
        for _ in 0..7 {
            let start = std::time::Instant::now();
            for _ in 0..lookup_repetitions {
                for id in 0..row_count as u64 {
                    lookup_checksum = lookup_checksum.wrapping_add(
                        cache.get(&table, id).unwrap()[id as usize % ROW_BYTES] as u64,
                    );
                }
            }
            lookup_rounds.push(
                start.elapsed().as_nanos() / (row_count as u128 * lookup_repetitions as u128),
            );
        }
        lookup_rounds.sort_unstable();
        let lookup_p50_ns_per_row = lookup_rounds[lookup_rounds.len() / 2];
        let lookup_ids: Vec<u64> = (0..row_count as u64).collect();
        let mut batch_lookup_rounds = Vec::with_capacity(7);
        for _ in 0..7 {
            let start = std::time::Instant::now();
            for _ in 0..lookup_repetitions {
                cache
                    .for_each_lookup(&table, &lookup_ids, |_, row| {
                        lookup_checksum = lookup_checksum
                            .wrapping_add(row.expect("profiled batch row is resident")[0] as u64);
                        Ok::<(), ()>(())
                    })
                    .unwrap();
            }
            batch_lookup_rounds.push(
                start.elapsed().as_nanos() / (row_count as u128 * lookup_repetitions as u128),
            );
        }
        batch_lookup_rounds.sort_unstable();
        let batch_lookup_p50_ns_per_row = batch_lookup_rounds[batch_lookup_rounds.len() / 2];
        assert!(lookup_checksum > 0);
        let after_kib = current_rss_kib();
        assert_eq!(cache.len(), row_count);
        assert_eq!(cache.resident_bytes(), row_count * ROW_BYTES);
        assert!(cache.resident_bytes() <= row_count * ROW_BYTES);
        println!(
            "A70_CACHE_PROFILE {{\"rows\":{row_count},\"payload_bytes\":{},\"rss_before_kib\":{before_kib},\"rss_after_kib\":{after_kib},\"rss_delta_kib\":{},\"insert_ns_per_row\":{insert_ns_per_row},\"lookup_repetitions_per_round\":{lookup_repetitions},\"lookup_p50_ns_per_row\":{lookup_p50_ns_per_row},\"batch_lookup_p50_ns_per_row\":{batch_lookup_p50_ns_per_row}}}",
            cache.resident_bytes(), after_kib.saturating_sub(before_kib)
        );
        if let Ok(limit) = std::env::var("VF_CACHE_RSS_MAX_DELTA_KIB") {
            let limit: u64 = limit.parse().expect("valid RSS delta limit");
            assert!(
                after_kib.saturating_sub(before_kib) <= limit,
                "cache RSS delta exceeds the experiment budget"
            );
        }
    }

    /// Fresh-process paired benchmark for the production scalar and contiguous batch APIs.
    #[test]
    #[ignore = "explicit cold-insert CPU/RSS profile; set VF_CACHE_PROFILE_ROWS and VF_CACHE_INSERT_VARIANT"]
    fn engram_cache_insert_batch_profile() {
        let row_count: usize = std::env::var("VF_CACHE_PROFILE_ROWS")
            .expect("set VF_CACHE_PROFILE_ROWS")
            .parse()
            .unwrap();
        assert!(row_count > 0 && row_count <= u32::MAX as usize);
        let variant =
            std::env::var("VF_CACHE_INSERT_VARIANT").expect("set VF_CACHE_INSERT_VARIANT");
        assert!(variant == "scalar" || variant == "batch");
        let table = TableIdentity {
            model_revision: "a72-production-insert-profile-v1".into(),
            table_sha256: "51".repeat(32),
            tokenizer_map_sha256: "62".repeat(32),
            layer: 1,
            epoch: 1,
            num_embeddings: row_count as u64,
        };
        let row_ids: Vec<u64> = (0..row_count as u64).collect();
        let payload: Vec<u8> = row_ids
            .iter()
            .flat_map(|&id| (0..ROW_BYTES).map(move |col| id.wrapping_add(col as u64) as u8))
            .collect();
        let before_kib = current_rss_kib();
        let mut cache = EngramRowCache::new(row_count * ROW_BYTES);
        let started = std::time::Instant::now();
        match variant.as_str() {
            "scalar" => {
                for (index, &id) in row_ids.iter().enumerate() {
                    let start = index * ROW_BYTES;
                    assert!(cache
                        .insert(&table, id, &payload[start..start + ROW_BYTES])
                        .unwrap());
                }
            }
            "batch" => cache.insert_many(&table, &row_ids, &payload).unwrap(),
            _ => unreachable!(),
        }
        let insert_ns_per_row = started.elapsed().as_nanos() / row_count as u128;
        let after_kib = current_rss_kib();
        let mut checksum = 0u64;
        for &id in &row_ids {
            let row = cache
                .get(&table, id)
                .expect("every profile row was inserted");
            assert_eq!(
                row,
                &payload[id as usize * ROW_BYTES..(id as usize + 1) * ROW_BYTES]
            );
            checksum = checksum.wrapping_add(row[0] as u64);
        }
        assert_eq!(cache.len(), row_count);
        assert_eq!(cache.resident_bytes(), row_count * ROW_BYTES);
        assert!(checksum > 0);
        println!(
            "A72_INSERT_PROFILE {{\"variant\":\"{variant}\",\"rows\":{row_count},\"payload_bytes\":{},\"insert_ns_per_row\":{insert_ns_per_row},\"rss_delta_kib\":{},\"checksum\":{checksum}}}",
            cache.resident_bytes(), after_kib.saturating_sub(before_kib)
        );
    }
}
