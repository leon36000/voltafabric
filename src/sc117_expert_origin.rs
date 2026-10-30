//! Compare SC117 native experts against the locally observed GGUF source slices.
//! This establishes local consistency, not model publisher authentication.
use crate::{invalid, sha256, strata_native_layout::NativeLayout, Result};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Seek, SeekFrom},
};

const INDEX_SHA: &str = "adba4968661f3d59f749a676a87948e29236c81997f649f5526daf71148edc24";
const SOURCE_BYTES: u64 = 55_063_448_064;
const MAX_BYTES: u64 = 4 * 1024 * 1024;

type Identity = (u64, u64, u64, i64, i64, i64, i64);
#[cfg(unix)]
fn identity(file: &File) -> Result<Identity> {
    use std::os::unix::fs::MetadataExt;
    let m = file.metadata().map_err(|e| invalid(e.to_string()))?;
    if !m.is_file() {
        return Err(invalid("expert source not regular"));
    }
    Ok((
        m.dev(),
        m.ino(),
        m.len(),
        m.mtime(),
        m.mtime_nsec(),
        m.ctime(),
        m.ctime_nsec(),
    ))
}
fn lengths(gu: u32, down: u32) -> Result<(u64, u64)> {
    let block = |kind| match kind {
        18 => Ok((256, 98)),
        21 => Ok((256, 110)),
        22 => Ok((256, 82)),
        23 => Ok((256, 136)),
        20 => Ok((32, 18)),
        42 => Ok((64, 18)),
        _ => Err(invalid("unsupported origin quantization")),
    };
    let (ge, gb) = block(gu)?;
    let (de, db) = block(down)?;
    if 2560 % ge != 0 || 640 % de != 0 {
        return Err(invalid("origin block alignment"));
    }
    Ok(((2560 / ge) * gb * 640, (640 / de) * db * 2560))
}
fn slices<R: Read + Seek>(
    reader: &mut R,
    source_bytes: u64,
    offsets: [u64; 3],
    lengths: [u64; 3],
    expert: u64,
    expected: u64,
) -> Result<String> {
    let total = lengths
        .iter()
        .try_fold(0u64, |n, x| n.checked_add(*x))
        .ok_or_else(|| invalid("origin length overflow"))?;
    if total != expected || total > MAX_BYTES || expert >= 512 {
        return Err(invalid("origin expert bounds"));
    }
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(total as usize)
        .map_err(|_| invalid("origin allocation"))?;
    for (base, len) in offsets.into_iter().zip(lengths) {
        let at = expert
            .checked_mul(len)
            .and_then(|n| base.checked_add(n))
            .ok_or_else(|| invalid("origin offset overflow"))?;
        if at.checked_add(len).is_none_or(|end| end > source_bytes) {
            return Err(invalid("origin span exceeds source"));
        }
        reader
            .seek(SeekFrom::Start(at))
            .map_err(|e| invalid(e.to_string()))?;
        let from = bytes.len();
        bytes.resize(from + len as usize, 0);
        reader
            .read_exact(&mut bytes[from..])
            .map_err(|e| invalid(e.to_string()))?;
    }
    Ok(sha256::hex(&bytes))
}
/// Pinned descriptor for the specific observed first SC117 shard. The caller must
/// authenticate model provenance and the index independently of this local check.
pub struct Origin {
    file: File,
    identity: Identity,
    digests: BTreeMap<(usize, u64), String>,
    hits: u64,
    misses: u64,
}
#[cfg(unix)]
impl Origin {
    pub fn open(file: File, layout: &NativeLayout) -> Result<Self> {
        let id = identity(&file)?;
        if id.2 != SOURCE_BYTES || layout.index_sha256() != INDEX_SHA {
            return Err(invalid("origin identity or index"));
        }
        Ok(Self {
            file,
            identity: id,
            digests: BTreeMap::new(),
            hits: 0,
            misses: 0,
        })
    }
    /// Successful lookups only; these counters are not a performance measurement.
    pub fn digest_cache_stats(&self) -> (u64, u64, usize) {
        (self.hits, self.misses, self.digests.len())
    }
    pub fn expert_digest(
        &mut self,
        layout: &NativeLayout,
        layer: usize,
        expert: u64,
    ) -> Result<String> {
        if layout.index_sha256() != INDEX_SHA || identity(&self.file)? != self.identity {
            return Err(invalid("origin descriptor or index changed"));
        }
        if let Some(digest) = self.digests.get(&(layer, expert)) {
            self.hits = self
                .hits
                .checked_add(1)
                .ok_or_else(|| invalid("origin cache hit count overflow"))?;
            return Ok(digest.clone());
        }
        let entry = layout
            .layers()
            .get(layer)
            .ok_or_else(|| invalid("origin layer"))?;
        let span = layout.expert(layer, expert)?;
        let (gu, down) = lengths(entry.gate_up_type, entry.down_type)?;
        let digest = slices(
            &mut self.file,
            SOURCE_BYTES,
            entry.gguf_offsets,
            [gu, gu, down],
            expert,
            span.bytes,
        )?;
        if identity(&self.file)? != self.identity {
            return Err(invalid("origin descriptor changed during read"));
        }
        self.misses = self
            .misses
            .checked_add(1)
            .ok_or_else(|| invalid("origin cache miss count overflow"))?;
        self.digests.insert((layer, expert), digest.clone());
        Ok(digest)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    #[test]
    fn joins_independent_expert_slices_and_rejects_bounds() {
        let data: Vec<u8> = (0..40).collect();
        let mut c = Cursor::new(data.clone());
        let expected = sha256::hex(&[
            data[2], data[3], data[12], data[13], data[22], data[23], data[24],
        ]);
        assert_eq!(
            slices(&mut c, 40, [0, 10, 19], [2, 2, 3], 1, 7).unwrap(),
            expected
        );
        assert!(slices(&mut c, 40, [0, 10, 39], [2, 2, 3], 1, 7).is_err());
        assert!(slices(&mut c, 40, [0, 10, 19], [2, 2, 3], 512, 7).is_err());
        assert!(slices(&mut c, 40, [u64::MAX, 10, 19], [2, 2, 3], 1, 7).is_err());
        assert!(slices(&mut c, 40, [0, 10, 19], [2, 2, 3], 1, 8).is_err());
        assert!(lengths(0, 20).is_err());
    }
    #[test]
    fn observed_format_geometry() {
        assert_eq!(lengths(18, 20).unwrap(), (627_200, 921_600));
        assert_eq!(lengths(22, 42).unwrap(), (524_800, 460_800));
    }

    #[cfg(unix)]
    #[test]
    fn cached_origin_digest_skips_reads_and_refuses_changed_source() {
        use std::os::unix::fs::FileExt;
        use std::time::{SystemTime, UNIX_EPOCH};
        let index = include_str!("../fixtures/strata_sc117_native_experts_v3.txt");
        let layout = NativeLayout::parse_sc117_v3(index, 50_292_326_400).unwrap();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(".runs")
            .join(format!("origin-cache-test-{}-{now}", std::process::id()));
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        // A sparse file provides the required offsets; it is unlinked before
        // use so the test cannot leave a large apparent file behind.
        file.set_len(SOURCE_BYTES).unwrap();
        std::fs::remove_file(&path).unwrap();
        let writer = file.try_clone().unwrap();
        let mut origin = Origin::open(file, &layout).unwrap();
        let expected = "50eff9144c635807dfdc0cafd1c3bb978d0202d537ae892ec8ebed90dc13e172";
        assert_eq!(origin.expert_digest(&layout, 0, 0).unwrap(), expected);
        assert_eq!(origin.digest_cache_stats(), (0, 1, 1));
        origin.file.seek(SeekFrom::Start(0)).unwrap();
        assert_eq!(origin.expert_digest(&layout, 0, 0).unwrap(), expected);
        assert_eq!(origin.file.stream_position().unwrap(), 0);
        assert_eq!(origin.digest_cache_stats(), (1, 1, 1));
        assert!(origin.expert_digest(&layout, 48, 0).is_err());
        writer.write_all_at(&[1], 0).unwrap();
        assert!(origin.expert_digest(&layout, 0, 0).is_err());
        assert_eq!(origin.digest_cache_stats(), (1, 1, 1));
    }
}
