//! Bounded local SC117 admission source. It does not authenticate the publisher,
//! measure process RSS, evict residents, or provide a hardware lease.
use crate::{
    invalid,
    moe::{Registry, Stamp},
    native_ggml_cpu::Binding,
    sc117_expert_origin::Origin,
    sc117_layer_chain::{ExpertAdmission, MissingExpertSource},
    strata_native_layout::NativeLayout,
    Result,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
};

/// Every already-loaded resident must be supplied exactly once. The caller is
/// responsible for reserving the initial budget before loading those residents.
#[derive(Clone, Copy, Debug)]
pub struct Seed {
    pub layer: u32,
    pub expert: u32,
    pub rank: u32,
    pub local: u32,
}
#[derive(Clone, Debug)]
pub struct Identity {
    pub run_id: String,
    pub model_revision: String,
    pub plan_id: String,
    pub epoch: u64,
}

/// Accounting and identity for a single segment source.
pub struct Config<'a> {
    pub identity: Identity,
    pub ranks: Vec<u32>,
    pub seeds: &'a [Seed],
    pub limit: u64,
}

fn charge(used: u64, bytes: u64, limit: u64) -> Result<u64> {
    let next = used
        .checked_add(bytes)
        .ok_or_else(|| invalid("expert payload budget overflow"))?;
    if next > limit {
        return Err(invalid("expert payload budget exceeded"));
    }
    Ok(next)
}
fn validate_rank_limits(ranks: &[u32], limit: u64, caps: &BTreeMap<u32, u64>) -> Result<()> {
    if caps.len() != ranks.len()
        || ranks
            .iter()
            .any(|r| caps.get(r).is_none_or(|&v| v == 0 || v > limit))
    {
        return Err(invalid("rank payload budget identities"));
    }
    Ok(())
}

struct RankCapacity<'a> {
    limits: &'a BTreeMap<u32, u64>,
    used: &'a BTreeMap<u32, u64>,
    bytes: u64,
}
fn placement(
    registry: &Registry,
    order: &BTreeMap<u32, Vec<u32>>,
    ranks: &[u32],
    epoch: u64,
    expert: u32,
    capacity: Option<RankCapacity<'_>>,
) -> Result<(u32, u32)> {
    if registry.epoch != epoch || order.len() != ranks.len() || ranks.is_empty() {
        return Err(invalid("expert placement epoch or rank set"));
    }
    let mut pairs = BTreeSet::new();
    let mut experts = BTreeSet::new();
    for copy in &registry.copies {
        if !ranks.contains(&copy.rank)
            || !pairs.insert((copy.rank, copy.local))
            || !experts.insert((copy.rank, copy.expert))
            || copy.expert == expert
        {
            return Err(invalid("expert placement duplicate or unknown rank"));
        }
    }
    for &rank in ranks {
        let locals = order
            .get(&rank)
            .ok_or_else(|| invalid("expert rank absent"))?;
        let actual: BTreeSet<_> = locals.iter().copied().collect();
        let expected: BTreeSet<_> = registry
            .copies
            .iter()
            .filter(|c| c.rank == rank)
            .map(|c| c.local)
            .collect();
        if actual.len() != locals.len() || actual != expected || locals.is_empty() {
            return Err(invalid("expert resident order differs from registry"));
        }
    }
    let rank = if let Some(RankCapacity {
        limits,
        used,
        bytes,
    }) = capacity
    {
        *ranks
            .iter()
            .filter(|&&r| {
                limits
                    .get(&r)
                    .zip(used.get(&r))
                    .is_some_and(|(&limit, &amount)| {
                        limit > 0 && amount.checked_add(bytes).is_some_and(|next| next <= limit)
                    })
            })
            .min_by_key(|&&r| {
                (
                    u128::from(used[&r]) * 1_000_000 / u128::from(limits[&r]),
                    order[&r].len(),
                    r,
                )
            })
            .ok_or_else(|| invalid("no rank has expert payload capacity"))?
    } else {
        *ranks
            .iter()
            .min_by_key(|&&r| (order[&r].len(), r))
            .ok_or_else(|| invalid("expert rank"))?
    };
    let local = order[&rank]
        .iter()
        .copied()
        .max()
        .and_then(|x| x.checked_add(1))
        .ok_or_else(|| invalid("expert local overflow"))?;
    Ok((rank, local))
}

struct StickyInputs<'a> {
    ranks: &'a [u32],
    order: &'a BTreeMap<u32, Vec<u32>>,
    used: &'a BTreeMap<u32, u64>,
    caps: Option<&'a BTreeMap<u32, u64>>,
    limit: u64,
    bytes: u64,
}

// Pin an already-seen (layer, expert) to the original producer rank. The
// caller first runs placement() to validate the complete current registry.
fn sticky_placement(
    pinned: Option<u32>,
    suggested: (u32, u32),
    ctx: &StickyInputs<'_>,
) -> Result<(u32, u32)> {
    let Some(rank) = pinned else {
        return Ok(suggested);
    };
    if !ctx.ranks.contains(&rank) {
        return Err(invalid("sticky expert rank absent"));
    }
    let used = *ctx
        .used
        .get(&rank)
        .ok_or_else(|| invalid("sticky rank accounting absent"))?;
    let cap = match ctx.caps {
        Some(caps) => *caps
            .get(&rank)
            .ok_or_else(|| invalid("sticky rank cap absent"))?,
        None => ctx.limit,
    };
    charge(used, ctx.bytes, cap)?;
    let local = ctx
        .order
        .get(&rank)
        .and_then(|v| v.iter().copied().max())
        .and_then(|v| v.checked_add(1))
        .ok_or_else(|| invalid("sticky rank local overflow"))?;
    Ok((rank, local))
}

fn matches_accounted(
    layer: u32,
    registry: &Registry,
    known: &BTreeMap<(u32, u32), (u32, u32)>,
) -> bool {
    let expected: BTreeMap<_, _> = known
        .iter()
        .filter(|((l, _), _)| *l == layer)
        .map(|((_, e), p)| (*e, *p))
        .collect();
    let actual: BTreeMap<_, _> = registry
        .copies
        .iter()
        .map(|r| (r.expert, (r.rank, r.local)))
        .collect();
    actual.len() == registry.copies.len() && actual == expected
}

fn release_accounted(
    known: &mut BTreeMap<(u32, u32), (u32, u32)>,
    used: &mut u64,
    rank_used: &mut BTreeMap<u32, u64>,
    binding: &Binding,
    bytes: u64,
    expected: u64,
) -> Result<()> {
    if known.get(&(binding.layer, binding.expert)) != Some(&(binding.rank, binding.local))
        || bytes != expected
    {
        return Err(invalid("eviction accounting changed"));
    }
    let next = used
        .checked_sub(bytes)
        .ok_or_else(|| invalid("eviction accounting underflow"))?;
    let next_rank = rank_used
        .get(&binding.rank)
        .and_then(|v| v.checked_sub(bytes))
        .ok_or_else(|| invalid("rank eviction accounting underflow"))?;
    known.remove(&(binding.layer, binding.expert));
    *used = next;
    rank_used.insert(binding.rank, next_rank);
    Ok(())
}

/// One instance accounts for all seeds and later admissions in its segment.
/// It assumes exclusive control of the described sessions. Accounting is
/// conservative: a returned payload is charged even if its append is uncertain.
pub struct Source<'a> {
    pack: &'a mut File,
    layout: &'a NativeLayout,
    index: &'a str,
    origin: Origin,
    identity: Identity,
    ranks: Vec<u32>,
    limit: u64,
    used: u64,
    rank_used: BTreeMap<u32, u64>,
    rank_limits: Option<BTreeMap<u32, u64>>,
    known: BTreeMap<(u32, u32), (u32, u32)>,
    sticky: Option<BTreeMap<(u32, u32), u32>>,
    ever_evicted: bool,
    admissions: Vec<(u32, u32, u32, u32, String)>,
}
impl<'a> Source<'a> {
    pub fn new(
        pack: &'a mut File,
        layout: &'a NativeLayout,
        index: &'a str,
        origin: Origin,
        config: Config<'_>,
    ) -> Result<Self> {
        Self::build(pack, layout, index, origin, config, None)
    }

    /// Apply independently admitted expert-payload caps for each rank. These
    /// caps are not host-memory measurements, topology proof or hardware leases.
    pub fn new_with_rank_limits(
        pack: &'a mut File,
        layout: &'a NativeLayout,
        index: &'a str,
        origin: Origin,
        config: Config<'_>,
        rank_limits: BTreeMap<u32, u64>,
    ) -> Result<Self> {
        Self::build(pack, layout, index, origin, config, Some(rank_limits))
    }

    fn build(
        pack: &'a mut File,
        layout: &'a NativeLayout,
        index: &'a str,
        origin: Origin,
        config: Config<'_>,
        rank_limits: Option<BTreeMap<u32, u64>>,
    ) -> Result<Self> {
        let Config {
            identity,
            mut ranks,
            seeds,
            limit,
        } = config;
        if identity.run_id.is_empty()
            || identity.model_revision.is_empty()
            || identity.plan_id.is_empty()
            || identity.epoch == 0
            || limit == 0
            || crate::sha256::hex(index.as_bytes()) != layout.index_sha256()
        {
            return Err(invalid("expert source identity, index or budget"));
        }
        ranks.sort_unstable();
        if ranks.is_empty() || ranks.windows(2).any(|p| p[0] == p[1]) {
            return Err(invalid("expert source ranks"));
        }
        if let Some(caps) = &rank_limits {
            validate_rank_limits(&ranks, limit, caps)?;
        }
        let mut used = 0;
        let mut rank_used: BTreeMap<u32, u64> = ranks.iter().map(|&r| (r, 0)).collect();
        let mut known = BTreeMap::new();
        let mut slots = BTreeSet::new();
        for s in seeds {
            if !ranks.contains(&s.rank)
                || known
                    .insert((s.layer, s.expert), (s.rank, s.local))
                    .is_some()
                || !slots.insert((s.layer, s.rank, s.local))
            {
                return Err(invalid("expert seed duplicate or rank"));
            }
            let bytes = layout.expert(s.layer as usize, s.expert.into())?.bytes;
            used = charge(used, bytes, limit)?;
            let rank_limit = rank_limits.as_ref().map_or(limit, |caps| caps[&s.rank]);
            rank_used.insert(s.rank, charge(rank_used[&s.rank], bytes, rank_limit)?);
        }
        Ok(Self {
            pack,
            layout,
            index,
            origin,
            identity,
            ranks,
            limit,
            used,
            rank_used,
            rank_limits,
            known,
            sticky: None,
            ever_evicted: false,
            admissions: Vec::new(),
        })
    }
    /// Enable once, before any admissions or evictions, with a plan that binds
    /// this policy. No numerical equivalence between different ranks is implied.
    pub fn enable_rank_affinity(&mut self) -> Result<()> {
        if self.sticky.is_some() || !self.admissions.is_empty() || self.ever_evicted {
            return Err(invalid("rank affinity must precede admission and eviction"));
        }
        self.sticky = Some(
            self.known
                .iter()
                .map(|(&k, &(rank, _))| (k, rank))
                .collect(),
        );
        Ok(())
    }

    pub fn used_bytes(&self) -> u64 {
        self.used
    }
    pub fn limit_bytes(&self) -> u64 {
        self.limit
    }
    pub fn rank_used_bytes(&self) -> &BTreeMap<u32, u64> {
        &self.rank_used
    }
    pub fn rank_limits(&self) -> Option<&BTreeMap<u32, u64>> {
        self.rank_limits.as_ref()
    }
    pub fn admissions(&self) -> &[(u32, u32, u32, u32, String)] {
        &self.admissions
    }
    pub fn origin_digest_cache_stats(&self) -> (u64, u64, usize) {
        self.origin.digest_cache_stats()
    }
    pub(crate) fn resident_bytes(&self, layer: u32, expert: u32) -> Result<u64> {
        if !self.known.contains_key(&(layer, expert)) {
            return Err(invalid("expert not accounted"));
        }
        Ok(self.layout.expert(layer as usize, expert.into())?.bytes)
    }

    /// Reconstruct a locally observed exact binding. This does not authenticate
    /// the publisher, and it does not imply permission to evict the expert.
    pub fn resident_binding(&mut self, layer: u32, expert: u32) -> Result<Binding> {
        let &(rank, local) = self
            .known
            .get(&(layer, expert))
            .ok_or_else(|| invalid("expert not accounted"))?;
        let digest = self
            .origin
            .expert_digest(self.layout, layer as usize, expert.into())?;
        Ok(Binding {
            model_revision: self.identity.model_revision.clone(),
            rank,
            layer,
            expert,
            local,
            epoch: self.identity.epoch,
            index_sha256: self.layout.index_sha256().into(),
            blob_sha256: digest,
        })
    }

    pub(crate) fn validate_eviction(
        &mut self,
        stamp: &Stamp,
        binding: &Binding,
        registry: &Registry,
        order: &BTreeMap<u32, Vec<u32>>,
    ) -> Result<u64> {
        if stamp.run_id != self.identity.run_id
            || stamp.model_revision != self.identity.model_revision
            || stamp.plan_id != self.identity.plan_id
            || stamp.epoch != self.identity.epoch
            || binding.layer != stamp.layer
            || registry.epoch != stamp.epoch
            || !matches_accounted(stamp.layer, registry, &self.known)
            || order.len() != self.ranks.len()
        {
            return Err(invalid("eviction source identity or accounting"));
        }
        for &rank in &self.ranks {
            let locals = order
                .get(&rank)
                .ok_or_else(|| invalid("eviction rank absent"))?;
            let actual: BTreeSet<_> = locals.iter().copied().collect();
            let expected: BTreeSet<_> = registry
                .copies
                .iter()
                .filter(|c| c.rank == rank)
                .map(|c| c.local)
                .collect();
            if actual.len() != locals.len() || actual != expected || locals.is_empty() {
                return Err(invalid("eviction order differs"));
            }
        }
        if order.get(&binding.rank).is_none_or(|r| r.len() <= 1)
            || self.resident_binding(binding.layer, binding.expert)? != *binding
        {
            return Err(invalid("eviction binding or last owner"));
        }
        let bytes = self
            .layout
            .expert(binding.layer as usize, binding.expert.into())?
            .bytes;
        self.used
            .checked_sub(bytes)
            .ok_or_else(|| invalid("eviction accounting underflow"))?;
        self.rank_used
            .get(&binding.rank)
            .and_then(|n| n.checked_sub(bytes))
            .ok_or_else(|| invalid("rank eviction accounting underflow"))?;
        Ok(bytes)
    }

    pub(crate) fn commit_eviction(&mut self, binding: &Binding, bytes: u64) -> Result<()> {
        let expected = self
            .layout
            .expert(binding.layer as usize, binding.expert.into())?
            .bytes;
        release_accounted(
            &mut self.known,
            &mut self.used,
            &mut self.rank_used,
            binding,
            bytes,
            expected,
        )?;
        self.ever_evicted = true;
        Ok(())
    }
}
impl MissingExpertSource for Source<'_> {
    fn load(
        &mut self,
        stamp: &Stamp,
        attempt: u64,
        expert: u32,
        registry: &Registry,
        order: &BTreeMap<u32, Vec<u32>>,
    ) -> Result<ExpertAdmission> {
        if attempt == 0
            || stamp.run_id != self.identity.run_id
            || stamp.model_revision != self.identity.model_revision
            || stamp.plan_id != self.identity.plan_id
            || stamp.epoch != self.identity.epoch
            || self.known.contains_key(&(stamp.layer, expert))
        {
            return Err(invalid("expert source identity or duplicate"));
        }
        if !matches_accounted(stamp.layer, registry, &self.known) {
            return Err(invalid("expert registry differs from accounted residents"));
        }
        let span = self.layout.expert(stamp.layer as usize, expert.into())?;
        let next = charge(self.used, span.bytes, self.limit)?;
        let capacity = self.rank_limits.as_ref().map(|caps| RankCapacity {
            limits: caps,
            used: &self.rank_used,
            bytes: span.bytes,
        });
        let suggested = placement(registry, order, &self.ranks, stamp.epoch, expert, capacity)?;
        let pinned = self
            .sticky
            .as_ref()
            .and_then(|m| m.get(&(stamp.layer, expert)).copied());
        let (rank, local) = sticky_placement(
            pinned,
            suggested,
            &StickyInputs {
                ranks: &self.ranks,
                order,
                used: &self.rank_used,
                caps: self.rank_limits.as_ref(),
                limit: self.limit,
                bytes: span.bytes,
            },
        )?;
        let rank_limit = self
            .rank_limits
            .as_ref()
            .map_or(self.limit, |caps| caps[&rank]);
        let next_rank = charge(self.rank_used[&rank], span.bytes, rank_limit)?;
        let digest = self
            .origin
            .expert_digest(self.layout, stamp.layer as usize, expert.into())?;
        let blob = self.layout.read_verified_expert(
            self.pack,
            stamp.layer as usize,
            expert.into(),
            self.layout.index_sha256(),
            &digest,
        )?;
        let pack_bytes = self
            .pack
            .metadata()
            .map_err(|e| invalid(e.to_string()))?
            .len();
        let binding = Binding {
            model_revision: stamp.model_revision.clone(),
            rank,
            layer: stamp.layer,
            expert,
            local,
            epoch: stamp.epoch,
            index_sha256: self.layout.index_sha256().into(),
            blob_sha256: digest.clone(),
        };
        self.known.insert((stamp.layer, expert), (rank, local));
        if let Some(sticky) = &mut self.sticky {
            sticky.insert((stamp.layer, expert), rank);
        }
        self.used = next;
        self.rank_used.insert(rank, next_rank);
        self.admissions
            .push((stamp.layer, expert, rank, local, digest));
        Ok(ExpertAdmission {
            binding,
            index: self.index.into(),
            pack_bytes,
            bytes: blob.bytes,
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::moe::Replica;
    #[test]
    fn sticky_rank_keeps_producer_and_refuses_capacity_shortfall() {
        let ranks = vec![2, 3];
        let order = BTreeMap::from([(2, vec![0, 2]), (3, vec![1])]);
        let used = BTreeMap::from([(2, 8), (3, 1)]);
        let caps = BTreeMap::from([(2, 10), (3, 20)]);
        let ctx = |bytes| StickyInputs {
            ranks: &ranks,
            order: &order,
            used: &used,
            caps: Some(&caps),
            limit: 100,
            bytes,
        };
        assert_eq!(sticky_placement(Some(2), (3, 2), &ctx(2)).unwrap(), (2, 3));
        assert!(sticky_placement(Some(2), (3, 2), &ctx(3)).is_err());
        assert_eq!(sticky_placement(None, (3, 2), &ctx(3)).unwrap(), (3, 2));
        assert!(sticky_placement(Some(4), (3, 2), &ctx(1)).is_err());
    }

    #[test]
    fn sticky_source_retains_rank_after_accounted_eviction_and_readmission() {
        const INDEX: &str = include_str!("../fixtures/strata_sc117_native_experts_v3.txt");
        let layout = NativeLayout::parse_sc117_v3(INDEX, 50_292_326_400).unwrap();
        // Sparse zero-filled files provide the exact logical lengths without
        // storing model data or launching any workers.
        fn sparse_file(name: &str, len: u64) -> File {
            let path = std::env::temp_dir().join(format!("vf-r576-{}-{name}", std::process::id()));
            let file = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(&path)
                .unwrap();
            std::fs::remove_file(path).unwrap();
            file.set_len(len).unwrap();
            file
        }
        let mut pack = sparse_file("pack", 50_292_326_400);
        let shard = sparse_file("shard", 55_063_448_064);
        let origin = Origin::open(shard, &layout).unwrap();
        let seeds = [
            Seed {
                layer: 0,
                expert: 0,
                rank: 2,
                local: 0,
            },
            Seed {
                layer: 0,
                expert: 1,
                rank: 3,
                local: 0,
            },
        ];
        let identity = Identity {
            run_id: "unit".into(),
            model_revision: "m".into(),
            plan_id: "sticky-plan".into(),
            epoch: 1,
        };
        let mut source = Source::new(
            &mut pack,
            &layout,
            INDEX,
            origin,
            Config {
                identity,
                ranks: vec![2, 3],
                seeds: &seeds,
                limit: 20_000_000,
            },
        )
        .unwrap();
        source.enable_rank_affinity().unwrap();
        assert!(source.enable_rank_affinity().is_err());
        let stamp = Stamp {
            run_id: "unit".into(),
            model_revision: "m".into(),
            plan_id: "sticky-plan".into(),
            layer: 0,
            batch: 0,
            epoch: 1,
        };
        let registry = Registry {
            epoch: 1,
            copies: vec![
                Replica {
                    expert: 0,
                    rank: 2,
                    local: 0,
                },
                Replica {
                    expert: 1,
                    rank: 3,
                    local: 0,
                },
            ],
        };
        let order = BTreeMap::from([(2, vec![0]), (3, vec![0])]);
        let first = source.load(&stamp, 1, 2, &registry, &order).unwrap();
        assert_eq!((first.binding.rank, first.binding.local), (2, 1));
        let size = first.bytes.len() as u64;
        source.commit_eviction(&first.binding, size).unwrap();
        assert_eq!(source.sticky.as_ref().unwrap().get(&(0, 2)), Some(&2));
        // Simulate a different valid capacity profile to make the ordinary
        // balancing suggestion choose rank 3, while rank 2 still fits.
        source.rank_limits = Some(BTreeMap::from([(2, size * 2), (3, size * 9)]));
        let caps = source.rank_limits.as_ref().unwrap();
        let suggested = placement(
            &registry,
            &order,
            &source.ranks,
            1,
            2,
            Some(RankCapacity {
                limits: caps,
                used: &source.rank_used,
                bytes: size,
            }),
        )
        .unwrap();
        assert_eq!(suggested.0, 3);
        let second = source.load(&stamp, 2, 2, &registry, &order).unwrap();
        assert_eq!((second.binding.rank, second.binding.local), (2, 1));
    }

    #[test]
    fn budget_is_bounded_and_overflow_safe() {
        assert_eq!(charge(8, 2, 10).unwrap(), 10);
        assert!(charge(9, 2, 10).is_err());
        assert!(charge(u64::MAX, 1, u64::MAX).is_err());
        assert_eq!(
            charge(9, 2, 10).unwrap_err().to_string(),
            "INVALID_INPUT: expert payload budget exceeded"
        );
    }
    #[test]
    fn acknowledged_release_changes_accounting_once_only() {
        let binding = Binding {
            model_revision: "m".into(),
            rank: 2,
            layer: 4,
            expert: 5,
            local: 3,
            epoch: 1,
            index_sha256: "1".repeat(64),
            blob_sha256: "2".repeat(64),
        };
        let mut known = BTreeMap::from([((4, 5), (2, 3))]);
        let mut used = 10;
        let mut rank_used = BTreeMap::from([(2, 10)]);
        assert!(
            release_accounted(&mut known, &mut used, &mut rank_used, &binding, 11, 11).is_err()
        );
        assert!(release_accounted(&mut known, &mut used, &mut rank_used, &binding, 3, 4).is_err());
        assert_eq!((used, known.len()), (10, 1));
        rank_used.insert(2, 3);
        assert!(release_accounted(&mut known, &mut used, &mut rank_used, &binding, 4, 4).is_err());
        assert_eq!((used, known.len(), rank_used[&2]), (10, 1, 3));
        rank_used.insert(2, 10);
        release_accounted(&mut known, &mut used, &mut rank_used, &binding, 4, 4).unwrap();
        assert_eq!((used, known.len(), rank_used[&2]), (6, 0, 6));
        assert!(release_accounted(&mut known, &mut used, &mut rank_used, &binding, 4, 4).is_err());
        assert_eq!(used, 6);
    }
    #[test]
    fn rank_capabilities_must_match_the_configured_rank_set() {
        assert!(validate_rank_limits(&[2, 3], 100, &BTreeMap::from([(2, 40), (3, 60)])).is_ok());
        assert!(validate_rank_limits(&[2, 3], 100, &BTreeMap::from([(2, 40)])).is_err());
        assert!(validate_rank_limits(&[2, 3], 100, &BTreeMap::from([(2, 40), (3, 0)])).is_err());
        assert!(validate_rank_limits(&[2, 3], 100, &BTreeMap::from([(2, 40), (3, 101)])).is_err());
        assert!(validate_rank_limits(&[2, 3], 100, &BTreeMap::from([(2, 40), (4, 60)])).is_err());
    }

    #[test]
    fn placement_rejects_registry_mismatch_and_balances_deterministically() {
        let r = Registry {
            epoch: 1,
            copies: vec![
                Replica {
                    rank: 2,
                    expert: 0,
                    local: 0,
                },
                Replica {
                    rank: 3,
                    expert: 1,
                    local: 3,
                },
            ],
        };
        let o = BTreeMap::from([(2, vec![0]), (3, vec![3])]);
        assert_eq!(placement(&r, &o, &[2, 3], 1, 9, None).unwrap(), (2, 1));
        let caps = BTreeMap::from([(2, 10), (3, 30)]);
        let used = BTreeMap::from([(2, 9), (3, 10)]);
        assert_eq!(
            placement(
                &r,
                &o,
                &[2, 3],
                1,
                9,
                Some(RankCapacity {
                    limits: &caps,
                    used: &used,
                    bytes: 2
                })
            )
            .unwrap(),
            (3, 4)
        );
        let full = BTreeMap::from([(2, 10), (3, 30)]);
        assert!(placement(
            &r,
            &o,
            &[2, 3],
            1,
            9,
            Some(RankCapacity {
                limits: &caps,
                used: &full,
                bytes: 1
            })
        )
        .is_err());
        let fraction = BTreeMap::from([(2, 5), (3, 10)]);
        assert_eq!(
            placement(
                &r,
                &o,
                &[2, 3],
                1,
                9,
                Some(RankCapacity {
                    limits: &caps,
                    used: &fraction,
                    bytes: 1
                })
            )
            .unwrap(),
            (3, 4)
        );
        let known = BTreeMap::from([((4, 0), (2, 0)), ((4, 1), (3, 3)), ((5, 8), (2, 0))]);
        assert!(matches_accounted(4, &r, &known));
        assert!(!matches_accounted(5, &r, &known));
        assert!(!matches_accounted(4, &r, &BTreeMap::new()));
        assert!(placement(&r, &o, &[2, 3], 2, 9, None).is_err());
        assert!(placement(&r, &o, &[2, 3], 1, 0, None).is_err());
        let bad = BTreeMap::from([(2, vec![0, 0]), (3, vec![3])]);
        assert!(placement(&r, &bad, &[2, 3], 1, 9, None).is_err());
        assert!(placement(&r, &o, &[2, 4], 1, 9, None).is_err());
    }
}
