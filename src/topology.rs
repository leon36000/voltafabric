use crate::cost::CostProvenance;
use crate::work::WorkBudget;
use crate::{invalid, Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Origin {
    Synthetic,
    ImportedMeasurement,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Device {
    pub rank: u32,
    pub host: String,
    pub capacity_bytes: u64,
    pub reserved_bytes: u64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Transport {
    PcieP2p,
    HostStaged,
    Rdma,
    Synthetic,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PcieLinkProfile {
    pub generation: u8,
    pub lane_width: u8,
}
impl PcieLinkProfile {
    /// PHY-side ceiling from signaling rate and line encoding only. This excludes
    /// packet/FLIT, FEC, software, and transfer-engine overhead; it is not throughput.
    pub fn theoretical_ceiling_bytes_per_s(&self) -> Result<u64> {
        if !(1..=7).contains(&self.generation)
            || !matches!(self.lane_width, 1 | 2 | 4 | 8 | 16 | 32)
        {
            return Err(invalid("unsupported PCIe generation or lane width"));
        }
        let (rate_per_lane, encoded_bits, raw_bits) = match self.generation {
            1 => (2_500_000_000_u128, 8_u128, 10_u128),
            2 => (5_000_000_000, 8, 10),
            3 => (8_000_000_000, 128, 130),
            4 => (16_000_000_000, 128, 130),
            5 => (32_000_000_000, 128, 130),
            // PCI-SIG's Gen6/7 data rates are already the raw bit rates per lane.
            // Applying PAM4's two bits per symbol again would double-count it.
            // These ceilings still exclude FLIT/FEC overhead.
            6 => (64_000_000_000, 1, 1),
            7 => (128_000_000_000, 1, 1),
            _ => unreachable!("generation range checked above"),
        };
        let bytes_per_s = rate_per_lane * encoded_bits * u128::from(self.lane_width) / raw_bits / 8;
        u64::try_from(bytes_per_s).map_err(|_| invalid("PCIe PHY ceiling overflow"))
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Link {
    pub id: String,
    pub src: u32,
    pub dst: u32,
    pub bandwidth_bytes_per_s: f64,
    pub latency_us: f64,
    pub enabled: bool,
    pub transport: Transport,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pcie: Option<PcieLinkProfile>,
    pub resources: Vec<String>,
    #[serde(default)]
    pub cost_provenance: CostProvenance,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Topology {
    pub schema_version: u32,
    pub origin: Origin,
    pub source_note: String,
    pub devices: Vec<Device>,
    pub links: Vec<Link>,
}
impl Topology {
    pub fn validate(&self) -> Result<()> {
        validate_with_budget(self, &mut WorkBudget::standard())
    }
    /// Select an enabled directed link. No fabricated reverse or multi-hop path.
    pub fn transfer(&self, src: u32, dst: u32, bytes: u64) -> Result<(f64, Vec<String>)> {
        let mut budget = WorkBudget::standard();
        PreparedTopology::new(self, &mut budget)?.transfer(src, dst, bytes, &mut budget)
    }
}
fn validate_with_budget(t: &Topology, budget: &mut WorkBudget) -> Result<()> {
    budget.charge(1)?;
    if t.schema_version != 1
        || t.source_note.len() > 4096
        || t.source_note.trim().is_empty()
        || t.devices.is_empty()
        || t.devices.len() > 256
        || t.links.len() > 65536
    {
        return Err(invalid(
            "topology version, provenance, or bounded device/link count",
        ));
    }
    let mut ranks = BTreeSet::new();
    let mut hosts_by_rank = BTreeMap::new();
    for d in &t.devices {
        budget.charge(1)?;
        if !ranks.insert(d.rank)
            || d.host.len() > 256
            || d.host.trim().is_empty()
            || d.capacity_bytes == 0
            || d.reserved_bytes > d.capacity_bytes
        {
            return Err(invalid("duplicate rank or invalid device capacity/host"));
        }
        hosts_by_rank.insert(d.rank, d.host.as_str());
    }
    let mut ids = BTreeSet::new();
    for l in &t.links {
        budget.charge(1)?;
        l.cost_provenance.validate()?;
        if l.id.is_empty()
            || l.id.len() > 128
            || !ids.insert(&l.id)
            || l.src == l.dst
            || !ranks.contains(&l.src)
            || !ranks.contains(&l.dst)
            || !l.bandwidth_bytes_per_s.is_finite()
            || l.bandwidth_bytes_per_s <= 0.0
            || !l.latency_us.is_finite()
            || l.latency_us < 0.0
        {
            return Err(invalid("invalid directed link, timing or identity"));
        }
        // PCIe peer-to-peer is a same-host fabric edge; inter-host transfers must use
        // a transport that actually crosses the host boundary (for example RDMA).
        if matches!(&l.transport, Transport::PcieP2p)
            && hosts_by_rank.get(&l.src) != hosts_by_rank.get(&l.dst)
        {
            return Err(invalid(
                "PCIe P2P link endpoints must belong to the same host",
            ));
        }
        match (&l.transport, &l.pcie) {
            (Transport::PcieP2p, Some(profile)) => {
                let ceiling = profile.theoretical_ceiling_bytes_per_s()?;
                if l.bandwidth_bytes_per_s > ceiling as f64 {
                    return Err(invalid(
                        "PCIe bandwidth exceeds the theoretical PHY ceiling",
                    ));
                }
            }
            (_, Some(_)) => {
                return Err(invalid("PCIe profile requires PCIe P2P transport"));
            }
            (_, None) => {}
        }
        budget.charge(l.resources.len())?;
        let mut unique = BTreeSet::new();
        for x in &l.resources {
            if x.is_empty() || x.len() > 256 || x.starts_with("vf:") || !unique.insert(x) {
                return Err(invalid("invalid resource identity (vf: reserved)"));
            }
        }
    }
    Ok(())
}
/// Validated once per planner invocation. Parallel links remain selectable by byte count.
pub(crate) struct PreparedTopology<'a> {
    ranks: BTreeSet<u32>,
    links: BTreeMap<(u32, u32), Vec<&'a Link>>,
}
impl<'a> PreparedTopology<'a> {
    pub(crate) fn new(t: &'a Topology, budget: &mut WorkBudget) -> Result<Self> {
        validate_with_budget(t, budget)?;
        let mut ranks = BTreeSet::new();
        for d in &t.devices {
            budget.charge(1)?;
            ranks.insert(d.rank);
        }
        let mut links: BTreeMap<_, Vec<_>> = BTreeMap::new();
        for l in &t.links {
            budget.charge(1)?;
            if l.enabled {
                links.entry((l.src, l.dst)).or_default().push(l);
            }
        }
        Ok(Self { ranks, links })
    }
    pub(crate) fn has_rank(&self, rank: u32) -> bool {
        self.ranks.contains(&rank)
    }
    pub(crate) fn transfer(
        &self,
        src: u32,
        dst: u32,
        bytes: u64,
        budget: &mut WorkBudget,
    ) -> Result<(f64, Vec<String>)> {
        budget.charge(1)?;
        if !self.has_rank(src) || !self.has_rank(dst) {
            return Err(invalid("unknown transfer endpoint"));
        }
        if src == dst || bytes == 0 {
            return Ok((0.0, Vec::new()));
        }
        let mut best: Option<(&Link, f64)> = None;
        let mut saw_overflow = false;
        if let Some(links) = self.links.get(&(src, dst)) {
            for l in links {
                budget.charge(1)?;
                let cost = l.latency_us + (bytes as f64 / l.bandwidth_bytes_per_s) * 1e6;
                if !cost.is_finite() {
                    saw_overflow = true;
                    continue;
                }
                if best.is_none_or(|(old, t)| cost < t || (cost == t && l.id < old.id)) {
                    best = Some((l, cost));
                }
            }
        }
        let (l, cost) = match best {
            Some(best) => best,
            None if saw_overflow => {
                return Err(Error::new(
                    "TRANSFER_TIME_OVERFLOW",
                    format!("all enabled paths overflow transfer time for {src}->{dst}"),
                ));
            }
            None => {
                return Err(Error::new(
                    "NO_PATH",
                    format!("no qualified enabled path {src}->{dst}"),
                ));
            }
        };
        budget.charge(l.resources.len() + 1)?;
        let mut resources = l.resources.clone();
        resources.push(format!("vf:link:{}", l.id));
        Ok((cost, resources))
    }
}
