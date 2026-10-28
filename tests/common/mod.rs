#![allow(dead_code)]
use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use voltafabric::{moe::*, planner::*, topology::*, traffic::*};

/// Per-invocation directory for executable scripts used by process tests.
/// CARGO_TARGET_TMPDIR is shared by test binaries, so fixed script names can
/// collide when Cargo or the test harness runs workers concurrently.
pub struct UniqueScriptDir(PathBuf);

impl UniqueScriptDir {
    pub fn new() -> io::Result<Self> {
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        let root = Path::new(env!("CARGO_TARGET_TMPDIR"));
        std::fs::create_dir_all(root)?;
        loop {
            let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
            let path = root.join(format!("vf-script-{}-{id}", std::process::id()));
            match std::fs::create_dir(&path) {
                Ok(()) => return Ok(Self(path)),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error),
            }
        }
    }

    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for UniqueScriptDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

pub fn problem() -> Problem {
    let experts = [10, 25, 99]
        .into_iter()
        .enumerate()
        .map(|(i, id)| Expert {
            id,
            gate: vec![0.2 + i as f64 * 0.1, -0.3, 0.5, 0.7],
            up: vec![0.4, 0.8, -0.2, 0.6],
            down: vec![0.3, -0.5, 0.1, 0.9],
        })
        .collect();
    Problem {
        model: Model {
            hidden: 2,
            intermediate: 2,
            experts,
        },
        batch: Batch {
            inputs: vec![vec![1.0, -2.0], vec![0.25, 0.7]],
            routes: vec![
                vec![
                    Choice {
                        expert: 10,
                        weight: 0.2,
                    },
                    Choice {
                        expert: 99,
                        weight: 0.8,
                    },
                ],
                vec![
                    Choice {
                        expert: 25,
                        weight: 0.0,
                    },
                    Choice {
                        expert: 10,
                        weight: 0.3,
                    },
                ],
            ],
        },
        registry: Registry {
            epoch: 7,
            copies: vec![
                Replica {
                    expert: 10,
                    rank: 2,
                    local: 0,
                },
                Replica {
                    expert: 25,
                    rank: 9,
                    local: 0,
                },
                Replica {
                    expert: 99,
                    rank: 2,
                    local: 1,
                },
            ],
        },
        stamp: Stamp {
            run_id: "run-1".into(),
            model_revision: "fixture".into(),
            plan_id: "plan-1".into(),
            layer: 0,
            batch: 3,
            epoch: 7,
        },
        preferred_owners: BTreeMap::new(),
    }
}
pub fn topology() -> Topology {
    Topology {
        schema_version: 1,
        origin: Origin::Synthetic,
        source_note: "synthetic test only".into(),
        devices: vec![
            Device {
                rank: 2,
                host: "a".into(),
                capacity_bytes: 100,
                reserved_bytes: 10,
            },
            Device {
                rank: 9,
                host: "b".into(),
                capacity_bytes: 100,
                reserved_bytes: 10,
            },
        ],
        links: vec![
            Link {
                id: "a-b".into(),
                src: 2,
                dst: 9,
                bandwidth_bytes_per_s: 1e9,
                latency_us: 5.0,
                enabled: true,
                transport: Transport::Synthetic,
                pcie: None,
                resources: vec!["uplink".into()],
                cost_provenance: Default::default(),
            },
            Link {
                id: "b-a".into(),
                src: 9,
                dst: 2,
                bandwidth_bytes_per_s: 1e9,
                latency_us: 8.0,
                enabled: true,
                transport: Transport::Synthetic,
                pcie: None,
                resources: vec!["uplink".into()],
                cost_provenance: Default::default(),
            },
        ],
    }
}
pub fn placement() -> PlacementProblem {
    PlacementProblem {
        topology: topology(),
        experts: vec![
            ExpertProfile {
                id: 10,
                resident_bytes: 40,
                compute_us: 2.0,
                cost_provenance: Default::default(),
            },
            ExpertProfile {
                id: 25,
                resident_bytes: 40,
                compute_us: 2.0,
                cost_provenance: Default::default(),
            },
            ExpertProfile {
                id: 99,
                resident_bytes: 40,
                compute_us: 2.0,
                cost_provenance: Default::default(),
            },
        ],
        demands: vec![
            Demand {
                source_rank: 2,
                experts: vec![10, 25],
                frequency: 10.0,
            },
            Demand {
                source_rank: 9,
                experts: vec![99],
                frequency: 5.0,
            },
        ],
        hidden: 4096,
        wire: WireFormat::default(),
        max_iterations: 32,
        wire_policy: WirePolicy::Adaptive,
        max_work_units: 1_000_000,
    }
}
