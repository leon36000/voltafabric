use crate::work::WorkBudget;
use crate::{invalid, Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Job {
    pub id: String,
    pub duration_us: f64,
    pub depends_on: Vec<String>,
    pub resources: Vec<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Execution {
    pub id: String,
    pub start_us: f64,
    pub end_us: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Schedule {
    pub result_kind: String,
    pub total_us: f64,
    pub jobs: Vec<Execution>,
}
/// The refusal of a schedule whose finish time overflows f64 (each duration is finite).
pub(crate) const TIME_OVERFLOW: &str = "schedule time overflow";
/// Deterministic list scheduling with exclusive resources, not a bandwidth simulator.
pub fn schedule(jobs: &[Job]) -> Result<Schedule> {
    schedule_with_budget(jobs, &mut WorkBudget::standard())
}
pub(crate) fn schedule_with_budget(jobs: &[Job], budget: &mut WorkBudget) -> Result<Schedule> {
    if jobs.len() > 4096 {
        return Err(invalid("at most 4096 jobs"));
    }
    let mut ids = BTreeSet::new();
    for j in jobs {
        budget.charge(1)?;
        if j.id.is_empty()
            || j.id.len() > 256
            || !ids.insert(&j.id)
            || !j.duration_us.is_finite()
            || j.duration_us < 0.0
        {
            return Err(invalid(
                "duplicate/empty/oversized job ID or invalid duration",
            ));
        }
        for list in [&j.depends_on, &j.resources] {
            budget.charge(list.len())?;
            let mut unique = BTreeSet::new();
            for x in list {
                if x.is_empty() || x.len() > 256 || !unique.insert(x) {
                    return Err(invalid("empty/oversized/duplicate dependency or resource"));
                }
            }
        }
    }
    for j in jobs {
        budget.charge(1)?;
        budget.charge(j.depends_on.len())?;
        if j.depends_on.iter().any(|d| !ids.contains(d)) {
            return Err(invalid("missing dependency"));
        }
    }
    let mut end: BTreeMap<String, f64> = BTreeMap::new();
    let mut busy: BTreeMap<String, f64> = BTreeMap::new();
    let mut out = Vec::new();
    while out.len() < jobs.len() {
        let mut best: Option<(&Job, f64)> = None;
        for j in jobs {
            // Charge even completed and not-ready jobs; this loop is quadratic.
            budget.charge(1)?;
            if end.contains_key(&j.id) {
                continue;
            }
            budget.charge(j.depends_on.len())?;
            if !j.depends_on.iter().all(|d| end.contains_key(d)) {
                continue;
            }
            let mut start = 0.0_f64;
            budget.charge(j.depends_on.len())?;
            for d in &j.depends_on {
                start = start.max(end[d]);
            }
            budget.charge(j.resources.len())?;
            for r in &j.resources {
                start = start.max(*busy.get(r).unwrap_or(&0.0));
            }
            if best
                .as_ref()
                .is_none_or(|(b, t)| start < *t || (start == *t && j.id < b.id))
            {
                best = Some((j, start));
            }
        }
        let (j, start) = best.ok_or_else(|| Error::new("DEPENDENCY_CYCLE", "no ready job"))?;
        let finish = start + j.duration_us;
        if !finish.is_finite() {
            return Err(invalid(TIME_OVERFLOW));
        }
        budget.charge(j.resources.len())?;
        for r in &j.resources {
            busy.insert(r.clone(), finish);
        }
        end.insert(j.id.clone(), finish);
        out.push(Execution {
            id: j.id.clone(),
            start_us: start,
            end_us: finish,
        });
    }
    budget.charge(end.len())?;
    Ok(Schedule {
        result_kind: "analytical_schedule_not_hardware_measurement".into(),
        total_us: end.values().copied().fold(0.0, f64::max),
        jobs: out,
    })
}

#[cfg(test)]
mod budget_tests {
    use super::*;
    #[test]
    fn completed_and_not_ready_jobs_still_consume_scan_budget() {
        let jobs: Vec<_> = (0..128)
            .map(|i| Job {
                id: format!("job:{i:03}"),
                duration_us: 1.0,
                depends_on: if i == 0 {
                    vec![]
                } else {
                    vec![format!("job:{:03}", i - 1)]
                },
                resources: vec![],
            })
            .collect();
        assert_eq!(
            schedule_with_budget(&jobs, &mut WorkBudget::new(1000).unwrap())
                .unwrap_err()
                .code,
            "WORK_BUDGET_EXCEEDED"
        );
        assert_eq!(schedule(&jobs).unwrap().total_us, 128.0);
    }
    #[test]
    fn resources_are_charged_before_validation_allocation() {
        let jobs = [Job {
            id: "job".into(),
            duration_us: 1.0,
            depends_on: vec![],
            resources: (0..1000).map(|i| format!("r:{i}")).collect(),
        }];
        assert_eq!(
            schedule_with_budget(&jobs, &mut WorkBudget::new(100).unwrap())
                .unwrap_err()
                .code,
            "WORK_BUDGET_EXCEEDED"
        );
    }
    #[test]
    fn final_makespan_scan_consumes_the_remaining_budget() {
        let jobs = [Job {
            id: "single".into(),
            duration_us: 1.0,
            depends_on: vec![],
            resources: vec![],
        }];
        assert_eq!(
            schedule_with_budget(&jobs, &mut WorkBudget::new(3).unwrap())
                .unwrap_err()
                .code,
            "WORK_BUDGET_EXCEEDED"
        );
    }
}
