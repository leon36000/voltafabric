use voltafabric::pipeline::*;

fn problem(costs: &[&[f64]], bytes: &[u64], caps: &[u64], boundaries: &[f64]) -> PipelineProblem {
    PipelineProblem {
        layers: costs
            .iter()
            .zip(bytes)
            .map(|(c, b)| Layer {
                bytes: *b,
                compute_us_by_stage: c.to_vec(),
            })
            .collect(),
        stages: caps
            .iter()
            .zip(boundaries)
            .enumerate()
            .map(|(i, (c, b))| Stage {
                rank: i as u32,
                available_bytes: *c,
                boundary_after_us: *b,
            })
            .collect(),
    }
}

fn direct_sum(p: &PipelineProblem, stage: usize, start: usize, end: usize) -> f64 {
    let mut s = 0.0;
    for l in &p.layers[start..end] {
        s += l.compute_us_by_stage[stage];
    }
    s
}

fn all_cuts(n: usize, m: usize) -> Vec<Vec<usize>> {
    fn go(from: usize, parts: usize, n: usize, cur: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if parts == 1 {
            let mut c = cur.clone();
            c.push(n);
            out.push(c);
            return;
        }
        for end in from + 1..=n - (parts - 1) {
            cur.push(end);
            go(end, parts - 1, n, cur, out);
            cur.pop();
        }
    }
    let mut out = Vec::new();
    go(0, m, n, &mut vec![0], &mut out);
    out
}

/// Independent enumeration of every contiguous split; segment sums are direct left-to-right sums.
fn brute_force(p: &PipelineProblem) -> Option<f64> {
    let (n, m) = (p.layers.len(), p.stages.len());
    let mut best: Option<f64> = None;
    'cuts: for cut in all_cuts(n, m) {
        let mut worst = 0.0_f64;
        for k in 0..m {
            let bytes: u64 = p.layers[cut[k]..cut[k + 1]].iter().map(|l| l.bytes).sum();
            if bytes > p.stages[k].available_bytes {
                continue 'cuts;
            }
            let cost = direct_sum(p, k, cut[k], cut[k + 1]) + p.stages[k].boundary_after_us;
            worst = worst.max(cost);
        }
        if best.is_none_or(|b| worst < b) {
            best = Some(worst);
        }
    }
    best
}

fn check_structure(p: &PipelineProblem, r: &Pipeline) {
    let (n, m) = (p.layers.len(), p.stages.len());
    assert_eq!(r.segments.len(), m);
    let mut next = 0;
    let mut worst = 0.0_f64;
    let mut serial_compute = 0.0;
    for (k, s) in r.segments.iter().enumerate() {
        assert_eq!(s.rank, p.stages[k].rank);
        assert_eq!(s.first_layer, next);
        assert!(s.end_layer > s.first_layer);
        next = s.end_layer;
        let bytes: u64 = p.layers[s.first_layer..s.end_layer]
            .iter()
            .map(|l| l.bytes)
            .sum();
        assert_eq!(s.bytes, bytes);
        assert!(bytes <= p.stages[k].available_bytes);
        let direct = direct_sum(p, k, s.first_layer, s.end_layer);
        assert_eq!(s.compute_us.to_bits(), direct.to_bits());
        worst = worst.max(direct + p.stages[k].boundary_after_us);
        serial_compute += s.compute_us;
    }
    assert_eq!(next, n);
    assert_eq!(r.bottleneck_proxy_us.to_bits(), worst.to_bits());
    let boundaries: f64 = p.stages.iter().map(|s| s.boundary_after_us).sum();
    assert_eq!(
        r.serial_latency_us.to_bits(),
        (serial_compute + boundaries).to_bits()
    );
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        (self.next() >> 11) % n
    }
}

#[test]
fn catastrophic_cancellation_regression_picks_the_exact_optimum() {
    let p = problem(
        &[&[1.0, 1e20], &[1.0, 10.0], &[1.0, 20.0]],
        &[1, 1, 1],
        &[3, 3],
        &[0.0, 0.0],
    );
    let r = partition(&p).unwrap();
    assert_eq!(r.segments[0].end_layer, 2);
    assert_eq!(r.segments[1].first_layer, 2);
    assert_eq!(r.segments[0].compute_us, 2.0);
    assert_eq!(r.segments[1].compute_us, 20.0);
    assert_eq!(r.bottleneck_proxy_us, 20.0);
    assert_eq!(r.serial_latency_us, 22.0);
    check_structure(&p, &r);
}

#[test]
fn huge_layer_before_and_after_the_segment_does_not_hide_small_costs() {
    // The 1e20 sits in the middle of the row in one case and at its end in the other.
    let middle = problem(
        &[&[1.0, 10.0], &[1.0, 1e20], &[1.0, 20.0]],
        &[1, 1, 1],
        &[3, 3],
        &[0.0, 0.0],
    );
    let r = partition(&middle).unwrap();
    check_structure(&middle, &r);
    assert_eq!(r.segments[0].end_layer, 2);
    assert_eq!(r.bottleneck_proxy_us, 20.0);
    let after = problem(
        &[&[1.0, 1.0], &[1.0, 10.0], &[1.0, 20.0], &[1.0, 1e20]],
        &[1, 1, 1, 1],
        &[4, 4],
        &[0.0, 0.0],
    );
    let r = partition(&after).unwrap();
    check_structure(&after, &r);
    assert_eq!(r.bottleneck_proxy_us, brute_force(&after).unwrap());
    assert_eq!(r.segments[1].compute_us, 1e20);
}

#[test]
fn mixed_scales_in_every_position_for_every_small_grid_cell() {
    let values = [1.0, 10.0, 1e20];
    for caps in [[3, 3], [2, 3], [3, 2], [1, 3], [3, 1]] {
        for code in 0..3usize.pow(6) {
            let mut c = code;
            let mut rows = [[0.0; 2]; 3];
            for row in &mut rows {
                for v in row.iter_mut() {
                    *v = values[c % 3];
                    c /= 3;
                }
            }
            let p = problem(
                &[&rows[0], &rows[1], &rows[2]],
                &[1, 1, 1],
                &caps,
                &[0.0, 0.0],
            );
            match (partition(&p), brute_force(&p)) {
                (Ok(r), Some(best)) => {
                    check_structure(&p, &r);
                    assert_eq!(r.bottleneck_proxy_us.to_bits(), best.to_bits());
                }
                (Err(e), None) => assert_eq!(e.code, "PIPELINE_INFEASIBLE"),
                (a, b) => panic!("mismatch {a:?} vs {b:?} for {rows:?} {caps:?}"),
            }
        }
    }
}

#[test]
fn random_exact_integer_scales_match_a_u128_oracle() {
    let scales: [u32; 6] = [0, 17, 33, 60, 100, 0];
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for _ in 0..3000 {
        let m = 1 + rng.below(4) as usize;
        let n = m + rng.below(5) as usize;
        let shift: Vec<u32> = (0..m).map(|_| scales[rng.below(6) as usize]).collect();
        let small: Vec<Vec<u128>> = (0..n)
            .map(|_| (0..m).map(|_| rng.below(10) as u128).collect())
            .collect();
        let bsmall: Vec<u128> = (0..m)
            .map(|k| if k + 1 == m { 0 } else { rng.below(6) as u128 })
            .collect();
        let bytes: Vec<u64> = (0..n).map(|_| 1 + rng.below(4)).collect();
        let caps: Vec<u64> = (0..m).map(|_| 2 + rng.below(8)).collect();
        let rows: Vec<Vec<f64>> = small
            .iter()
            .map(|r| {
                r.iter()
                    .zip(&shift)
                    .map(|(v, s)| ((*v) << s) as f64)
                    .collect()
            })
            .collect();
        let row_refs: Vec<&[f64]> = rows.iter().map(|r| r.as_slice()).collect();
        let bounds: Vec<f64> = bsmall
            .iter()
            .zip(&shift)
            .map(|(v, s)| ((*v) << s) as f64)
            .collect();
        let p = problem(&row_refs, &bytes, &caps, &bounds);
        let mut oracle: Option<u128> = None;
        'cuts: for cut in all_cuts(n, m) {
            let mut worst = 0u128;
            for k in 0..m {
                let b: u64 = bytes[cut[k]..cut[k + 1]].iter().sum();
                if b > caps[k] {
                    continue 'cuts;
                }
                let seg: u128 = small[cut[k]..cut[k + 1]].iter().map(|r| r[k]).sum();
                worst = worst.max((seg + bsmall[k]) << shift[k]);
            }
            if oracle.is_none_or(|o| worst < o) {
                oracle = Some(worst);
            }
        }
        match (partition(&p), oracle) {
            (Ok(r), Some(best)) => {
                check_structure(&p, &r);
                assert_eq!(r.bottleneck_proxy_us, best as f64, "{p:?}");
            }
            (Err(e), None) => assert_eq!(e.code, "PIPELINE_INFEASIBLE"),
            (a, b) => panic!("mismatch {a:?} vs {b:?} for {p:?}"),
        }
    }
}

#[test]
fn random_mixed_magnitude_rows_match_direct_enumeration() {
    let values = [0.0, 1.0, 3.0, 10.0, 20.0, 0.1, 1e8, 1e16, 1e20, 1e30, 1e300];
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    for _ in 0..4000 {
        let m = 1 + rng.below(4) as usize;
        let n = m + rng.below(5) as usize;
        let rows: Vec<Vec<f64>> = (0..n)
            .map(|_| {
                (0..m)
                    .map(|_| values[rng.below(values.len() as u64) as usize])
                    .collect()
            })
            .collect();
        let row_refs: Vec<&[f64]> = rows.iter().map(|r| r.as_slice()).collect();
        let bytes: Vec<u64> = (0..n).map(|_| 1 + rng.below(3)).collect();
        let caps: Vec<u64> = (0..m).map(|_| 1 + rng.below(8)).collect();
        let bounds: Vec<f64> = (0..m)
            .map(|k| {
                if k + 1 == m {
                    0.0
                } else {
                    values[rng.below(values.len() as u64) as usize]
                }
            })
            .collect();
        let p = problem(&row_refs, &bytes, &caps, &bounds);
        match (partition(&p), brute_force(&p)) {
            (Ok(r), Some(best)) => {
                check_structure(&p, &r);
                assert_eq!(r.bottleneck_proxy_us.to_bits(), best.to_bits(), "{p:?}");
            }
            (Err(e), None) => assert_eq!(e.code, "PIPELINE_INFEASIBLE", "{p:?}"),
            (a, b) => panic!("mismatch {a:?} vs {b:?} for {p:?}"),
        }
    }
}

#[test]
fn equal_cost_ties_keep_the_earliest_last_segment_start() {
    let p = problem(
        &[&[1.0, 1.0], &[1.0, 1.0], &[1.0, 1.0]],
        &[1, 1, 1],
        &[3, 3],
        &[0.0, 0.0],
    );
    let r = partition(&p).unwrap();
    assert_eq!(r.segments[0].end_layer, 1);
    assert_eq!(r.bottleneck_proxy_us, 2.0);
    assert_eq!(r.serial_latency_us, 3.0);
}

#[test]
fn boundary_cost_and_capacity_still_shape_the_choice() {
    let costs: [&[f64]; 4] = [&[2.0, 2.0]; 4];
    let p = problem(&costs, &[10; 4], &[30, 30], &[1.0, 0.0]);
    let r = partition(&p).unwrap();
    assert_eq!(r.segments[0].end_layer, 2);
    assert_eq!(r.bottleneck_proxy_us, 5.0);
    assert_eq!(r.serial_latency_us, 9.0);
    let p = problem(&costs, &[10; 4], &[10, 30], &[1.0, 0.0]);
    let r = partition(&p).unwrap();
    assert_eq!(r.segments[0].end_layer, 1);
    assert_eq!(r.bottleneck_proxy_us, 6.0);
    let p = problem(&costs, &[10; 4], &[10, 10], &[1.0, 0.0]);
    assert_eq!(partition(&p).unwrap_err().code, "PIPELINE_INFEASIBLE");
    // A large boundary is part of the cost even when the compute is tiny.
    let p = problem(&[&[1.0, 1.0], &[1.0, 1.0]], &[1, 1], &[1, 1], &[1e20, 0.0]);
    assert_eq!(partition(&p).unwrap().bottleneck_proxy_us, 1e20);
}

#[test]
fn nonfinite_negative_and_shape_errors_are_rejected() {
    let ok = || problem(&[&[1.0, 1.0], &[1.0, 1.0]], &[1, 1], &[2, 2], &[0.0, 0.0]);
    partition(&ok()).unwrap();
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
        let mut p = ok();
        p.layers[1].compute_us_by_stage[0] = bad;
        assert_eq!(partition(&p).unwrap_err().code, "INVALID_INPUT");
        let mut p = ok();
        p.stages[0].boundary_after_us = bad;
        assert_eq!(partition(&p).unwrap_err().code, "INVALID_INPUT");
    }
    let mut p = ok();
    p.stages[1].boundary_after_us = 1.0;
    assert!(partition(&p).is_err());
    let mut p = ok();
    p.layers[0].compute_us_by_stage.pop();
    assert!(partition(&p).is_err());
    let mut p = ok();
    p.stages[0].available_bytes = 0;
    assert!(partition(&p).is_err());
    let mut p = ok();
    p.stages[1].rank = 0;
    assert!(partition(&p).is_err());
    let mut p = ok();
    p.layers[0].bytes = u64::MAX;
    p.layers[1].bytes = 1;
    assert!(partition(&p).is_err());
}

#[test]
fn overflow_policy_row_total_candidate_cost_and_serial() {
    // Row total (left-to-right sum of one stage) overflowing is rejected up front.
    let p = problem(
        &[&[f64::MAX, 1.0], &[f64::MAX, 1.0]],
        &[1, 1],
        &[2, 2],
        &[0.0, 0.0],
    );
    assert_eq!(partition(&p).unwrap_err().code, "INVALID_INPUT");
    // Only candidate overflows: no representable result, reported as overflow, not as infeasible.
    let p = problem(
        &[&[f64::MAX, 1.0], &[1.0, 1.0]],
        &[1, 1],
        &[1, 1],
        &[f64::MAX, 0.0],
    );
    let e = partition(&p).unwrap_err();
    assert_eq!(e.code, "INVALID_INPUT");
    assert!(e.message.contains("overflow"), "{e:?}");
    // An overflowing candidate is worse than any finite one, so the finite cut still wins.
    let p = problem(
        &[&[1.0, 1.0], &[1.2e308, 1.0], &[1.0, 1.0]],
        &[1, 1, 1],
        &[3, 3],
        &[1e308, 0.0],
    );
    let r = partition(&p).unwrap();
    check_structure(&p, &r);
    assert_eq!(r.segments[0].end_layer, 1);
    assert_eq!(r.bottleneck_proxy_us, 1e308);
    // Serial sum overflow is still an error although every candidate cost is finite.
    let p = problem(
        &[&[f64::MAX / 2.0, 1.0], &[1.0, f64::MAX]],
        &[1, 1],
        &[1, 1],
        &[f64::MAX / 2.0, 0.0],
    );
    let e = partition(&p).unwrap_err();
    assert_eq!(e.code, "INVALID_INPUT");
    assert!(e.message.contains("overflow"), "{e:?}");
}

#[test]
fn size_bounds_and_maximum_problem() {
    let layers = |n: usize, m: usize| {
        let row = vec![1.0; m];
        let rows: Vec<&[f64]> = (0..n).map(|_| row.as_slice()).collect();
        problem(&rows, &vec![1; n], &vec![u64::MAX; m], &vec![0.0; m])
    };
    assert!(partition(&layers(1025, 2)).is_err());
    assert!(partition(&layers(65, 65)).is_err());
    assert!(partition(&layers(2, 3)).is_err());
    let mut empty = layers(1, 1);
    empty.stages.clear();
    assert!(partition(&empty).is_err());
    let p = layers(1024, 64);
    let r = partition(&p).unwrap();
    check_structure(&p, &r);
    assert_eq!(r.bottleneck_proxy_us, 16.0);
    assert_eq!(r.serial_latency_us, 1024.0);
    assert!(r.segments.iter().all(|s| s.end_layer - s.first_layer == 16));
    let one = layers(1024, 1);
    assert_eq!(partition(&one).unwrap().bottleneck_proxy_us, 1024.0);
}

#[test]
fn huge_first_layer_does_not_hide_a_thousand_small_costs() {
    let n = 1024;
    let rows: Vec<Vec<f64>> = (0..n)
        .map(|i| vec![1.0, if i == 0 { 1e20 } else { 1.0 }])
        .collect();
    let refs: Vec<&[f64]> = rows.iter().map(|r| r.as_slice()).collect();
    let p = problem(&refs, &vec![1; n], &[2000, 2000], &[0.0, 0.0]);
    let r = partition(&p).unwrap();
    check_structure(&p, &r);
    assert_eq!(r.segments[0].end_layer, 512);
    assert_eq!(r.bottleneck_proxy_us, 512.0);
    assert_eq!(r.serial_latency_us, 1024.0);
}

#[test]
fn candidate_overflow_does_not_hide_an_infeasible_capacity_partition() {
    // Both aggregate insufficiency and an ordered final-stage capacity failure.
    // All individual inputs and row totals are finite.
    for first_capacity in [2, 100] {
        let p = problem(
            &[&[1e308, 1.0], &[1.0, 1.0], &[1.0, 1.0]],
            &[1, 1, 2],
            &[first_capacity, 1],
            &[1e308, 0.0],
        );
        assert_eq!(partition(&p).unwrap_err().code, "PIPELINE_INFEASIBLE");
    }
}

#[test]
fn legacy_whole_profile_overflow_precedes_capacity_validation() {
    let p = problem(
        &[&[f64::MAX, 1.0], &[f64::MAX, 1.0], &[1.0, 1.0]],
        &[1, 1, 5],
        &[1, 1],
        &[0.0, 0.0],
    );
    let error = partition(&p).unwrap_err();
    assert_eq!(error.code, "INVALID_INPUT");
    assert!(error.message.contains("timing overflow"));
}

// A40 guard sweep: with each clause disabled these inputs were refused only later,
// as infeasible or as a timing overflow; each must fail for its own reason.
#[test]
fn stage_contract_refuses_for_its_own_reason() {
    let p = problem(&[&[1.0, 1.0]], &[1], &[100, 100], &[1.0, 0.0]);
    let error = partition(&p).unwrap_err();
    assert!(
        error.message.contains("require 1..64 nonempty stages"),
        "{error}"
    );
    let row: &[f64] = &[1.0, 1.0];
    let base = problem(&[row; 3], &[1; 3], &[100, 100], &[1.0, 0.0]);
    for edit in [
        (|p: &mut PipelineProblem| p.stages[0].available_bytes = 0) as fn(&mut PipelineProblem),
        |p| p.stages[0].boundary_after_us = f64::NAN,
        |p| p.stages[0].boundary_after_us = f64::INFINITY,
    ] {
        let mut p = base.clone();
        edit(&mut p);
        assert_eq!(partition(&p).unwrap_err().message, "invalid pipeline stage");
    }
}
