use vf_qpn2_ffi::*;
fn dimensions() -> Dimensions {
    Dimensions {
        experts: 1,
        n: 64,
        k: 64,
        slots: 1,
        input_rows: 1,
        in_div: 1,
        splitk: 4,
        nacc: 1,
    }
}
fn buffers() -> DeviceBuffers {
    DeviceBuffers {
        codes: DeviceBuffer {
            address: 0x10000,
            capacity: 2048,
        },
        scales: DeviceBuffer {
            address: 0x20000,
            capacity: 256,
        },
        gscales: DeviceBuffer {
            address: 0x30000,
            capacity: 2,
        },
        x: DeviceBuffer {
            address: 0x40000,
            capacity: 64,
        },
        y: DeviceBuffer {
            address: 0x50000,
            capacity: 64,
        },
        block_expert: DeviceBuffer {
            address: 0x60000,
            capacity: 1,
        },
        sorted_slots: DeviceBuffer {
            address: 0x70000,
            capacity: 8,
        },
        ntpp: DeviceBuffer {
            address: 0x80000,
            capacity: 1,
        },
    }
}
fn metadata() -> HostMetadata<'static> {
    HostMetadata {
        block_expert: &[0],
        sorted_slots: &[0, 1, 1, 1, 1, 1, 1, 1],
        ntpp: 8,
    }
}

#[test]
fn a142_rank_partial_four_slots_in_two_expert_blocks_validate_on_cpu() {
    // A142 selects global route positions [0, 1, 10, 11]. After mapping to
    // local experts 0 and 1, each expert contributes two rows in its own
    // eight-row block; 4 is the padding sentinel for slots >= slots_count.
    let host = HostMetadata {
        block_expert: &[0, 1],
        sorted_slots: &[0, 3, 4, 4, 4, 4, 4, 4, 1, 2, 4, 4, 4, 4, 4, 4],
        ntpp: 16,
    };
    for (n, k) in [(4608usize, 5120usize), (5120, 2304)] {
        let dimensions = Dimensions {
            experts: 6,
            n: n as u32,
            k: k as u32,
            slots: 4,
            input_rows: 4,
            in_div: 1,
            splitk: 8,
            nacc: 1,
        };
        let codes_bytes = 6 * n * k / 2;
        let scales_bytes = 6 * n * k / 16;
        let buffers = DeviceBuffers {
            codes: DeviceBuffer {
                address: 0x0100_0000,
                capacity: codes_bytes,
            },
            scales: DeviceBuffer {
                address: 0x1000_0000,
                capacity: scales_bytes,
            },
            gscales: DeviceBuffer {
                address: 0x2000_0000,
                capacity: 6 * n / 32,
            },
            x: DeviceBuffer {
                address: 0x3000_0000,
                capacity: 4 * k,
            },
            y: DeviceBuffer {
                address: 0x4000_0000,
                capacity: 4 * n,
            },
            block_expert: DeviceBuffer {
                address: 0x5000_0000,
                capacity: 2,
            },
            sorted_slots: DeviceBuffer {
                address: 0x6000_0000,
                capacity: 16,
            },
            ntpp: DeviceBuffer {
                address: 0x7000_0000,
                capacity: 1,
            },
        };
        assert_eq!(validate(dimensions, &buffers, host), Ok(()), "n={n}, k={k}");
    }
}

#[test]
fn real_validator_accepts_opaque_non_host_device_addresses() {
    assert_eq!(validate(dimensions(), &buffers(), metadata()), Ok(()));
}
#[test]
fn all_null_device_addresses_are_rejected() {
    for field in 0..8 {
        let mut b = buffers();
        match field {
            0 => b.codes.address = 0,
            1 => b.scales.address = 0,
            2 => b.gscales.address = 0,
            3 => b.x.address = 0,
            4 => b.y.address = 0,
            5 => b.block_expert.address = 0,
            6 => b.sorted_slots.address = 0,
            _ => b.ntpp.address = 0,
        }
        assert_eq!(
            validate(dimensions(), &b, metadata()),
            Err(Status::Pointer),
            "field={field}"
        );
    }
}
#[test]
fn minimum_data_capacities_and_exact_metadata_counts_are_checked() {
    for field in 0..8 {
        let mut b = buffers();
        match field {
            0 => b.codes.capacity -= 1,
            1 => b.scales.capacity -= 1,
            2 => b.gscales.capacity -= 1,
            3 => b.x.capacity -= 1,
            4 => b.y.capacity -= 1,
            5 => b.block_expert.capacity += 1,
            6 => b.sorted_slots.capacity += 1,
            _ => b.ntpp.capacity = 0,
        }
        assert_eq!(
            validate(dimensions(), &b, metadata()),
            Err(Status::Length),
            "field={field}"
        );
    }
    let mut b = buffers();
    b.codes.capacity += 128;
    assert_eq!(validate(dimensions(), &b, metadata()), Ok(()));
}
#[test]
fn output_alias_address_wrap_and_misalignment_are_rejected() {
    for address in [
        buffers().codes.address,
        buffers().x.address,
        metadata().block_expert.as_ptr() as usize,
        metadata().sorted_slots.as_ptr() as usize,
    ] {
        let mut b = buffers();
        b.y.address = address;
        assert_eq!(validate(dimensions(), &b, metadata()), Err(Status::Pointer));
    }
    let mut b = buffers();
    b.codes.address = usize::MAX - 7;
    assert_eq!(validate(dimensions(), &b, metadata()), Err(Status::Pointer));
    let mut b = buffers();
    b.x.address += 2;
    assert_eq!(validate(dimensions(), &b, metadata()), Err(Status::Pointer));
    let mut b = buffers();
    b.y.address = b.codes.address + b.codes.capacity;
    assert_eq!(validate(dimensions(), &b, metadata()), Ok(())); // Adjacent disjoint spans.
}
#[test]
fn short_or_empty_host_slices_are_rejected_without_oob_reads() {
    for host in [
        HostMetadata {
            block_expert: &[],
            sorted_slots: metadata().sorted_slots,
            ntpp: 8,
        },
        HostMetadata {
            block_expert: &[0],
            sorted_slots: &[],
            ntpp: 8,
        },
        HostMetadata {
            block_expert: &[0],
            sorted_slots: &[0, 1, 1],
            ntpp: 8,
        },
    ] {
        assert_eq!(
            validate(dimensions(), &buffers(), host),
            Err(Status::Length)
        );
    }
}
#[test]
fn negative_host_indices_and_ntpp_are_checked_by_c() {
    for host in [
        HostMetadata {
            block_expert: &[-2],
            sorted_slots: metadata().sorted_slots,
            ntpp: 8,
        },
        HostMetadata {
            block_expert: &[1],
            sorted_slots: metadata().sorted_slots,
            ntpp: 8,
        },
        HostMetadata {
            block_expert: &[0],
            sorted_slots: &[-1, 1, 1, 1, 1, 1, 1, 1],
            ntpp: 8,
        },
        HostMetadata {
            block_expert: &[-1],
            sorted_slots: metadata().sorted_slots,
            ntpp: 8,
        },
        HostMetadata {
            block_expert: &[0],
            sorted_slots: metadata().sorted_slots,
            ntpp: -8,
        },
        HostMetadata {
            block_expert: &[0],
            sorted_slots: metadata().sorted_slots,
            ntpp: 7,
        },
        HostMetadata {
            block_expert: &[0],
            sorted_slots: metadata().sorted_slots,
            ntpp: 16,
        },
    ] {
        assert_eq!(validate(dimensions(), &buffers(), host), Err(Status::Index));
    }
}
#[test]
fn duplicate_or_missing_output_coverage_is_rejected() {
    for sorted in [&[0, 0, 1, 1, 1, 1, 1, 1], &[1, 1, 1, 1, 1, 1, 1, 1]] {
        assert_eq!(
            validate(
                dimensions(),
                &buffers(),
                HostMetadata {
                    block_expert: &[0],
                    sorted_slots: sorted,
                    ntpp: 8
                }
            ),
            Err(Status::Coverage)
        );
    }
    assert_eq!(
        validate(
            dimensions(),
            &buffers(),
            HostMetadata {
                ntpp: 0,
                ..metadata()
            }
        ),
        Err(Status::Coverage)
    );
}
#[test]
fn unsupported_dimensions_and_geometry_are_rejected() {
    for field in 0..7 {
        let mut d = dimensions();
        match field {
            0 => d.experts = 0,
            1 => d.n = 33,
            2 => d.k = 63,
            3 => d.slots = 0,
            4 => d.input_rows = 2,
            5 => d.in_div = 65,
            _ => d.nacc = 2,
        }
        assert_eq!(
            validate(d, &buffers(), metadata()),
            Err(Status::Dimension),
            "field={field}"
        );
    }
}
#[test]
fn host_metadata_lifetimes_cover_synchronous_calls() {
    let blocks = vec![0];
    let slots = vec![0, 1, 1, 1, 1, 1, 1, 1];
    let host = HostMetadata {
        block_expert: &blocks,
        sorted_slots: &slots,
        ntpp: 8,
    };
    assert_eq!(validate(dimensions(), &buffers(), host), Ok(()));
    assert_eq!(blocks, vec![0]);
    assert_eq!(slots, vec![0, 1, 1, 1, 1, 1, 1, 1]);
}
