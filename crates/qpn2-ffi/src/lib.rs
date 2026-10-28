//! Synchronous safe binding to the real CPU C validator; no CUDA launch entry point.
//! Device addresses are opaque arithmetic values, never host-dereferenced here.
//! Validation does not establish device accessibility, mirror equality or GPU results.
#![deny(unsafe_op_in_unsafe_fn)]
#[cfg(not(target_pointer_width = "64"))]
compile_error!("QPN2 ABI v1/struct208 requires a native 64-bit target");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dimensions {
    pub experts: u32,
    pub n: u32,
    pub k: u32,
    pub slots: u32,
    pub input_rows: u32,
    pub in_div: u32,
    pub splitk: u32,
    pub nacc: u32,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceBuffer {
    pub address: usize,
    pub capacity: usize,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceBuffers {
    /// Byte capacities.
    pub codes: DeviceBuffer,
    pub scales: DeviceBuffer,
    /// Number of F32 values.
    pub gscales: DeviceBuffer,
    /// Number of FP16 bit-pattern values.
    pub x: DeviceBuffer,
    pub y: DeviceBuffer,
    /// Exact mirrored i32 table counts, rather than extra allocated capacity.
    pub block_expert: DeviceBuffer,
    pub sorted_slots: DeviceBuffer,
    /// Exact device scalar count (one).
    pub ntpp: DeviceBuffer,
}
#[derive(Debug, Clone, Copy)]
pub struct HostMetadata<'a> {
    pub block_expert: &'a [i32],
    pub sorted_slots: &'a [i32],
    pub ntpp: i32,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Abi,
    Dimension,
    Length,
    Pointer,
    Index,
    Coverage,
    Resource,
    Cuda,
    Unknown(i32),
}
impl Status {
    pub const fn code(self) -> i32 {
        match self {
            Self::Abi => -1,
            Self::Dimension => -2,
            Self::Length => -3,
            Self::Pointer => -4,
            Self::Index => -5,
            Self::Coverage => -6,
            Self::Resource => -7,
            Self::Cuda => -8,
            Self::Unknown(code) => code,
        }
    }
}
impl std::fmt::Display for Status {
    fn fmt(&self, output: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(output, "QPN2 C validation {:?} ({})", self, self.code())
    }
}
impl std::error::Error for Status {}
#[repr(C)]
struct Params {
    abi_version: u32,
    struct_size: u32,
    experts: u32,
    n: u32,
    k: u32,
    slots: u32,
    input_rows: u32,
    in_div: u32,
    splitk: u32,
    nacc: u32,
    codes: *const u8,
    codes_bytes: usize,
    scales: *const u8,
    scales_bytes: usize,
    gscales: *const f32,
    gscales_count: usize,
    x: *const u16,
    x_count: usize,
    y: *mut u16,
    y_count: usize,
    block_expert: *const i32,
    block_expert_count: usize,
    sorted_slots: *const i32,
    sorted_slots_count: usize,
    ntpp: *const i32,
    ntpp_count: usize,
    block_expert_host: *const i32,
    block_expert_host_count: usize,
    sorted_slots_host: *const i32,
    sorted_slots_host_count: usize,
    ntpp_host: i32,
}
include!(concat!(env!("OUT_DIR"), "/abi.rs"));
const R_OFFSETS: [usize; 31] = [
    std::mem::offset_of!(Params, abi_version),
    std::mem::offset_of!(Params, struct_size),
    std::mem::offset_of!(Params, experts),
    std::mem::offset_of!(Params, n),
    std::mem::offset_of!(Params, k),
    std::mem::offset_of!(Params, slots),
    std::mem::offset_of!(Params, input_rows),
    std::mem::offset_of!(Params, in_div),
    std::mem::offset_of!(Params, splitk),
    std::mem::offset_of!(Params, nacc),
    std::mem::offset_of!(Params, codes),
    std::mem::offset_of!(Params, codes_bytes),
    std::mem::offset_of!(Params, scales),
    std::mem::offset_of!(Params, scales_bytes),
    std::mem::offset_of!(Params, gscales),
    std::mem::offset_of!(Params, gscales_count),
    std::mem::offset_of!(Params, x),
    std::mem::offset_of!(Params, x_count),
    std::mem::offset_of!(Params, y),
    std::mem::offset_of!(Params, y_count),
    std::mem::offset_of!(Params, block_expert),
    std::mem::offset_of!(Params, block_expert_count),
    std::mem::offset_of!(Params, sorted_slots),
    std::mem::offset_of!(Params, sorted_slots_count),
    std::mem::offset_of!(Params, ntpp),
    std::mem::offset_of!(Params, ntpp_count),
    std::mem::offset_of!(Params, block_expert_host),
    std::mem::offset_of!(Params, block_expert_host_count),
    std::mem::offset_of!(Params, sorted_slots_host),
    std::mem::offset_of!(Params, sorted_slots_host_count),
    std::mem::offset_of!(Params, ntpp_host),
];
const R_SIZES: [usize; 31] = [
    std::mem::size_of::<u32>(),
    std::mem::size_of::<u32>(),
    std::mem::size_of::<u32>(),
    std::mem::size_of::<u32>(),
    std::mem::size_of::<u32>(),
    std::mem::size_of::<u32>(),
    std::mem::size_of::<u32>(),
    std::mem::size_of::<u32>(),
    std::mem::size_of::<u32>(),
    std::mem::size_of::<u32>(),
    std::mem::size_of::<*const u8>(),
    std::mem::size_of::<usize>(),
    std::mem::size_of::<*const u8>(),
    std::mem::size_of::<usize>(),
    std::mem::size_of::<*const f32>(),
    std::mem::size_of::<usize>(),
    std::mem::size_of::<*const u16>(),
    std::mem::size_of::<usize>(),
    std::mem::size_of::<*mut u16>(),
    std::mem::size_of::<usize>(),
    std::mem::size_of::<*const i32>(),
    std::mem::size_of::<usize>(),
    std::mem::size_of::<*const i32>(),
    std::mem::size_of::<usize>(),
    std::mem::size_of::<*const i32>(),
    std::mem::size_of::<usize>(),
    std::mem::size_of::<*const i32>(),
    std::mem::size_of::<usize>(),
    std::mem::size_of::<*const i32>(),
    std::mem::size_of::<usize>(),
    std::mem::size_of::<i32>(),
];
const R_ALIGNMENTS: [usize; 31] = [
    std::mem::align_of::<u32>(),
    std::mem::align_of::<u32>(),
    std::mem::align_of::<u32>(),
    std::mem::align_of::<u32>(),
    std::mem::align_of::<u32>(),
    std::mem::align_of::<u32>(),
    std::mem::align_of::<u32>(),
    std::mem::align_of::<u32>(),
    std::mem::align_of::<u32>(),
    std::mem::align_of::<u32>(),
    std::mem::align_of::<*const u8>(),
    std::mem::align_of::<usize>(),
    std::mem::align_of::<*const u8>(),
    std::mem::align_of::<usize>(),
    std::mem::align_of::<*const f32>(),
    std::mem::align_of::<usize>(),
    std::mem::align_of::<*const u16>(),
    std::mem::align_of::<usize>(),
    std::mem::align_of::<*mut u16>(),
    std::mem::align_of::<usize>(),
    std::mem::align_of::<*const i32>(),
    std::mem::align_of::<usize>(),
    std::mem::align_of::<*const i32>(),
    std::mem::align_of::<usize>(),
    std::mem::align_of::<*const i32>(),
    std::mem::align_of::<usize>(),
    std::mem::align_of::<*const i32>(),
    std::mem::align_of::<usize>(),
    std::mem::align_of::<*const i32>(),
    std::mem::align_of::<usize>(),
    std::mem::align_of::<i32>(),
];
// Verify the independent native C measurements before any safe call can compile.
const _: () = {
    assert!(C_ABI == 1);
    assert!(C_SIZE == 208);
    assert!(std::mem::size_of::<Params>() == C_SIZE);
    assert!(std::mem::align_of::<Params>() == C_ALIGN);
    let mut i = 0;
    while i < 31 {
        assert!(R_OFFSETS[i] == C_OFFSETS[i]);
        assert!(R_SIZES[i] == C_SIZES[i]);
        assert!(R_ALIGNMENTS[i] == C_ALIGNMENTS[i]);
        i += 1;
    }
    let statuses = [0, -1, -2, -3, -4, -5, -6, -7, -8];
    i = 0;
    while i < 9 {
        assert!(C_STATUSES[i] == statuses[i]);
        i += 1;
    }
};

unsafe extern "C" {
    // Exact declaration from qpn2_moe_ffi.h; Params layout is checked above.
    fn vf_qpn2_moe_validate(params: *const Params) -> i32;
}

/// Validate one projection synchronously using the real CPU C validator.
///
/// Device addresses are opaque and need not be host-accessible. Capacities use
/// the units documented on DeviceBuffers. HOST lengths are generated from the
/// borrowed slices, so callers cannot forge an out-of-bounds HOST span.
/// Success does not prove device allocation, contents or equality with HOST
/// metadata; those remain the caller's obligations before any future launch.
pub fn validate(
    dimensions: Dimensions,
    buffers: &DeviceBuffers,
    host: HostMetadata<'_>,
) -> Result<(), Status> {
    let params = Params {
        abi_version: 1,
        struct_size: std::mem::size_of::<Params>() as u32,
        experts: dimensions.experts,
        n: dimensions.n,
        k: dimensions.k,
        slots: dimensions.slots,
        input_rows: dimensions.input_rows,
        in_div: dimensions.in_div,
        splitk: dimensions.splitk,
        nacc: dimensions.nacc,
        codes: buffers.codes.address as *const u8,
        codes_bytes: buffers.codes.capacity,
        scales: buffers.scales.address as *const u8,
        scales_bytes: buffers.scales.capacity,
        gscales: buffers.gscales.address as *const f32,
        gscales_count: buffers.gscales.capacity,
        x: buffers.x.address as *const u16,
        x_count: buffers.x.capacity,
        y: buffers.y.address as *mut u16,
        y_count: buffers.y.capacity,
        block_expert: buffers.block_expert.address as *const i32,
        block_expert_count: buffers.block_expert.capacity,
        sorted_slots: buffers.sorted_slots.address as *const i32,
        sorted_slots_count: buffers.sorted_slots.capacity,
        ntpp: buffers.ntpp.address as *const i32,
        ntpp_count: buffers.ntpp.capacity,
        block_expert_host: host.block_expert.as_ptr(),
        block_expert_host_count: host.block_expert.len(),
        sorted_slots_host: host.sorted_slots.as_ptr(),
        sorted_slots_host_count: host.sorted_slots.len(),
        ntpp_host: host.ntpp,
    };
    // SAFETY: params is initialized, alive for this synchronous call and its
    // repr(C) layout, field sizes/alignments and status values are independently
    // verified against the actual C header at compile time. Both HOST pointers
    // refer to immutable live i32 slices with their real lengths. The actual C
    // validator checks HOST/device count equality, bounded table sizes and ntpp
    // before reading HOST elements, including when either slice is empty.
    // All DEVICE pointers are opaque: this CPU function performs only integer
    // span/alignment checks on them and never dereferences or writes them.
    // It does not retain any pointer, launch CUDA or call a device backend.
    let code = unsafe { vf_qpn2_moe_validate(&params) };
    match code {
        0 => Ok(()),
        -1 => Err(Status::Abi),
        -2 => Err(Status::Dimension),
        -3 => Err(Status::Length),
        -4 => Err(Status::Pointer),
        -5 => Err(Status::Index),
        -6 => Err(Status::Coverage),
        -7 => Err(Status::Resource),
        -8 => Err(Status::Cuda),
        other => Err(Status::Unknown(other)),
    }
}

#[cfg(test)]
mod abi_tests {
    use super::*;
    #[test]
    fn every_field_is_measured_by_the_native_c_probe() {
        assert_eq!(
            C_NAMES,
            [
                "abi_version",
                "struct_size",
                "experts",
                "n",
                "k",
                "slots",
                "input_rows",
                "in_div",
                "splitk",
                "nacc",
                "codes",
                "codes_bytes",
                "scales",
                "scales_bytes",
                "gscales",
                "gscales_count",
                "x",
                "x_count",
                "y",
                "y_count",
                "block_expert",
                "block_expert_count",
                "sorted_slots",
                "sorted_slots_count",
                "ntpp",
                "ntpp_count",
                "block_expert_host",
                "block_expert_host_count",
                "sorted_slots_host",
                "sorted_slots_host_count",
                "ntpp_host"
            ]
        );
        assert_eq!(R_OFFSETS, C_OFFSETS);
        assert_eq!(R_SIZES, C_SIZES);
        assert_eq!(R_ALIGNMENTS, C_ALIGNMENTS);
        assert_eq!(std::mem::size_of::<Params>(), C_SIZE);
        assert_eq!(std::mem::align_of::<Params>(), C_ALIGN);
        assert_eq!(
            C_STATUSES,
            [
                0,
                Status::Abi.code(),
                Status::Dimension.code(),
                Status::Length.code(),
                Status::Pointer.code(),
                Status::Index.code(),
                Status::Coverage.code(),
                Status::Resource.code(),
                Status::Cuda.code()
            ]
        );
    }
}
