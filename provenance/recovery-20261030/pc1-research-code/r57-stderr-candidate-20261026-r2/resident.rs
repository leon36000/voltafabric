//! Safe, synchronous owner for the R13 GPU ABI. No CUDA pointer leaves the C++ owner.
//! Process-local handle; not Send/Sync. Typed output exists only after successful sync.
use std::{
    marker::PhantomData,
    rc::Rc,
    sync::{
        atomic::{AtomicI32, Ordering},
        Mutex,
    },
};
#[cfg(not(target_pointer_width = "64"))]
compile_error!("R13 ABI requires a native 64-bit target");
#[repr(C)]
#[derive(Clone, Copy)]
struct Span {
    data: *const u8,
    bytes: u64,
}
#[repr(C)]
struct RawWeights {
    abi: u32,
    size: u32,
    hidden: u32,
    middle: u32,
    planes: [Span; 6],
}
#[repr(C)]
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stats {
    pub weight_upload_bytes: u64,
    pub weight_upload_copies: u64,
    pub resident_weight_bytes: u64,
    pub input_h2d_bytes: u64,
    pub output_d2h_bytes: u64,
    pub status_d2h_bytes: u64,
    pub exec_calls: u64,
    pub published: u64,
    pub rejected: u64,
    pub kernel_launches: u64,
    pub down_launches: u64,
    pub device_allocations: u64,
    pub scratch_device_bytes: u64,
    pub poisoned: u64,
}
#[link(name = "vf_r13")]
unsafe extern "C" {
    fn vf_r13_create(w: *const RawWeights, id: *mut u64) -> i32;
    fn vf_r13_exec(
        id: u64,
        input: *const u32,
        n: u64,
        weight: u32,
        output: *mut u32,
        outputs: u64,
    ) -> i32;
    fn vf_r13_stats(id: u64, out: *mut Stats, bytes: u64) -> i32;
    fn vf_r13_destroy(id: u64) -> i32;
    fn vf_r13_weights_struct_bytes() -> u64;
    fn vf_r13_stats_struct_bytes() -> u64;
    fn vf_r13_cuda_api_calls() -> u64;
    fn vf_r13_cleanup_errors() -> u64;
    fn vf_r13_live_owners() -> u64;
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Status(pub i32);
fn check(code: i32) -> Result<(), Status> {
    if code == 0 {
        Ok(())
    } else {
        Err(Status(code))
    }
}
static DROP_ERROR: AtomicI32 = AtomicI32::new(0);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CleanupFailure {
    pub handle: u64,
    pub status: Status,
}
static FIRST_CLEANUP_FAILURE: Mutex<Option<CleanupFailure>> = Mutex::new(None);
fn record_cleanup_failure(handle: u64, code: i32) {
    DROP_ERROR.store(code, Ordering::SeqCst);
    let mut first = FIRST_CLEANUP_FAILURE
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    if first.is_none() {
        *first = Some(CleanupFailure {
            handle,
            status: Status(code),
        });
    }
}
/// Terminal process diagnostic, never a retry ticket. An abandoned refused close
/// keeps native allocations live until process teardown. Unknown native cleanup
/// may already have removed the handle. The supervisor must terminate/recycle this
/// worker; safe creation/execution stays disabled and no cleanup is replayed.
pub fn first_cleanup_failure() -> Option<CleanupFailure> {
    *FIRST_CLEANUP_FAILURE
        .lock()
        .unwrap_or_else(|p| p.into_inner())
}
pub struct Owner {
    id: u64,
    hidden: usize,
    drop_armed: bool,
    close_error: i32,
    _thread: PhantomData<Rc<()>>,
}
impl Owner {
    pub fn new(hidden: u32, middle: u32, planes: [&[u8]; 6]) -> Result<Self, Status> {
        if DROP_ERROR.load(Ordering::SeqCst) != 0 {
            return Err(Status(-10));
        }
        // SAFETY: size-query functions take no pointer and perform no device operation.
        if unsafe { vf_r13_weights_struct_bytes() } != std::mem::size_of::<RawWeights>() as u64
            || unsafe { vf_r13_stats_struct_bytes() } != std::mem::size_of::<Stats>() as u64
        {
            return Err(Status(-1));
        }
        let raw = RawWeights {
            abi: 1,
            size: std::mem::size_of::<RawWeights>() as u32,
            hidden,
            middle,
            planes: planes.map(|p| Span {
                data: p.as_ptr(),
                bytes: p.len() as u64,
            }),
        };
        let mut id = 0;
        // SAFETY: raw is aligned/live; immutable slices outlive the synchronous copy;
        // id is a separate valid u64. The C ABI catches all C++ exceptions.
        check(unsafe { vf_r13_create(&raw, &mut id) })?;
        if id == 0 {
            return Err(Status(-1));
        }
        Ok(Self {
            id,
            hidden: hidden as usize,
            drop_armed: true,
            close_error: 0,
            _thread: PhantomData,
        })
    }
    pub fn execute(&mut self, input: &[u32], weight_bits: u32) -> Result<Vec<u32>, Status> {
        if self.id == 0 {
            return Err(Status(-6));
        }
        if self.close_error != 0 || DROP_ERROR.load(Ordering::SeqCst) != 0 {
            return Err(Status(-10));
        }
        let mut output = vec![0xdeadbeef; self.hidden];
        // SAFETY: nonoverlapping live input/output slices; exact capacities supplied.
        // Exclusive &mut and !Send/Sync prohibit safe concurrent/context-thread use.
        // C validates BF16/finite input and coefficient before any CUDA API and commits
        // caller output only after synchronization and numerical validation.
        check(unsafe {
            vf_r13_exec(
                self.id,
                input.as_ptr(),
                input.len() as u64,
                weight_bits,
                output.as_mut_ptr(),
                output.len() as u64,
            )
        })?;
        Ok(output)
    }
    pub fn stats(&self) -> Result<Stats, Status> {
        let mut out = Stats::default();
        // SAFETY: aligned exact repr(C) output, versioned size checked by C.
        check(unsafe { vf_r13_stats(self.id, &mut out, std::mem::size_of::<Stats>() as u64) })?;
        Ok(out)
    }
    /// A refused close retains the handle for explicit reconciliation. It disarms
    /// automatic Drop cleanup: neither UNKNOWN nor a known refusal is retried by
    /// a destructor. After failure, only stats or an explicit close are allowed.
    /// Only BUSY/DEVICE are known unexecuted close refusals and may be reconciled
    /// explicitly. CUDA cleanup failure is terminal for the process. Dropping an
    /// unresolved owner records its ID and also requires worker process teardown;
    /// it deliberately cannot free storage in an unrelated CUDA context.
    pub fn close(&mut self) -> Result<(), Status> {
        if self.id == 0 {
            return Err(Status(-6));
        }
        if self.close_error != 0 && self.close_error != -7 && self.close_error != -12 {
            return Err(Status(self.close_error));
        }
        self.drop_armed = false;
        // SAFETY: sole Rust owner; native destruction validates context before
        // removing the handle. BUSY/DEVICE mean no owned storage was touched.
        let code = unsafe { vf_r13_destroy(self.id) };
        if code == 0 {
            self.id = 0;
            self.close_error = 0;
        } else {
            self.close_error = code;
            if code == -8 {
                record_cleanup_failure(self.id, code);
            }
        }
        check(code)
    }
}
impl Drop for Owner {
    fn drop(&mut self) {
        if self.id != 0 && !self.drop_armed {
            record_cleanup_failure(self.id, self.close_error);
        } else if self.id != 0 {
            // SAFETY: sole owner; no other safe reference can outlive this synchronous drop.
            let code = unsafe { vf_r13_destroy(self.id) };
            if code != 0 {
                record_cleanup_failure(self.id, code);
            }
            self.id = 0;
        }
    }
}
pub fn lifecycle() -> (u64, u64, i32) {
    // SAFETY: scalar counters/registry size, no caller pointers or GPU launches.
    unsafe {
        (
            vf_r13_live_owners(),
            vf_r13_cleanup_errors(),
            DROP_ERROR.load(Ordering::SeqCst),
        )
    }
}
pub fn api_calls() -> u64 {
    unsafe { vf_r13_cuda_api_calls() }
}

/// Address of the exact linked create symbol, used only to attest the loader inode.
pub(crate) fn library_symbol() -> *const libc::c_void {
    vf_r13_create as *const () as *const libc::c_void
}
