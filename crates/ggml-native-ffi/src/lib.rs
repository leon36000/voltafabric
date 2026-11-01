#![deny(unsafe_op_in_unsafe_fn)]
use std::{
    ffi::{c_void, CString},
    fs::File,
    ptr::NonNull,
};

#[cfg(target_os = "linux")]
#[repr(C)]
struct GgufInitParams {
    no_alloc: bool,
    ctx: *mut *mut c_void,
}

#[cfg(target_os = "linux")]
#[repr(C)]
struct GgufContext {
    _private: [u8; 0],
}

#[cfg(target_os = "linux")]
unsafe extern "C" {
    fn gguf_init_from_file(
        path: *const std::ffi::c_char,
        params: GgufInitParams,
    ) -> *mut GgufContext;
    fn gguf_free(ctx: *mut GgufContext);
    fn gguf_get_data_offset(ctx: *const GgufContext) -> usize;
    fn gguf_find_tensor(ctx: *const GgufContext, name: *const std::ffi::c_char) -> i64;
    fn gguf_get_tensor_offset(ctx: *const GgufContext, tensor_id: i64) -> usize;
    fn gguf_get_tensor_size(ctx: *const GgufContext, tensor_id: i64) -> usize;
    fn gguf_get_tensor_type(ctx: *const GgufContext, tensor_id: i64) -> i32;
    fn gguf_get_tensor_ne(ctx: *const GgufContext, tensor_id: i64) -> *const i64;
}

/// Metadata-only span for one tensor in a GGUF file, using the pinned GGUF reader.
#[cfg(target_os = "linux")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GgufTensorSpan {
    pub offset: u64,
    pub bytes: u64,
    pub ggml_type: i32,
    /// GGUF/ggml extents in physical order; unused dimensions are one.
    pub dimensions: [i64; 4],
}

#[cfg(target_os = "linux")]
struct GgufGuard(NonNull<GgufContext>);

#[cfg(target_os = "linux")]
impl Drop for GgufGuard {
    fn drop(&mut self) {
        // SAFETY: this guard uniquely owns the context returned by gguf_init_from_file.
        unsafe { gguf_free(self.0.as_ptr()) }
    }
}

/// Read tensor metadata without allocating its tensor data. The descriptor is already
/// open and regular; `/proc/self/fd` makes the pinned GGUF parser inspect that same inode.
#[cfg(target_os = "linux")]
pub fn gguf_tensor_span(file: &File, name: &str) -> Result<GgufTensorSpan, &'static str> {
    use std::os::fd::AsRawFd;
    let descriptor_path = CString::new(format!("/proc/self/fd/{}", file.as_raw_fd()))
        .map_err(|_| "invalid checkpoint descriptor")?;
    let tensor_name = CString::new(name).map_err(|_| "GGUF tensor name contains NUL")?;
    // SAFETY: paths remain NUL-terminated and valid during the call. `ctx` is null and
    // `no_alloc=true`, so GGUF reads only bounded metadata and leaves tensor bytes on disk.
    let raw = unsafe {
        gguf_init_from_file(
            descriptor_path.as_ptr(),
            GgufInitParams {
                no_alloc: true,
                ctx: std::ptr::null_mut(),
            },
        )
    };
    let ctx = GgufGuard(NonNull::new(raw).ok_or("pinned GGUF metadata parse failed")?);
    // SAFETY: the context guard stays live and the tensor ID is checked below.
    let id = unsafe { gguf_find_tensor(ctx.0.as_ptr(), tensor_name.as_ptr()) };
    if id < 0 {
        return Err("GGUF tensor name is absent");
    }
    // SAFETY: the nonnegative ID came from gguf_find_tensor for this live context.
    let (relative, bytes, ggml_type, ne) = unsafe {
        let ne = gguf_get_tensor_ne(ctx.0.as_ptr(), id);
        if ne.is_null() {
            return Err("GGUF tensor dimensions are unavailable");
        }
        let dimensions = [*ne, *ne.add(1), *ne.add(2), *ne.add(3)];
        (
            gguf_get_tensor_offset(ctx.0.as_ptr(), id),
            gguf_get_tensor_size(ctx.0.as_ptr(), id),
            gguf_get_tensor_type(ctx.0.as_ptr(), id),
            dimensions,
        )
    };
    // SAFETY: `ctx` remains live under the guard until the offset is copied out.
    let absolute = unsafe { gguf_get_data_offset(ctx.0.as_ptr()) }
        .checked_add(relative)
        .ok_or("GGUF tensor offset overflow")?;
    if absolute > 128 * 1024 * 1024 {
        return Err("GGUF metadata header exceeds 128 MiB");
    }
    Ok(GgufTensorSpan {
        offset: u64::try_from(absolute).map_err(|_| "GGUF offset does not fit u64")?,
        bytes: u64::try_from(bytes).map_err(|_| "GGUF tensor size does not fit u64")?,
        ggml_type,
        dimensions: ne,
    })
}

unsafe extern "C" {
    #[cfg(feature = "scalar-reference")]
    fn vf_native_scalar_prepare(
        input: *const f32,
        input_len: usize,
        output: *mut f32,
        output_len: usize,
    ) -> i32;
    #[cfg(feature = "scalar-reference")]
    fn vf_native_scalar_reference_prepared(
        gu: i32,
        down: i32,
        blob: *const u8,
        blob_len: usize,
        input: *const f32,
        input_len: usize,
        output: *mut f64,
        output_len: usize,
    ) -> i32;
    #[cfg(feature = "scalar-reference")]
    fn vf_native_scalar_reference(
        gu: i32,
        down: i32,
        blob: *const u8,
        blob_len: usize,
        input: *const f32,
        input_len: usize,
        output: *mut f64,
        output_len: usize,
    ) -> i32;
    fn vf_native_create(gu: i32, down: i32, bytes: *const u8, len: usize) -> *mut c_void;
    fn vf_native_forward(
        instance: *mut c_void,
        input: *const f32,
        input_len: usize,
        output: *mut f32,
        output_len: usize,
    ) -> i32;
    fn vf_native_destroy(instance: *mut c_void);
    fn vf_native_prepare(gu: i32, input: *const f32, input_len: usize) -> *mut c_void;
    fn vf_native_forward_prepared(
        instance: *mut c_void,
        prepared: *const c_void,
        output: *mut f32,
        output_len: usize,
    ) -> i32;
    fn vf_native_prepared_destroy(prepared: *mut c_void);

    fn vf_native_matrix_create(
        kind: i32,
        rows: usize,
        columns: usize,
        bytes: *const u8,
        len: usize,
    ) -> *mut c_void;
    fn vf_native_matrix_forward(
        instance: *mut c_void,
        input: *const f32,
        input_len: usize,
        output: *mut f32,
        output_len: usize,
    ) -> i32;
    fn vf_native_matrix_destroy(instance: *mut c_void);
    fn vf_native_routed_matrix_create(
        kind: i32,
        experts: usize,
        rows: usize,
        columns: usize,
        bytes: *const u8,
        len: usize,
    ) -> *mut c_void;
    fn vf_native_routed_matrix_forward(
        instance: *mut c_void,
        slot_inputs: *const f32,
        slot_input_len: usize,
        expert_ids: *const i32,
        slots: usize,
        slot_outputs: *mut f32,
        output_len: usize,
    ) -> i32;
    fn vf_native_routed_matrix_destroy(instance: *mut c_void);
}

/// Decoded Q8_K activation scoped by the caller to one token and batch.
/// Its private contents can only be made by the checked scalar preparation.
#[cfg(feature = "scalar-reference")]
pub struct NativeScalarPrepared {
    activation: Vec<f32>,
}

#[cfg(feature = "scalar-reference")]
pub fn scalar_prepare(input: &[f32]) -> Result<NativeScalarPrepared, &'static str> {
    if input.len() != 2560 || input.iter().any(|v| !v.is_finite()) {
        return Err("invalid scalar activation");
    }
    let mut activation = vec![0f32; 2560];
    // SAFETY: checked immutable input and unique output remain valid for the call.
    let status = unsafe {
        vf_native_scalar_prepare(
            input.as_ptr(),
            input.len(),
            activation.as_mut_ptr(),
            activation.len(),
        )
    };
    if status != 0 || activation.iter().any(|x| !x.is_finite()) {
        return Err("scalar activation preparation refused");
    }
    Ok(NativeScalarPrepared { activation })
}

#[cfg(feature = "scalar-reference")]
pub fn scalar_reference_prepared(
    gu: i32,
    down: i32,
    blob: &[u8],
    prepared: &NativeScalarPrepared,
) -> Result<Vec<f64>, &'static str> {
    if blob.is_empty() || blob.len() > 4 * 1024 * 1024 {
        return Err("invalid scalar reference input");
    }
    let mut output = vec![0f64; 2560];
    // SAFETY: blob and the private prepared activation are immutable; output is unique.
    let status = unsafe {
        vf_native_scalar_reference_prepared(
            gu,
            down,
            blob.as_ptr(),
            blob.len(),
            prepared.activation.as_ptr(),
            prepared.activation.len(),
            output.as_mut_ptr(),
            output.len(),
        )
    };
    if status != 0 || output.iter().any(|x| !x.is_finite()) {
        return Err("scalar reference refused prepared input");
    }
    Ok(output)
}

/// Diagnostic-only unweighted SC117 f64 values; caller authenticates bytes and format.
#[cfg(feature = "scalar-reference")]
pub fn scalar_reference(
    gu: i32,
    down: i32,
    blob: &[u8],
    input: &[f32],
) -> Result<Vec<f64>, &'static str> {
    if blob.is_empty()
        || blob.len() > 4 * 1024 * 1024
        || input.len() != 2560
        || input.iter().any(|x| !x.is_finite())
    {
        return Err("invalid scalar reference input");
    }
    let mut output = vec![0f64; 2560];
    // SAFETY: immutable input slices and uniquely borrowed output remain valid and
    // disjoint throughout this call; C++ copies output only after complete success.
    let status = unsafe {
        vf_native_scalar_reference(
            gu,
            down,
            blob.as_ptr(),
            blob.len(),
            input.as_ptr(),
            input.len(),
            output.as_mut_ptr(),
            output.len(),
        )
    };
    if status != 0 || output.iter().any(|x| !x.is_finite()) {
        return Err("scalar reference refused input");
    }
    Ok(output)
}

/// An opaque CPU resident; the native library copies the blob once on load.
/// This wrapper does not authenticate blobs or establish model identity.
pub struct NativeResident {
    ptr: NonNull<c_void>,
    gate_up_type: i32,
}

/// An opaque finite activation quantized for one gate/up format; caller must
/// scope it to the intended token and batch. It contains no model weights.
pub struct NativePrepared {
    ptr: NonNull<c_void>,
    gate_up_type: i32,
}
impl Drop for NativePrepared {
    fn drop(&mut self) {
        // SAFETY: the prepared pointer is uniquely owned and destroyed once.
        unsafe { vf_native_prepared_destroy(self.ptr.as_ptr()) }
    }
}
impl NativeResident {
    pub fn load(gu: i32, down: i32, blob: &[u8]) -> Result<Self, &'static str> {
        // SAFETY: the immutable slice remains valid during the call; C++ copies it.
        let raw = unsafe { vf_native_create(gu, down, blob.as_ptr(), blob.len()) };
        Ok(Self {
            ptr: NonNull::new(raw).ok_or("native load refused")?,
            gate_up_type: gu,
        })
    }
    pub fn gate_up_type(&self) -> i32 {
        self.gate_up_type
    }
    pub fn prepare(&self, input: &[f32]) -> Result<NativePrepared, &'static str> {
        if input.len() != 2560 || input.iter().any(|v| !v.is_finite()) {
            return Err("invalid native input");
        }
        // SAFETY: input is a valid immutable slice; native code copies the prepared data.
        let raw = unsafe { vf_native_prepare(self.gate_up_type, input.as_ptr(), input.len()) };
        Ok(NativePrepared {
            ptr: NonNull::new(raw).ok_or("native preparation failed")?,
            gate_up_type: self.gate_up_type,
        })
    }
    pub fn forward_prepared(
        &mut self,
        prepared: &NativePrepared,
    ) -> Result<Vec<f32>, &'static str> {
        if prepared.gate_up_type != self.gate_up_type {
            return Err("prepared native format mismatch");
        }
        let mut output = vec![0_f32; 2560];
        // SAFETY: both opaque pointers are alive; resident is exclusively borrowed,
        // prepared input is immutable, and output is a disjoint correctly sized slice.
        let status = unsafe {
            vf_native_forward_prepared(
                self.ptr.as_ptr(),
                prepared.ptr.as_ptr(),
                output.as_mut_ptr(),
                output.len(),
            )
        };
        if status != 0 || output.iter().any(|v| !v.is_finite()) {
            return Err("native execution failed");
        }
        Ok(output)
    }
    pub fn forward(&mut self, input: &[f32]) -> Result<Vec<f32>, &'static str> {
        if input.len() != 2560 || input.iter().any(|v| !v.is_finite()) {
            return Err("invalid native input");
        }
        let mut output = vec![0_f32; 2560];
        // SAFETY: instance was created by this library and is exclusively borrowed;
        // input and output are disjoint valid slices of the specified lengths.
        let status = unsafe {
            vf_native_forward(
                self.ptr.as_ptr(),
                input.as_ptr(),
                input.len(),
                output.as_mut_ptr(),
                output.len(),
            )
        };
        if status != 0 || output.iter().any(|v| !v.is_finite()) {
            return Err("native execution failed");
        }
        Ok(output)
    }
}
impl Drop for NativeResident {
    fn drop(&mut self) {
        // SAFETY: pointer is owned uniquely and destroyed exactly once.
        unsafe { vf_native_destroy(self.ptr.as_ptr()) }
    }
}

/// One bounded resident GGML matrix; caller authenticates the source and format.
pub struct NativeMatrix {
    ptr: NonNull<c_void>,
    rows: usize,
    columns: usize,
}
impl NativeMatrix {
    pub fn load(
        kind: i32,
        rows: usize,
        columns: usize,
        bytes: &[u8],
    ) -> Result<Self, &'static str> {
        // SAFETY: immutable bytes remain valid during the call and native code copies them.
        let raw =
            unsafe { vf_native_matrix_create(kind, rows, columns, bytes.as_ptr(), bytes.len()) };
        Ok(Self {
            ptr: NonNull::new(raw).ok_or("native matrix refused")?,
            rows,
            columns,
        })
    }
    pub fn forward(&mut self, input: &[f32]) -> Result<Vec<f32>, &'static str> {
        if input.len() != self.columns || input.iter().any(|v| !v.is_finite()) {
            return Err("invalid matrix input");
        }
        let mut output = vec![0.0; self.rows];
        // SAFETY: uniquely owned instance and disjoint bounded input/output slices.
        let status = unsafe {
            vf_native_matrix_forward(
                self.ptr.as_ptr(),
                input.as_ptr(),
                input.len(),
                output.as_mut_ptr(),
                output.len(),
            )
        };
        if status != 0 || output.iter().any(|v| !v.is_finite()) {
            return Err("native matrix execution failed");
        }
        Ok(output)
    }
}
impl Drop for NativeMatrix {
    fn drop(&mut self) {
        // SAFETY: instance is uniquely owned and destroyed exactly once.
        unsafe { vf_native_matrix_destroy(self.ptr.as_ptr()) }
    }
}

/// Resident stack of same-format expert matrices for CPU GGML_OP_MUL_MAT_ID.
/// It returns raw, unweighted FP32 rows in caller slot order.
pub struct NativeRoutedMatrix {
    ptr: NonNull<c_void>,
    experts: usize,
    rows: usize,
    columns: usize,
}
impl NativeRoutedMatrix {
    /// Copy one bounded matrix stack into resident native storage exactly once.
    pub fn load(
        kind: i32,
        experts: usize,
        rows: usize,
        columns: usize,
        bytes: &[u8],
    ) -> Result<Self, &'static str> {
        if experts == 0
            || experts > 512
            || rows == 0
            || rows > 10240
            || columns == 0
            || columns > 6144
            || bytes.is_empty()
            || bytes.len() > 1_250_000_000
        {
            return Err("invalid routed matrix binding");
        }
        // SAFETY: bytes are immutable for the call; native code validates and copies them.
        let raw = unsafe {
            vf_native_routed_matrix_create(
                kind,
                experts,
                rows,
                columns,
                bytes.as_ptr(),
                bytes.len(),
            )
        };
        Ok(Self {
            ptr: NonNull::new(raw).ok_or("native routed matrix refused")?,
            experts,
            rows,
            columns,
        })
    }

    /// Run a compact slot batch; IDs index the physical resident matrix order.
    pub fn forward(
        &mut self,
        slot_inputs: &[f32],
        expert_ids: &[u32],
    ) -> Result<Vec<f32>, &'static str> {
        let input_len = expert_ids
            .len()
            .checked_mul(self.columns)
            .ok_or("routed input size overflow")?;
        let output_len = expert_ids
            .len()
            .checked_mul(self.rows)
            .ok_or("routed output size overflow")?;
        if expert_ids.is_empty()
            || expert_ids.len() > 512
            || slot_inputs.len() != input_len
            || slot_inputs.iter().any(|x| !x.is_finite())
            || expert_ids.iter().any(|&id| id as usize >= self.experts)
        {
            return Err("invalid routed matrix batch");
        }
        let ids = expert_ids
            .iter()
            .map(|&id| i32::try_from(id).map_err(|_| "expert ID overflow"))
            .collect::<Result<Vec<_>, _>>()?;
        let mut output = vec![0.0; output_len];
        // SAFETY: the resident is uniquely borrowed; all slices are valid for the synchronous call,
        // and the native bridge stages results before writing this unique output buffer.
        let status = unsafe {
            vf_native_routed_matrix_forward(
                self.ptr.as_ptr(),
                slot_inputs.as_ptr(),
                slot_inputs.len(),
                ids.as_ptr(),
                ids.len(),
                output.as_mut_ptr(),
                output.len(),
            )
        };
        if status != 0 || output.iter().any(|x| !x.is_finite()) {
            return Err("native routed matrix execution failed");
        }
        Ok(output)
    }
}
impl Drop for NativeRoutedMatrix {
    fn drop(&mut self) {
        // SAFETY: the opaque native resident is uniquely owned and destroyed once.
        unsafe { vf_native_routed_matrix_destroy(self.ptr.as_ptr()) }
    }
}
