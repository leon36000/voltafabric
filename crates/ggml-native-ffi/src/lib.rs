#![deny(unsafe_op_in_unsafe_fn)]
use std::{ffi::c_void, ptr::NonNull};

unsafe extern "C" {
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
