//! Experimental single-process native GGML owner, separate from MXFP4/QPN2.
//! Bindings are caller-provided evidence, not a distributed lease or Session.
use crate::{
    invalid, sha256,
    strata_native_layout::{NativeLayout, VerifiedNativeExpert},
    Result,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Binding {
    pub model_revision: String,
    pub rank: u32,
    pub layer: u32,
    pub expert: u32,
    pub local: u32,
    pub epoch: u64,
    pub index_sha256: String,
    pub blob_sha256: String,
}

pub struct Resident {
    binding: Binding,
    owner: vf_ggml_native_ffi::NativeResident,
}
impl Resident {
    /// Recheck the complete local binding and payload hash before native load.
    /// Callers must obtain the expected digests from a trusted model manifest.
    pub fn load(
        layout: &NativeLayout,
        expert: VerifiedNativeExpert,
        binding: Binding,
    ) -> Result<Self> {
        let expected = layout.expert(binding.layer as usize, u64::from(binding.expert))?;
        if binding.model_revision.trim().is_empty()
            || binding.model_revision.len() > 256
            || expert.layer != binding.layer as usize
            || expert.expert != u64::from(binding.expert)
            || expert.span != expected
            || binding.index_sha256 != layout.index_sha256()
            || expert.bytes.len() as u64 != expected.bytes
            || sha256::hex(&expert.bytes) != binding.blob_sha256
        {
            return Err(invalid("native expert binding or payload mismatch"));
        }
        let owner = vf_ggml_native_ffi::NativeResident::load(
            expected.gate_up_type as i32,
            expected.down_type as i32,
            &expert.bytes,
        )
        .map_err(invalid)?;
        Ok(Self { binding, owner })
    }
    pub fn binding(&self) -> &Binding {
        &self.binding
    }
    /// Returns one unweighted FP32 expert row; no gate is applied here.
    pub fn forward(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        self.owner.forward(input).map_err(invalid)
    }
}
