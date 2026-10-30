#![cfg(feature = "native-ggml-cpu")]
use voltafabric::{
    native_ggml_cpu::{MatrixBinding, QuantizedMatrix},
    sha256,
};
#[test]
fn bound_matrices_are_resident_and_verify_identity() {
    for (kind, row_bytes) in [(8, 272), (12, 144), (13, 176), (14, 210), (23, 136)] {
        let bytes = vec![0_u8; row_bytes * 3];
        let binding = MatrixBinding {
            model_revision: "model-observed".into(),
            layer: 0,
            tensor: "blk.0.attn_qkv.weight".into(),
            ggml_type: kind,
            rows: 3,
            columns: 256,
            sha256: sha256::hex(&bytes),
        };
        let mut resident = QuantizedMatrix::load(binding.clone(), &bytes).unwrap();
        let first = resident.forward(&vec![0.125; 256]).unwrap();
        assert_eq!(first, vec![0.0; 3]);
        assert_eq!(resident.forward(&vec![0.125; 256]).unwrap(), first);
        assert_eq!(resident.binding(), &binding);
        assert!(resident.forward(&[f32::NAN; 256]).is_err());
        assert!(resident.forward(&[0.0; 255]).is_err());
        let mut altered = bytes.clone();
        altered[0] = 1;
        assert!(QuantizedMatrix::load(binding.clone(), &altered).is_err());
        let mut wrong = binding.clone();
        wrong.columns = 255;
        assert!(QuantizedMatrix::load(wrong, &bytes).is_err());
        let mut wrong = binding;
        wrong.ggml_type = 39;
        assert!(QuantizedMatrix::load(wrong, &bytes).is_err());
    }
}
