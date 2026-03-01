use base64::{engine::general_purpose, Engine as _};
use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use openfhe::cxx::{CxxVector, UniquePtr, let_cxx_string};
use openfhe::ffi;
use std::fs;

#[pyclass(unsendable)]
pub struct BiometricEngine {
    context: UniquePtr<ffi::CryptoContextDCRTPoly>,
    key_pair: UniquePtr<ffi::KeyPairDCRTPoly>,
}

#[pymethods]
impl BiometricEngine {
    #[new]
    pub fn new() -> PyResult<Self> {
        println!("Rust/PyO3: Initializing Advanced SIMD CKKS Context...");
        
        let mut params = ffi::GenParamsCKKSRNS();
        // We increase the multiplicative depth to 2 to support the masking and rotation math
        params.pin_mut().SetMultiplicativeDepth(2); 
        params.pin_mut().SetScalingModSize(50);
        // We expand the batch size to 8192 slots so the server can pack 16 faces (16 * 512) into one ciphertext!
        params.pin_mut().SetBatchSize(8192); 

        let context = ffi::DCRTPolyGenCryptoContextByParamsCKKSRNS(&params);
        context.EnableByMask(ffi::PKESchemeFeature::PKE.repr as u32);
        context.EnableByMask(ffi::PKESchemeFeature::KEYSWITCH.repr as u32);
        context.EnableByMask(ffi::PKESchemeFeature::LEVELEDSHE.repr as u32);
        // ADVANCEDSHE is strictly required to enable Galois Rotations
        context.EnableByMask(ffi::PKESchemeFeature::ADVANCEDSHE.repr as u32); 

        println!("Rust/PyO3: Generating Master Keys...");
        let key_pair = context.KeyGen();

        println!("Rust/PyO3: Generating EvalMult (Multiplication) Keys...");
        context.EvalMultKeyGen(&key_pair.GetPrivateKey());

        println!("Rust/PyO3: Generating Galois (Rotation) Keys for 512-slot sums...");
        // To sum up 512 elements, the server uses a divide-and-conquer rotate-and-add algorithm.
        // This requires rotation keys for powers of 2 (1, 2, 4, 8, 16, 32, 64, 128, 256).
        let mut rot_indices = CxxVector::<i32>::new();
        for i in 0..9 {
            rot_indices.pin_mut().push(1 << i);
        }
        context.EvalRotateKeyGen(&key_pair.GetPrivateKey(), rot_indices.as_ref().unwrap(), &key_pair.GetPublicKey());

        println!("Rust/PyO3: SIMD Engine Ready and Locked into RAM!");

        Ok(BiometricEngine {
            context,
            key_pair,
        })
    }

    /// 1. Export the CryptoContext Parameters
    pub fn export_context_base64(&self) -> PyResult<String> {
        println!("Rust/PyO3: Serializing CryptoContext...");
        
        // Safely construct a C++ string
        let_cxx_string!(path = "cc_temp.bin");
        
        // Use the explicit DCRTPoly prefix and SerialMode::BINARY
        if !ffi::DCRTPolySerializeCryptoContextToFile(&path, &self.context, ffi::SerialMode::BINARY) {
             return Err(PyRuntimeError::new_err("Failed to serialize CryptoContext"));
        }
        
        let bytes = std::fs::read("cc_temp.bin")?;
        Ok(general_purpose::STANDARD.encode(&bytes))
    }

    /// 2. Export the Multiplication Keys
    pub fn export_mult_keys_base64(&self) -> PyResult<String> {
        println!("Rust/PyO3: Serializing EvalMult Keys...");
        
        // Safely construct a C++ string
        let_cxx_string!(path = "mult_temp.bin");
        
        // Use the explicit DCRTPoly prefix, pass the context, and use SerialMode
        if !ffi::DCRTPolySerializeEvalMultKeyToFile(&path, &self.context, ffi::SerialMode::BINARY) {
             return Err(PyRuntimeError::new_err("Failed to serialize EvalMultKey"));
        }
        
        let bytes = fs::read("mult_temp.bin")?;
        Ok(general_purpose::STANDARD.encode(&bytes))
    }

    /// 3. Export the Galois Rotation Keys
    pub fn export_rot_keys_base64(&self) -> PyResult<String> {
        println!("Rust/PyO3: Serializing Galois Rotation Keys...");
        
        let_cxx_string!(path = "rot_temp.bin");
        
        if !ffi::DCRTPolySerializeEvalAutomorphismKeyToFile(&path, &self.context, ffi::SerialMode::BINARY) {
             return Err(PyRuntimeError::new_err("Failed to serialize Galois Keys"));
        }
        
        let bytes = fs::read("rot_temp.bin")?;
        Ok(general_purpose::STANDARD.encode(&bytes))
    }

    /// (Placeholder) We will update this to serialize the ciphertext in the next step!
    pub fn encrypt(&self, py_vector: Vec<f64>) -> PyResult<String> {
        Ok(format!("Ready to pack {} features into 8192 slots.", py_vector.len()))
    }
}

#[pymodule]
fn fhe_client(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<BiometricEngine>()?;
    Ok(())
}