use actix_web::{post, web, App, HttpServer, HttpResponse, Responder};
use base64::{engine::general_purpose, Engine as _};
use serde::Deserialize;
use std::sync::RwLock;
use std::fs;
use openfhe::cxx::{UniquePtr, let_cxx_string};
use openfhe::ffi;

// --- 1. Thread-Safe Wrapper for OpenFHE ---
// CXX opaque types are !Send by default. We wrap the UniquePtr so Actix-Web 
// can safely share the FHE math engine across its multiple async worker threads.
pub struct FheContext(pub UniquePtr<ffi::CryptoContextDCRTPoly>);
unsafe impl Send for FheContext {}
unsafe impl Sync for FheContext {}

// --- 2. JSON Payloads ---
#[derive(Deserialize)]
struct KeysPayload {
    context: String,
    mult_keys: String,
    rot_keys: String,
}

#[derive(Deserialize)]
struct VectorPayload {
    vector: Vec<f64>, // Placeholder until we serialize the actual ciphertext
}

// --- 3. The Server's Memory State ---
struct AppState {
    context: RwLock<Option<FheContext>>,
    keys_loaded: RwLock<bool>,
}

// --- 4. The Endpoints ---
#[post("/update_keys")]
async fn update_keys(payload: web::Json<KeysPayload>, data: web::Data<AppState>) -> impl Responder {
    println!("Cloud Server: Received Evaluation Keys from Edge Device!");

    // 1. Decode the Base64 strings back into raw C++ binary bytes
    let ctx_bytes = general_purpose::STANDARD.decode(&payload.context).unwrap();
    let mult_bytes = general_purpose::STANDARD.decode(&payload.mult_keys).unwrap();
    let rot_bytes = general_purpose::STANDARD.decode(&payload.rot_keys).unwrap();

    fs::write("server_cc.bin", ctx_bytes).expect("Failed to write context file");
    fs::write("server_mult.bin", mult_bytes).expect("Failed to write mult key file");
    fs::write("server_rot.bin", rot_bytes).expect("Failed to write rot key file");

    println!("Cloud Server: Deserializing OpenFHE Objects into RAM...");
    
    // 2. The Dummy Context Hack
    // We cannot instantiate an empty Opaque C++ type, so we generate a quick dummy 
    // context and pass its memory pointer to the deserializer to be overwritten.
    let dummy_params = ffi::GenParamsCKKSRNS();
    let mut deserialized_context = ffi::DCRTPolyGenCryptoContextByParamsCKKSRNS(&dummy_params);
    
    let_cxx_string!(cc_path = "server_cc.bin");

    // 3. Deserialize directly into the pinned pointer
    let cc_success = ffi::DCRTPolyDeserializeCryptoContextFromFile(
        &cc_path, 
        deserialized_context.pin_mut(),
        ffi::SerialMode::BINARY
    );

    if cc_success {
        // 4. Lock the context into the Actix Web State using our Thread-Safe Wrapper
        let mut state_context = data.context.write().unwrap();
        *state_context = Some(FheContext(deserialized_context));
        
        let mut keys_loaded = data.keys_loaded.write().unwrap();
        *keys_loaded = true;

        println!("Cloud Server: Keys successfully locked into memory. Ready for blind math!");
        HttpResponse::Ok().json("FHE Keys synchronized successfully.")
    } else {
        println!("Cloud Server ERROR: Failed to deserialize FHE parameters.");
        HttpResponse::InternalServerError().json("Failed to reconstruct FHE mathematics.")
    }
}

#[post("/enroll")]
async fn enroll(payload: web::Json<VectorPayload>, data: web::Data<AppState>) -> impl Responder {
    let keys_loaded = data.keys_loaded.read().unwrap();
    if !*keys_loaded {
        return HttpResponse::ServiceUnavailable().json("Server is not ready. Missing FHE keys.");
    }
    println!("Cloud Server: Received ENROLLMENT request. Preparing to homomorphically shift and add...");
    HttpResponse::Ok().json("Server successfully received padded vector for database enrollment.")
}

#[post("/search")]
async fn search(payload: web::Json<VectorPayload>, data: web::Data<AppState>) -> impl Responder {
    let keys_loaded = data.keys_loaded.read().unwrap();
    if !*keys_loaded {
        return HttpResponse::ServiceUnavailable().json("Server is not ready. Missing FHE keys.");
    }
    println!("Cloud Server: Received SEARCH request. Preparing 1-to-N homomorphic multiplication...");
    HttpResponse::Ok().json("Server successfully received duplicated vector for blind search.")
}

// --- 5. The Server Boot Sequence ---
#[actix_web::main]
async fn main() -> std::io::Result<()> {
    println!("--- Cloud Server Booting Up ---");
    println!("Awaiting FHE Evaluation Keys on port 8080...");

    let app_state = web::Data::new(AppState {
        context: RwLock::new(None),
        keys_loaded: RwLock::new(false),
    });

    // Move the app_state into the closure so multiple worker threads can access it safely
    HttpServer::new(move || {
        App::new()
            .app_data(app_state.clone()) 
            .service(update_keys)
            .service(enroll)
            .service(search)
    })
    .bind(("0.0.0.0", 8080))?
    .run()
    .await
}