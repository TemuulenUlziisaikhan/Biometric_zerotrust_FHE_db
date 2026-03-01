fn main() {
    // Point the linker to the Ubuntu installation folder
    println!("cargo:rustc-link-arg=-L/usr/local/lib");
    
    // Link the required OpenFHE libraries in the EXACT order the C++ compiler demands
    println!("cargo:rustc-link-arg=-lOPENFHEpke");
    println!("cargo:rustc-link-arg=-lOPENFHEbinfhe"); 
    println!("cargo:rustc-link-arg=-lOPENFHEcore");
    
    // ink OpenMP for parallel FHE mathematics
    println!("cargo:rustc-link-arg=-fopenmp");
    
    // library paths directly into the Python .so file 
    println!("cargo:rustc-link-arg=-Wl,-rpath=/usr/local/lib");
}