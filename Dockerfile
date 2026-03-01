# We use Ubuntu 22.04 because OpenFHE is most stable here
FROM ubuntu:22.04

# 1. Install standard tools & C++ compilers
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    build-essential cmake git curl libgmp-dev libssl-dev clang \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Rust (Official Script)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# 3. Clone & Compile OpenFHE (C++ Backend)
# We LOCK the version to v1.2.2 to prevent breaking changes
WORKDIR /opt
RUN git clone https://github.com/openfheorg/openfhe-development.git && \
    cd openfhe-development && \
    git checkout v1.2.2 && \
    mkdir build && cd build && \
    cmake .. -DBUILD_SHARED=ON -DINSTALL_LIB_DIR=/usr/lib -DINSTALL_INCLUDE_DIR=/usr/include/openfhe_core && \
    make -j$(nproc) && \
    make install

# 4. Pre-download the Rust Wrapper (Cache Step)
# This prevents re-downloading the internet every time you change your code
WORKDIR /app
RUN cargo new --lib cache_warmer
WORKDIR /app/cache_warmer
RUN echo 'openfhe = { git = "https://github.com/fairmath/openfhe-rs" }' >> Cargo.toml
RUN cargo build || true

# 5. Final Setup
WORKDIR /app/project
CMD ["bash"]
