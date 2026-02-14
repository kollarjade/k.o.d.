# JAIDE v40 — Relational AGI System

# Complete Technical Documentation

JAIDE (v40) is an artificial general intelligence  designed from first principles, implementing a custom cognitive architecture based on the Kalmár-Riesz-Gábor-Unity (KRGU) framework, a Self-Similar Relational Graph (SSRG) for dynamic knowledge representation, quantum-inspired state logic for ambiguity handling, and an Entangled Stochastic Symmetry Optimizer (ESSO) for learning. The system is written in Zig for performance-critical components with Futhark for GPU acceleration, Python for orchestration, and includes a comprehensive formal verification suite spanning Agda, Lean 4, Isabelle/HOL, TLA+, Viper, and SPIN. Training runs on Modal cloud infrastructure using 8x NVIDIA B200 GPUs with the SZTAKI-HLT/HunSum-1 Hungarian dataset (1.1M samples), and the system deploys via a Flask web server for interactive inference.

This document provides a complete, file-by-file technical description of every source file in the project. Each file has been read in full, and the descriptions below reflect the actual content — struct names, function signatures, constants, algorithms, and interconnections as they appear in the code.

---

## 1. Top-Level Files

### server.py — Flask Web Server

The server.py file implements a Flask-based web server that serves as the primary user-facing gateway to the JAIDE v40 system. It imports standard library modules for subprocess management, threading, timing, OS interaction, and signal handling, alongside Flask's core utilities for rendering templates and handling JSON requests. The server defines a binary path pointing to "./src/main," which is the compiled Zig executable that constitutes the JAIDE core runtime. Upon startup, the script attempts to launch this binary as a subprocess using Python's Popen, piping stdin, stdout, and stderr for bidirectional communication. The startup routine waits up to thirty seconds for the subprocess to emit a "READY" signal on its standard output, after which the process is considered initialized. A global lock protects access to the subprocess handle, ensuring thread safety when multiple HTTP requests arrive concurrently.

The file contains an extensive inline HTML template that renders a retro-styled terminal interface in the browser. The design uses a black background with bright green monospace text, scanline overlays, and a glowing header reading "JAIDE v40 :: AGI ROOT TERMINAL," evoking a classic hacker console aesthetic. The JavaScript embedded in the template captures user input from a text field, sends it as a JSON POST request to the "/interact" endpoint, and displays the returned response as a new entry in a scrollable terminal log. User messages appear right-aligned in white, system responses glow green on the left, and errors are rendered in red. The CSS includes smooth fade-in animations, custom scrollbar styling, and responsive layout rules.

The Flask application defines two routes. The root route simply renders the HTML template string. The "/interact" POST endpoint extracts the user's input from the JSON body, checks whether the JAIDE subprocess is alive and ready, and if not, attempts to restart it. It then writes the user's command to the subprocess's stdin, flushes, and reads a response line from stdout within a ten-second timeout window. The response is returned as JSON. If the subprocess has died or times out, appropriate error messages are returned. The server binds to all interfaces on port 5000 with debug mode disabled, making it suitable for deployment in a cloud environment like Replit.

### main.py — Python Entry Point

The main.py file is a minimal Python entry point that serves as a simple placeholder or default script for the project workspace. It defines a single function called main that prints the string "Hello from repl-nix-workspace!" to standard output. The file follows the standard Python idiom of checking whether it is being run as the main module via the __name__ guard before invoking the main function. This file does not contain any substantive application logic. It exists primarily as a default executable for the Python environment. The actual application logic for JAIDE is handled by server.py, which launches the compiled Zig binary and serves the web interface.

### generate_recreate_script.py — Project Recreation Script

The generate_recreate_script.py file is a utility that programmatically generates a shell script capable of recreating the entire JAIDE v40 project structure from scratch. It uses Python's os.walk to traverse the entire project directory tree, collecting every file's path and content. For each file encountered, it generates a corresponding mkdir -p command for the parent directory and a heredoc block (using a unique delimiter) that writes the file's exact contents. Binary files are handled separately using base64 encoding with a decode pipe. The output is a single self-contained bash script that, when executed on a fresh system, will recreate every directory, every source file, every configuration file, and every binary asset in the project exactly as they exist at the time of generation. This is useful for project backup, migration between systems, and reproducible builds.

### build.zig — Zig Build System

The build.zig file is the central build configuration for the entire Zig codebase, using the standard Zig build system API. It targets the native platform with a ReleaseSafe optimization mode and defines five separate executable targets. The first is "jaide," the main CPU training and inference binary, compiled from src/main.zig. The second is "jaide-gpu," compiled from src/main_gpu.zig for single-GPU training with Futhark acceleration. The third is "jaide-distributed," compiled from src/main_distributed.zig for multi-GPU distributed training. The fourth is "jaide-distributed-futhark," compiled from src/main_distributed_futhark.zig for Futhark-accelerated distributed training. The fifth is "jaide-inference-server," compiled from src/inference_server_main.zig for the production inference API server. All five executables share the same addModule configuration, which registers the core modules (tensor, memory, types, io, model_io), the processor modules (rsf, mgt, sfd, ssi, ranker), the distributed modules, the hardware acceleration modules, and the core_relational modules as importable packages.

Each executable is registered with b.installArtifact so that "zig build" will compile and install all of them. The file also defines two run steps: "run" for the main executable and "run-distributed" for the distributed trainer, both depending on the install step to ensure the binaries are built before execution. Additionally, three test configurations are defined: a general test step that runs the test suite embedded in src/main.zig, a "test-tensor" step that runs tests from src/core/tensor.zig, and a "test-memory" step for src/core/memory.zig.

### build.zig.zon — Package Manifest

The build.zig.zon file is the Zig package manifest written in Zig Object Notation format. It declares the package name as "jaide" and sets the version to 40.0.0, reflecting the project's designation as JAIDE v40. The minimum required Zig compiler version is specified as 0.13.0. The dependencies field is empty, indicating that the JAIDE project does not rely on any external Zig packages and is entirely self-contained. The paths field lists build.zig, build.zig.zon, and the src directory as the package contents.

### pyproject.toml — Python Project Configuration

The pyproject.toml file defines the Python project metadata and dependencies. It requires Python 3.11 or later and lists Flask version 3.1.2 or later as the sole Python runtime dependency. This minimal dependency footprint is consistent with the project's architecture: the Python layer serves only as a thin web server gateway that proxies user interactions to the compiled Zig binary. The Modal-based training scripts import the modal package separately since those run in Modal's own cloud environment.

---

## 2. Core Foundation Layer

### src/core/tensor.zig — Multi-Dimensional Tensor Engine

The tensor.zig file implements the central multi-dimensional array abstraction for the JAIDE system, built around two primary structs: Shape and Tensor. The Shape struct holds two dynamically allocated slices, dims and strides, both of type []usize. Its init function takes an allocator and a shape specification, copies the dimensions, and computes row-major strides by iterating from the last dimension backward, multiplying each stride by the corresponding dimension size and using @mulWithOverflow for overflow detection. Shape provides helper methods including totalSize (which computes the product of all dimensions with overflow checks), equals (which delegates to mem.eql on the dims slices), broadcastCompatible (which checks NumPy-style broadcasting rules by comparing trailing dimensions and allowing size-1 dimensions to match any target), isContiguous (which verifies strides match the expected row-major layout), copy (which duplicates both dims and strides via allocator.dupe), and deinit for cleanup.

The Tensor struct itself stores a data slice of []f32, a Shape, an Allocator, and three heap-allocated control fields: refcount (a *usize for atomic reference counting), cow (a *bool for copy-on-write tracking), and mutex (a *std.Thread.Mutex for thread safety). The init function validates that no dimension is zero, computes total size with overflow checks, allocates and zero-fills the f32 data array, and initializes the reference count to 1 with cow set to false. Reference counting is managed through retain (atomic add) and release/deinit (atomic subtract, deallocating when count reaches zero). The copy function creates an independent deep copy. The ensureWritable method implements copy-on-write semantics: when cow is true, it locks the original mutex, allocates new data and control structures, copies the data, atomically decrements the old refcount, and swaps in the new pointers, freeing old resources if this was the last reference. The markShared method sets the cow flag under mutex protection. View operations like newView, view, slice, and transpose all call retain and markShared, then return a new Tensor sharing the same underlying data but with a different Shape, enabling zero-copy reshaping and transposition.

Element-wise arithmetic operations (add, sub, mul, div) and their scalar variants (addScalar, subScalar, mulScalar, divScalar) all follow the same pattern: they verify shape compatibility, call ensureWritable for COW safety, then process data using SIMD vectorization with @Vector(4, f32). The main loop processes four elements at a time by casting data pointers to *const [4]f32, performing vector operations, and writing results back; a scalar tail loop handles remaining elements. The div operation additionally checks for division by zero both in the vector path (using @reduce(.Or, b == zero)) and the scalar tail. Unary math operations including exp, log, sin, cos, tan, sqrt, pow, and abs iterate element-wise, with log returning -inf for non-positive inputs and sqrt returning NaN for negative inputs.

Matrix operations form a large portion of the file. The matmul function verifies that both tensors are 2D and that inner dimensions match, creates a result tensor, and performs standard triple-loop matrix multiplication. The transpose2D function creates a transposed view by swapping dimensions and strides. Linear algebra operations include svd (Singular Value Decomposition using one-sided Jacobi iteration with Givens rotations), qr (QR decomposition via Gram-Schmidt orthogonalization with column norm computation and projection subtraction), cholesky (Cholesky factorization for symmetric positive-definite matrices with a square root and division cascade), lu (LU decomposition with partial pivoting using row swaps and Gaussian elimination), eigendecomposition (iterative QR algorithm that repeatedly decomposes and recomposes the matrix until off-diagonal elements converge below a threshold), inverse (Gauss-Jordan elimination on an augmented identity matrix), determinant (via LU decomposition, multiplying diagonal elements with sign tracking from pivots), and solve (via LU factorization with forward and backward substitution). Reduction operations include sum, mean, max, min, argmax, and argmin, all iterating through data elements. Gradient-related operations include softmax (with numerical stability via max subtraction), layerNorm (computing mean, variance, and normalizing with epsilon), and dropout (zeroing elements with probability p using PRNG-generated random numbers and scaling survivors by 1/(1-p)). Serialization is provided through save (writing shape metadata then raw f32 data) and load (reading metadata and data back).

### src/core/memory.zig — Six Allocators and Security Utilities

The memory.zig file provides comprehensive memory management with six specialized allocators, all thread-safe (mutex-protected) with security zeroing on deallocation. The first is Arena, a bump-pointer allocator that maintains a list of 1MB buffers and advances a current-offset pointer for each allocation, falling back to a new buffer when the current one is full. The second is ArenaAllocator, which wraps a child allocator and provides a vtable-based Allocator interface using the Arena internally. The third is SlabAllocator, designed for fixed-size objects, which maintains free lists per slab and allocates from pre-divided pages. The fourth is PoolAllocator, which manages a pool of identically-sized blocks with a free-list for O(1) allocation and deallocation. The fifth is BuddyAllocator, implementing the classic buddy system that splits power-of-two blocks recursively and merges them back on deallocation, tracking free lists at each level. The sixth is the SurpriseMemoryManager, an entropy-based eviction allocator that assigns priority scores based on information-theoretic surprise metrics and evicts the lowest-priority entries when memory pressure demands it.

The file also includes concurrent data structures: MutexQueue (a thread-safe FIFO using mutex and an ArrayList), MutexStack (a thread-safe LIFO), and ReadWriteLock (a reader-writer lock built from a mutex, a condition variable concept via atomic operations, and reader/writer counts). Security utilities include ChaCha20-Poly1305 encryption with a ChaCha20 struct implementing the quarter-round and block functions, AEAD construction with Poly1305 MAC, and key/nonce management. Deflate compression with a Deflate struct using LZ77-style sliding window matching and Huffman coding. Secure zeroing is performed by secureZero, which uses @memset with volatile semantics to prevent compiler optimization from eliding the zeroing of sensitive data. Virtual memory operations provide mmap and munmap wrappers for memory-mapped file I/O. Global memory statistics tracking via MemoryStats counts total allocations, deallocations, bytes allocated, and peak usage, all protected by an atomic counter.

### src/core/types.zig — Fixed-Point Arithmetic, PRNG, and Domain Types

The types.zig file is a sprawling type definitions module spanning thousands of lines. It begins with fixed-point arithmetic types: FixedPoint16 (8.8 format with a u16 raw field, supporting add, sub, mul with rounding via (a*b+128)/256, div, fromFloat, toFloat), FixedPoint32 (16.16 format using u32, with multiplication via i64 intermediate to avoid overflow), and FixedPoint64 (32.32 format using u64 with i128 intermediates). Each type includes conversion to and from floating point using the appropriate power-of-two scale factors (256, 65536, and 4294967296 respectively).

The generic Tensor type defined here (distinct from the f32-specific one in tensor.zig) uses a []u8 data slice for byte-level storage, coupled with a Shape structure containing dims, strides, and an element_size field. This enables storage of arbitrary element types. The PRNG struct implements the xoshiro256** algorithm, storing four u64 state words and providing next (full state advance with rotation and multiplication), random (mapping u64 to f64 in [0,1)), and randomRange (uniform in [min,max)) methods. The BitSet struct uses a []u64 backing array with per-bit set, unset, isSet, and count operations using @popCount. The ContextWindow holds a fixed-size token buffer and head pointer for sliding-window sequence management. The RankedSegment stores a document_id, score, and position for search result ranking.

The file defines hundreds of additional domain-specific types organized in namespaces. Graphics types include Vec2, Vec3, Vec4, Mat4 with transformation methods. Neural network types include Layer (weights, biases, activation function enum), Network (array of layers, forward pass), ConvLayer (filters, stride, padding), and AttentionHead (query, key, value projections). Quantum computing types include QubitState (two complex amplitudes), QuantumGate (2x2 complex matrix), and QuantumCircuit (array of gate-qubit pairs). The SSIHashTree and SSIHashNode types define the hash-tree data structure used by the semantic index, with bucket-based collision handling. Math utility functions include abs, clamp, lerp, smoothstep, factorial, fibonacci, gcd, lcm, isPrime, and dozens more.

### src/core/io.zig — I/O Primitives and Filesystem Utilities

The io.zig file provides the low-level I/O infrastructure. The MMAP struct wraps POSIX memory-mapped file operations, storing a data pointer and length obtained from mmap with MAP_SHARED or MAP_PRIVATE flags, and providing deinit via munmap. The DurableWriter implements crash-safe file writing by first writing to a temporary file (with a .tmp suffix), calling fsync to ensure data reaches disk, and then atomically renaming the temporary file to the target path via std.os.rename. This guarantees that readers always see either the old complete file or the new complete file, never a partially-written state.

BufferedReader wraps a file descriptor with an 8192-byte internal buffer, providing readLine (which scans the buffer for newlines, refilling as needed, and returns slices of the read data) and readAll (which accumulates the entire file contents into an ArrayList). BufferedWriter similarly buffers output in an 8192-byte array and provides write (which copies data into the buffer, flushing to the underlying file when full) and flush (which writes any remaining buffered data). Hashing utilities include a Wyhash implementation with runtime-initialized seeds, providing hash functions for byte slices. Filesystem utilities provide copyFile (with a progress callback invoked every 64KB), atomicMove (via rename with fallback to copy-and-delete), listDirectory (returning sorted arrays of file names), createBackup (appending a timestamp suffix), and ensureDirectory (creating directories recursively).

### src/core/model_io.zig — Binary Model Serialization with SHA-256 Integrity

The model_io.zig file implements the complete binary serialization and deserialization format for JAIDE models. The ModelMetadata struct stores name, description, version, timestamp, and component counts (num_rsf_layers, rsf_dim, num_ngrams, num_hash_functions, vocab_size). The MAGIC_HEADER is an 8-byte constant "JAIDE40\x00" that identifies valid model files. The ModelFormat struct aggregates a ModelMetadata with optional pointers to RSF, Ranker, and MGT components.

The exportModel function writes the complete binary format: first the 8-byte MAGIC_HEADER, then the version as u32 little-endian, then JSON-serialized metadata preceded by its u32 length. Each component (RSF, Ranker, MGT) is written with a presence byte (0 or 1). For RSF, the serialization writes num_layers and dim as u64 LE, then iterates over layers calling save on each layer's s_weight, t_weight, s_bias, and t_bias tensors. For Ranker, it writes a version byte, ngram_weights as bitcast u32 LE floats, hash function parameters as u64 pairs, and the seed. For MGT, it writes vocabulary size as u32, then iterates over sorted token keys writing each as a length-prefixed string. All data is simultaneously fed into a SHA-256 hasher, and the 32-byte checksum is appended to the file.

The importModel function reverses this process: it reads and validates the magic header, checks the version, reads and parses the metadata JSON, then processes each component. After all components, it computes the expected SHA-256 checksum, reads the stored checksum, and compares them using constantTimeCompare (a timing-attack-resistant byte comparison that XORs all bytes and checks for zero difference). If checksums don't match or extra bytes remain, the file is considered corrupted. The function uses comprehensive errdefer chains to ensure proper cleanup on any parsing failure.

---

## 3. Core Relational Subsystem

### src/core_relational/nsir_core.zig — Self-Similar Relational Graph

The nsir_core.zig file constitutes the foundational data model for the entire JAIDE system, defining the quantum-inspired relational graph that underpins all higher-level modules. It begins by importing the standard library's Allocator, ArrayList, StringHashMap, Sha256, and Complex types. The file defines EdgeQuality as a five-variant enum with integer backing (u8), encompassing superposition, entangled, coherent, collapsed, and fractal states, each with toString and fromString conversion methods. Helper functions dupeBytes, freeMapStringBytes, and putOwnedStringBytes provide owned-memory management for StringHashMap([]u8) instances.

The Qubit struct stores two Complex(f64) amplitudes a and b, representing the standard two-level quantum state, with factory methods initBasis0 and initBasis1 producing the |0⟩ and |1⟩ computational basis states respectively. The normalization method normalizeInPlace computes the norm-squared via magnitudeSquared on both components and rescales by the inverse square root, gracefully falling back to |0⟩ if the norm is zero, NaN, or infinite. Probability extraction is handled by prob0 and prob1, both clamped to [0,1].

The Node struct contains fields id and data as owned byte slices, a Qubit instance, a floating-point phase value, a StringHashMap for arbitrary metadata, and a reference to the Allocator. The Edge struct carries source and target as borrowed string slices, an EdgeQuality enum value, a floating-point weight, a Complex(f64) quantum_correlation, a fractal_dimension float, a metadata map, and an allocator. The main graph type, SelfSimilarRelationalGraph, uses a StringHashMap of Nodes keyed by node ID and a HashMap of EdgeKey to ArrayList(Edge) for multi-edges. It also maintains an entanglements map from EntanglementKey to EntanglementState (a four-element Complex(f64) amplitude array), a fractal_depth counter, a 65-byte topology_hash, and the allocator.

The topology hash computation in updateTopologyHash iterates all nodes and edges, computing per-element SHA-256 digests that incorporate the node's id, data, phase, qubit amplitudes, and XOR-accumulated metadata hashes, then XOR-folds these into accumulators. A final SHA-256 pass over all three accumulators produces a 32-byte digest rendered as a 64-character hex string stored in topology_hash. The encodeInformation method hashes input data with SHA-256, derives a 16-character hex node ID, creates a new node with basis-0 qubit and zero phase, links it to up to three existing nodes with coherent edges of weight 0.5, and updates the topology hash. The file concludes with unit tests verifying quantum gate basics (Hadamard producing equal superposition, Pauli-X flipping basis states, Pauli-Z negating the |1⟩ amplitude).

### src/core_relational/chaos_core.zig — Distributed Computing Kernel

The chaos_core.zig file implements a distributed computing kernel centered around the ChaosCoreKernel struct. The file opens with three critical supporting types. ContentAddressableStorage uses Blake2b-256 hashing for data deduplication: the store method computes the hash of incoming data and checks whether it already exists in the internal HashMap([]const u8), returning the existing entry if so, and only storing and duplicating the data if it is new. The retrieve method looks up data by its hash key. The DynamicTaskScheduler manages tasks via a priority queue (implemented as a min-heap based on priority score), with schedule, dequeue, and rebalance operations. The DataFlowAnalyzer tracks data dependencies between processing nodes using a HashMap of node_id to dependency lists, providing addDependency, getDependencies, and topologicalSort (which performs Kahn's algorithm using in-degree counting and a queue).

The ChaosCoreKernel itself ties these together. It holds the ContentAddressableStorage, DynamicTaskScheduler, DataFlowAnalyzer, a blocks HashMap mapping block IDs to data, a node_loads HashMap tracking per-node computational load, and an allocator. The processBlock method stores a data block, schedules a task referencing the block, and optionally records dependencies. The optimizeDistribution method attempts to balance load across nodes by identifying the most and least loaded nodes and migrating blocks between them when the load ratio exceeds a threshold. The getClusterMetrics method computes summary statistics: total blocks, total tasks, average load, and load variance.

### src/core_relational/crev_pipeline.zig — Knowledge Extraction Pipeline

The crev_pipeline.zig file implements the CREV (Contextual Relational Extraction and Verification) pipeline for extracting structured knowledge from raw text. The ExtractionStage enum defines the pipeline stages: tokenization, triplet_extraction, validation, integration, and indexing, with next and toString methods for stage progression. The RelationalTriplet struct holds subject, relation, and object strings, a confidence score (f64 clamped to [0,1]), a source_hash (32-byte SHA-256 digest of the source text), an extraction_timestamp, and a metadata StringHashMap. The TripletIndex provides fast lookup by subject, by object, and by relation using three separate HashMaps, each mapping string keys to ArrayLists of triplet pointers.

The StreamBuffer implements a ring buffer for streaming text processing, with write (advancing a head pointer modulo capacity) and read (advancing a tail pointer) operations. Pattern-based extraction functions process different input formats: extractFromText splits text on whitespace and identifies subject-verb-object patterns, extractFromCSV parses comma-separated records with header-driven field mapping, and extractFromImage is a stub for future image processing. The CREVPipeline struct orchestrates the full pipeline, maintaining a current stage, a triplet buffer, the index, and processing statistics. Its processInput method advances through stages sequentially, applying the appropriate extraction function at each stage.

### src/core_relational/dataset_obfuscation.zig — Privacy and Data Protection

The dataset_obfuscation.zig file implements a comprehensive data privacy layer. The PaillierCrypto struct implements the Paillier homomorphic encryption scheme with 256-bit keys, providing encrypt, decrypt, and homomorphicAdd operations that allow addition of encrypted values without decryption. The key generation uses a simplified two-prime product for n with n-squared as the public modulus. The LSHFingerprinter computes locality-sensitive hash fingerprints using random hyperplane projection: it generates random projection vectors, computes dot products with input data, and produces binary fingerprints based on the sign of each projection.

The DifferentialPrivacyBudget tracks epsilon and delta privacy parameters, providing addNoise (Laplace mechanism: adding noise drawn from a Laplace distribution scaled by sensitivity/epsilon), checkBudget (verifying remaining budget), and consumeBudget. The KAnonymizer groups records by quasi-identifier fields and ensures each group has at least k members, generalizing or suppressing groups that fall below the threshold. The MerkleProofGenerator constructs a Merkle tree from data blocks, computing leaf hashes via SHA-256 and internal node hashes by concatenating and re-hashing child pairs. The generateProof method returns the sibling hashes along the path from a leaf to the root, and verifyProof recomputes the root from a leaf hash and the proof path to check consistency.

### src/core_relational/esso_optimizer.zig — Entangled Stochastic Symmetry Optimizer

The esso_optimizer.zig file implements ESSO, the custom optimizer that combines simulated annealing with symmetry-group analysis. The SymmetryGroup struct defines a symmetry group with an order, a generator matrix (as a flat f64 array), and a group_type enum (cyclic, dihedral, permutation, continuous). The SymmetryTransform stores a transformation matrix alongside its source group reference. The OptimizationState holds the current solution as a node-edge representation of the graph, the current energy (computed as negative total edge weight), the best solution found so far and its energy, the temperature (initialized to 1.0), and a cooling rate.

The EntangledStochasticSymmetryOptimizer stores the state, a PRNG, a list of discovered SymmetryPatterns, iteration and stagnation counters, and configuration parameters (initial_temperature, min_temperature, cooling_rate, reheat_factor, max_stagnation). The optimize method runs for a specified number of iterations, each of which selects one of five move types with equal probability: perturbing an edge weight by Gaussian noise, swapping two edges, applying a symmetry transformation, adjusting a node's quantum phase, and modifying edge quality. After applying the move, the method computes the new energy and applies the Metropolis acceptance criterion (always accepting improvements, accepting worsening moves with probability exp(-delta/temperature)). The temperature is cooled multiplicatively each iteration, and if stagnation exceeds the threshold (50 iterations without improvement), the temperature is reheated by the reheat_factor. The adaptive cooling mechanism adjusts the cooling rate based on the acceptance rate: faster cooling when acceptance is too high, slower when too low.

### src/core_relational/fnds.zig — Fractal Tree Indexing

The fnds.zig file implements a fractal tree data structure for hierarchical indexing with self-similar properties. The FractalLevel struct represents a single level in the tree hierarchy, containing a level number, a list of child FractalLevel pointers, a data payload as a byte slice, a computed fractal_dimension (f64), and an allocator. The init method creates a new level with empty children and zero dimension, while addChild appends a child level and incrementally updates the parent's fractal dimension.

The fractal dimension is estimated using a box-counting algorithm: estimateBoxCountingDimension counts the number of non-empty boxes at different scales and performs a least-squares linear regression on log(count) vs log(1/scale) to estimate the dimension. The FractalTree struct manages the root FractalLevel and provides insert (which traverses down the tree based on a hash of the key, creating new levels as needed), search (traversing down and checking data at each level), and four traversal orders: depthFirst, breadthFirst (using an ArrayList as a queue), levelOrder (same as breadthFirst but returning only data payloads), and fractalOrder (which sorts children by fractal dimension before recursive traversal, visiting more complex subtrees first). The SelfSimilarIndex wraps a FractalTree with an LRU cache for recently accessed entries, providing cached insert and search operations.

### src/core_relational/formal_verification.zig — Theorem Prover

The formal_verification.zig file implements a built-in theorem prover and proof checker. It defines 26 proof rules (ModusPonens, UniversalInstantiation, ExistentialGeneralization, Induction, Contradiction, CutElimination, Resolution, ParaModulation, Demodulation, Subsumption, Superposition, ReflexivityResolution, OrderedResolution, HyperResolution, SetOfSupport, URResolution, NegativeResolution, PositiveResolution, BinaryResolution, FactorResolution, UnitResolution, InputResolution, LinearResolution, SLResolution, and ModelElimination) and 18 proposition types (Atomic, Conjunction, Disjunction, Implication, Negation, Universal, Existential, Equality, BiConditional, ExclusiveOr, Necessity, Possibility, TemporalAlways, TemporalEventually, TemporalUntil, TemporalSince, ContextualTruth, and RelationalPredicate).

The Term structure uses a union type with reference counting: variables (carrying a name and a binding level), constants (name only), and applications (a function term applied to an argument term, both as reference-counted pointers). The ProofStep struct records a proposition, the rule applied, and a list of premise ProofStep references. The TheoremProver implements Robinson unification (which recursively unifies terms, handling variable binding with occurs-check to prevent infinite types), resolution (which finds complementary literal pairs across two clauses and combines the remaining literals), and backward chaining (which attempts to prove a goal by matching it against known axioms and recursively proving their antecedents). The HoareLogicVerifier provides verification of Hoare triples (precondition, program, postcondition) using weakest precondition computation through assignment, sequential composition, conditional, and loop constructs.

### src/core_relational/ibm_quantum.zig — IBM Quantum API Client

The ibm_quantum.zig file implements an HTTP client wrapper for the IBM Quantum computing API. The IBMQuantumClient struct contains an allocator, an API token (stored as a heap-duplicated byte slice), and an std.http.Client instance. The init method duplicates the token for ownership safety, and deinit frees both the token and the HTTP client. The submitJob method constructs an HTTP POST request to the IBM Quantum API endpoint, sending a JSON payload containing QASM circuit code and backend specification, with the API token provided as a Bearer authorization header. The getJobResult method polls a job status endpoint using the job ID, returning the response body as a string. Both methods use chunked transfer encoding and read responses up to 1MB.

### src/core_relational/mod.zig — Module Re-Export Hub

The mod.zig file serves as the central public interface for the entire core_relational subsystem. It consists entirely of pub const declarations that re-export the primary types from each of the 16 sub-modules. These include SelfSimilarRelationalGraph and Node from nsir_core, ChaosCoreKernel from chaos_core, CREVPipeline from crev_pipeline, DatasetObfuscator from dataset_obfuscation, EntangledStochasticSymmetryOptimizer from esso_optimizer, FractalTree from fnds, TheoremProver from formal_verification, IBMQuantumClient from ibm_quantum, QuantumHardware from quantum_hardware, RelationalQuantumLogic from quantum_logic, QuantumTaskAdapter from quantum_task_adapter, ReasoningOrchestrator from reasoning_orchestrator, RGPU from r_gpu, SafetyModule from safety, SecurityProver from security_proofs, SignalPropagationEngine from signal_propagation, SurpriseMemoryManager from surprise_memory, TemporalGraph from temporal_graph, TypeChecker from type_theory, VerifiedInferenceEngine from verified_inference_engine, VPU from vpu, ZKVerifier from zk_verification, and ZRuntime from z_runtime. The file includes integration tests that instantiate each type to verify they compile and link correctly.

### src/core_relational/quantum_hardware.zig — IBM Backend Specifications

The quantum_hardware.zig file provides detailed specifications for five IBM quantum processor families. Each backend is defined with its qubit count, connectivity topology, gate set, coherence times (T1 and T2), and gate error rates. The supported backends include ibm_brisbane (127 qubits, heavy-hex topology), ibm_osaka (127 qubits, heavy-hex), ibm_kyoto (127 qubits), ibm_sherbrooke (127 qubits), and ibm_nazca (127 qubits). Calibration data parsing reads IBM's JSON calibration format and populates per-qubit error rates, readout error rates, and gate duration tables.

OPENQASM 3.0 circuit generation is provided through the generateQASM method, which translates the internal gate representation into standard QASM syntax with qubit register declarations, gate operations, and measurement instructions. Error mitigation techniques include zero-noise extrapolation (running circuits at multiple noise levels and extrapolating to zero noise) and probabilistic error cancellation. Preset circuit generators produce standard variational circuits for VQE (Variational Quantum Eigensolver) and QAOA (Quantum Approximate Optimization Algorithm) with parameterized rotation gates and entangling layers.

### src/core_relational/quantum_logic.zig — Quantum State Simulator

The quantum_logic.zig file implements a quantum logic simulator with 12 gate types. The LogicGate enum defines IDENTITY, HADAMARD, PAULI_X, PAULI_Y, PAULI_Z, CNOT, SWAP, TOFFOLI, PHASE_S, T_GATE, RELATIONAL_AND, and RELATIONAL_OR. The QuantumState struct stores an amplitude (f64), phase (f64), entanglement level (usize), and a measurement_collapse boolean. The RelationalQuantumLogic struct maintains an ArrayList of QuantumState instances, providing addState, getState, and setState for state management.

The applyGate method dispatches on the gate type, modifying the target state's amplitude and phase. For HADAMARD, it scales the amplitude by 1/sqrt(2) and adds pi/4 to the phase. For PAULI_X, it replaces amplitude a with sqrt(1-a²). For CNOT, it examines the control qubit's amplitude and flips the target if the control amplitude exceeds 0.5. For SWAP, it exchanges two states' amplitudes and phases. The measure method collapses a state by comparing its amplitude-squared probability against a PRNG-generated random number, setting amplitude to 1.0 or 0.0 accordingly and marking measurement_collapse as true. The applyFractalTransform scales all amplitudes by 1/sqrt(N) where N is the number of states, then renormalizes, producing a fractal-like distributed state.

### src/core_relational/quantum_task_adapter.zig — Quantum-Classical Bridge

The quantum_task_adapter.zig file bridges the graph data model with quantum execution. The QuantumTaskAdapter struct holds a reference to the SelfSimilarRelationalGraph and a RelationalQuantumLogic engine. The identifyQuantumSubgraph method scans the graph for nodes whose edges have EdgeQuality values of superposition or entangled, collecting them as candidates for quantum processing. The generateQASM method translates the identified subgraph into an OPENQASM 3.0 circuit string, with one qubit per node and gates derived from edge types.

The simulateLocally method runs a simplified quantum simulation without external hardware, applying gates from the generated circuit description to the RelationalQuantumLogic engine and returning measurement results. The applyResults method takes quantum measurement outcomes and updates the corresponding nodes in the graph: collapsing qubits to basis states based on measurement values, updating edge qualities from superposition to collapsed, and adjusting edge weights based on measurement probabilities. This adapter enables the system to offload specific graph subproblems to quantum processing while maintaining the classical graph structure for the rest of the computation.

### src/core_relational/reasoning_orchestrator.zig — Multi-Phase Reasoning

The reasoning_orchestrator.zig file implements a hierarchical multi-phase reasoning system. The ThoughtLevel enum has three variants: local, global, and meta. The ReasoningPhase struct captures a single reasoning phase's state, containing phase_id (derived from nanoTimestamp), level, inner and outer iteration counts, target_energy and current_energy, convergence_threshold (defaulting to 1e-6), start and end timestamps, and pattern_captures as an ArrayList of 32-byte arrays. The hasConverged method checks whether the absolute difference between current and target energy falls below the threshold.

The OrchestratorStatistics struct tracks total phases executed, broken down by level, total inner and outer loops, running average convergence time, best energy achieved (initialized to positive infinity), and patterns discovered. The ReasoningOrchestrator holds pointers to the graph, the ESSO optimizer, and the ChaosCoreKernel, along with the active phase, phase history, statistics, and tuning parameters (fast_inner_steps defaulting to 50, slow_outer_steps to 10, hierarchical_depth to 3). The orchestrate method runs a multi-phase reasoning loop: at each hierarchical level, it creates a new ReasoningPhase, runs inner optimization loops using ESSO, evaluates convergence, and progresses to the next level. Local phases optimize individual node neighborhoods, global phases optimize the entire graph structure, and meta phases analyze optimization patterns across previous phases to guide future exploration.

### src/core_relational/r_gpu.zig — Relational Graph Processing Unit

The r_gpu.zig file implements the RGPU, a software abstraction of a graph-specialized processing unit with a mesh network-on-chip (NoC) architecture. The ProcessingCore struct models an individual core with states (idle, loading, computing, storing, error), local memory as a byte array, a program counter, and cycle count. The NoCMessage struct carries a source and destination core ID, a payload byte slice, a priority level, and a message type enum (data, control, sync, interrupt). XY routing determines message paths through the mesh, advancing first in the X direction, then Y.

The GraphIsomorphismProcessor provides graph isomorphism detection by computing canonical forms: it sorts the adjacency representation of each graph by degree sequence and neighbor-ID lists, producing a canonical hash that can be compared across graphs. If two graphs have the same canonical hash, they are isomorphic. The DynamicEdgeWeighting module adjusts edge weights based on usage patterns, strengthening frequently traversed edges and weakening unused ones over time via exponential decay. The SparseActivationManager tracks which processing cores are active and which can be powered down, maintaining an activation bitmap. The PowerGatingController monitors per-core utilization and gates power to idle cores, waking them on demand when new graph processing tasks arrive.

### src/core_relational/safety.zig — Safe Type Casting and Cryptographic Utilities

The safety.zig file provides low-level safety primitives. The safeIntCast function performs checked integer type conversions, returning an error if the source value doesn't fit in the target type's range (checking against both maxInt and minInt). The safePtrCast performs pointer type conversions with alignment checking, verifying that the source pointer's address is properly aligned for the target type before performing @ptrCast. Pointer validation functions check for null, alignment, and whether a pointer falls within a reasonable memory range.

The SecureRng struct wraps the system's cryptographic random number generator, providing getRandomBytes (filling a buffer from std.crypto.random), getRandomU64, and getRandomFloat (mapping u64 to f64 in [0,1)). The MonotonicClock provides monotonic timestamps via std.time.nanoTimestamp for elapsed-time measurements that aren't affected by wall-clock adjustments. The secureCompare function performs constant-time byte comparison by XORing all byte pairs and accumulating the result, preventing timing side-channel attacks. The secureZeroBytes function uses volatile stores to zero memory, preventing the compiler from optimizing away the zeroing of sensitive data like cryptographic keys. The BigInt512 struct provides extended precision arithmetic with a [8]u64 limb array, supporting add, sub, mul (schoolbook multiplication with carry propagation), and comparison operations for cryptographic computations that require more than 64 bits.

### src/core_relational/security_proofs.zig — Security Framework

The security_proofs.zig file implements a comprehensive formal security framework. SecurityLevel defines a lattice of classification levels (unclassified, confidential, secret, top_secret) with a dominates method implementing the lattice partial order. IntegrityLevel similarly defines low, medium, high, and critical levels with their own dominance relation. AccessRightSet uses a u32 bitmask to represent combinations of read, write, execute, and admin permissions, with union_, intersection, and contains operations.

The SecurityLabel pairs a SecurityLevel with an IntegrityLevel, and label dominance requires both components to dominate. The Bell-LaPadula model implementation enforces the simple security property (no read-up: a subject can only read objects at or below its security level) and the star property (no write-down: a subject can only write to objects at or above its security level). The Biba integrity model enforces the dual: no read-down for integrity and no write-up for integrity. The InformationFlowAnalyzer tracks data flows between labeled entities and checks for illegal flows that violate the security policy. The NonInterferenceProver uses bisimulation to verify that high-security inputs cannot influence low-security outputs, establishing non-interference properties. The AccessControlMatrix manages subject-object permission mappings, and SeparationOfDuties enforces multi-person authorization for critical operations. The HashChainVerifier maintains a chain of SHA-256 hashes for audit log integrity, where each entry's hash incorporates the previous entry's hash, creating a tamper-evident chain.

### src/core_relational/signal_propagation.zig — Wave-Based Signal Engine

The signal_propagation.zig file implements wave-based signal propagation across the relational graph. The SignalState struct holds amplitude (f64), phase (f64), and frequency (f64) fields, representing a wave-like signal at a graph node. The ActivationTrace records the history of signal states at a node over time as an ArrayList of SignalState entries, enabling temporal analysis of signal evolution.

The SignalPropagationEngine holds a reference to the SelfSimilarRelationalGraph and a HashMap mapping node IDs to ActivationTrace instances. The propagate method takes a source node ID and an initial SignalState, then iteratively spreads the signal to neighboring nodes along graph edges. At each hop, the signal amplitude is scaled by the edge weight, the phase is shifted by a function of the edge's fractal dimension, and the frequency is preserved. The received signal at each neighbor is blended with its existing state using a 70/30 ratio (70% old state, 30% incoming signal), preventing abrupt state changes and enabling smooth signal diffusion. The trace at each affected node is updated with the new state. The getTrace method returns the full activation history for a node, and the getActivationLevel returns the current amplitude at a node.

### src/core_relational/surprise_memory.zig — Novelty-Driven Memory Manager

The surprise_memory.zig file implements an entropy-based memory management system that prioritizes novel information. The SurpriseMetrics struct computes three novelty measures for incoming data: Jaccard dissimilarity (1 minus the Jaccard similarity between the incoming data's byte-set and existing entries), hash distance (normalized Hamming distance between Wyhash values), and temporal novelty (a decay function based on the time elapsed since the last similar entry was stored). These three metrics are combined with configurable weights to produce a composite surprise score.

The SurpriseRecord struct stores a data entry alongside its computed surprise score, a timestamp, and an access count. The SurpriseMemoryManager maintains an ArrayList of SurpriseRecords, protected by a std.Thread.Mutex for thread safety. The store method computes the surprise score for new data, creates a SurpriseRecord, and inserts it into the list. When the list exceeds a configured capacity, the manager evicts the record with the lowest surprise score (i.e., the least novel entry), ensuring that memory is always occupied by the most surprising and presumably most informative data. The retrieve method searches for entries matching a query, and the reorganize method re-evaluates all entries' surprise scores based on the current memory contents, updating priorities as the context evolves.

### src/core_relational/temporal_graph.zig — Versioned Graph with Time Travel

The temporal_graph.zig file implements a version-controlled graph structure that supports point-in-time queries and rollback. The NodeVersion struct stores a snapshot of a node's state (data, qubit, phase, metadata) along with a version number and timestamp. The EdgeVersion similarly captures an edge's state (weight, quality, quantum_correlation, fractal_dimension) with version metadata. The TemporalNode struct wraps a current Node with an ArrayList of NodeVersion history entries. The TemporalEdge does the same for edges.

The TemporalGraph struct manages HashMaps of TemporalNodes and TemporalEdges, a current version counter, and a list of GraphSnapshot instances. The addNode and addEdge methods create new temporal entries with initial versions. The updateNode method saves the current state as a new version in the history, then applies the modification. The getNodeAtVersion method searches the version history for the entry matching a specific version number, returning the node state as it existed at that point in time. The snapshot method captures the complete graph state at the current version, and the rollback method restores the graph to a previously captured snapshot, replacing all current nodes and edges with their historical counterparts. This enables temporal analysis of how the knowledge graph has evolved and provides an undo capability for experimental modifications.

### src/core_relational/type_theory.zig — Dependent Type System

The type_theory.zig file implements a dependent type system with 21 TypeKind constructors including Unit, Bool, Nat, Int, Float, String, Pi (function types), Sigma (dependent pair types), Identity (propositional equality), Universe (type-of-types at a given level), Inductive, Record, Sum, List, Vector, Matrix, Tensor_type, Quantum, Effect, Linear, and Refinement. The Type struct is recursive, containing a kind enum and optional sub-type pointers for Pi (parameter type and return type), Sigma (first and second component types), and Identity (type and two endpoints).

Capture-avoiding substitution is implemented through the substitute method, which traverses the type structure and replaces variables matching a given name with a replacement type, recursively descending into Pi and Sigma types while avoiding capture by checking whether the bound variable name matches. The TypeContext struct maintains a scope chain of variable-type bindings using an ArrayList of name-type pairs, supporting push, pop, and lookup operations. The Term struct represents terms with constructors for variables, abstractions (lambda), applications, pairs, projections, reflexivity proofs, and type annotations. The TypeChecker provides type inference and checking: inferType dispatches on the term kind, looking up variables in the context, inferring Pi types for abstractions, checking applications against Pi types with substitution, and verifying that Identity terms have matching endpoints. The checkType method compares inferred types against expected types for bidirectional type checking.

### src/core_relational/verified_inference_engine.zig — Verified Inference Pipeline

The verified_inference_engine.zig file implements an inference pipeline with built-in cryptographic verification. The VerifiedInferenceEngine stores weight matrices for multiple layers, bias vectors, a SHA-256 hasher for commitment computation, and differential privacy parameters (epsilon and delta). The infer method processes input through sequential layers of matrix multiplication followed by bias addition and ReLU activation, producing output logits. Before returning, it computes a cryptographic commitment to the inference trace (a hash of the input, all intermediate activations, and the output) that can be used to prove the inference was performed correctly.

The differentialPrivacyNoise method adds calibrated Laplace noise to output logits, with the noise scale determined by the sensitivity divided by epsilon. The generateProof method creates a zero-knowledge proof structure containing the commitment, a Merkle root of the inference trace, and auxiliary data for verification. The BatchVerifier processes multiple inference requests, collecting individual proofs and aggregating them into a single batch proof. The ProofAggregator takes multiple individual proofs and combines them by constructing a Merkle tree over all proof commitments, producing a single root hash that verifies the entire batch.

### src/core_relational/vpu.zig — SIMD Vector Processing Unit

The vpu.zig file implements a generic SIMD vector processing unit. The SimdVector function is a comptime generic that takes an element type and lane count, producing specialized vector types: F32x4 (4-lane f32), F32x8 (8-lane f32), F64x2 (2-lane f64), and F64x4 (4-lane f64). Each generated type wraps Zig's native @Vector type and provides comprehensive operations: add, sub, mul, div for element-wise arithmetic; dot for dot product (multiply then @reduce(.Add)); normalize (dividing by the L2 norm); cross3 for 3D cross product; reflect for vector reflection across a normal; blend for interpolation between two vectors using a scalar parameter; and shuffle for lane permutation.

The VectorBatch struct manages batches of vectors for bulk processing, storing them in an ArrayList and providing processBatch (applying a function to all vectors) and reduceBatch (combining all vectors with a reduction function). The MemoryPool manages a pre-allocated pool of vector-sized memory blocks for allocation-free vector processing in hot loops. The loadFromGraph function reads node and edge data from a SelfSimilarRelationalGraph and packs it into SIMD vectors for accelerated graph processing, extracting floating-point properties (phases, edge weights, quantum amplitudes) into contiguous vector lanes.

### src/core_relational/zk_verification.zig — Zero-Knowledge Proof Pipeline

The zk_verification.zig file bridges the Zig runtime with external zero-knowledge proof tools (Circom and SnarkJS). The Groth16Proof struct stores the proof's three curve points (pi_a, pi_b, pi_c) as byte arrays, along with public inputs as an f64 array. The CircomProver manages the external proof generation process: it spawns a Node.js subprocess running snarkjs to compile the Circom circuit, generate the witness, and produce the Groth16 proof, reading the proof data from the subprocess output.

The InferenceWitness struct converts floating-point inference data to fixed-point representation suitable for the arithmetic circuit, scaling by 2^16 and rounding to integers. The CommitmentScheme provides Pedersen-style commitments using SHA-256, allowing the prover to commit to values that can be revealed later. The DifferentialPrivacy module implements the Laplace mechanism specifically for ZK-compatible noise addition, where the noise must be reproducible from a seed for proof verification. The MerkleTree struct builds a binary hash tree from leaf data, providing root computation and membership proofs (arrays of sibling hashes along the path to the root) with verification that recomputes the root from a leaf and proof path.

### src/core_relational/z_runtime.zig — High-Level Execution Environment

The z_runtime.zig file provides the top-level runtime for the JAIDE system, combining the relational graph, quantum logic, and execution history into a unified programming model. The ZVariable struct is the fundamental entity, combining a name, a heap-allocated SelfSimilarRelationalGraph, a heap-allocated RelationalQuantumLogic engine, and a history ArrayList of HistoryEntry records. The assign method encodes a value string into the graph via encodeInformation, computes a Wyhash-based hash, derives quantum amplitudes from cosine and sine of the hash, and records a history entry. The getValue method retrieves the latest node's data via decodeInformation. The relateTo method creates a quantum-correlated edge between two variables by computing the correlation as the product of one state with the conjugate of another.

The ZRuntime struct is the top-level manager, holding a variables StringHashMap, a global_graph, a global_logic engine, and an execution_history. The createVariable method allocates a ZVariable with an optional initial value. The relationalOperation method takes two variable names and an operation type (AND, OR, XOR, or entangle), creates a result variable, copies quantum states from both operands, applies the corresponding logic gate, and establishes relationships via relateTo. The executeQuantumCircuit method accepts an array of GateSpec structs (gate_name, indices, params) and sequentially applies gates to a target variable. The getSystemState method compiles a complete SystemState snapshot including variable count, total nodes, total edges, and average fractal dimension.

---

## 4. Processing Pipeline

### src/processor/rsf.zig — Reversible Scale Flow Neural Network

The rsf.zig file implements a Reversible Scale Flow neural network layer and multi-layer model. The RSFLayerConfig struct has four fields: clip_min (defaulting to -5.0), clip_max (defaulting to 5.0), seed_offset (u64 defaulting to 0), and grad_mean (bool defaulting to true). These parameters control numerical clamping, random seed offset for weight initialization, and whether gradient averaging is applied during backpropagation.

The RSFLayer struct is the core computational unit. It holds eight Tensor fields: s_weight and t_weight (learnable weight matrices for "scale" and "translation" sub-networks), s_bias and t_bias (corresponding bias vectors), and four matching gradient accumulator tensors. The initWithConfig function validates that dim is non-zero and clip bounds are finite, then computes Xavier initialization bounds as sqrt(6 / (fan_in + fan_out)) where both equal dim. It creates weight tensors of shape [dim, dim] with random uniform values within the Xavier bounds and bias tensors of shape [1, dim] initialized to zeros.

The forward method implements the coupling layer transformation. Given two input tensors x1 and x2 (both of shape [batch_size, dim]), it computes s(x2) = exp(clip(s_weight * x2^T + s_bias)), clamping to finite values, clipping to [clip_min, clip_max], exponentiating, and clamping again. Then x1 is element-wise multiplied by this scale factor. Next, it computes t(x1) = t_weight * x1^T + t_bias, clamps, and adds to x2. The inverse method reverses this: subtracting t(y1) from y2 to recover x2, then dividing y1 by exp(s(x2)) to recover x1, with an explicit check for zero or non-finite divisors. The backward method computes gradients via chain rule through both the scale and translation sub-networks, accumulating into the gradient tensors. The RSF struct manages multiple RSFLayers, providing forward (sequentially applying all layers), inverse (applying layers in reverse order), and backward (backpropagating through all layers and returning input gradients) methods. Serialization via save writes "RSF0" magic bytes followed by layer count, dim, and each layer's weight/bias tensors; load reverses this.

### src/tokenizer/mgt.zig — Morphological Grammar Tokenizer

The mgt.zig file implements the MGT (Morpheme-Guided Tokenization) system. The MGT struct is the central component, containing token_to_id and id_to_token bidirectional hash maps, prefixes, suffixes, and roots hash maps for morphological elements, bpe_pairs for Byte Pair Encoding merge rules with priority values, anchors for special reference-point tokens, allocated_strings for memory tracking, and an allocator. Four special tokens are hardcoded: PAD (0), UNK (1), BOS (2), and EOS (3).

The system provides comprehensive bilingual morphological support. English prefixes include un, re, pre, dis, mis, over, under, sub, inter, trans, super, semi, anti, and non. Hungarian prefixes include meg, el, fel, le, be, ki, rá, át, szét, vissza, ide, oda, alá, fölé, közé, össze, túl, hozzá, körül, alig, éppen, majd, csak, is, and leg. Hungarian suffixes include ság, ség, ban, ben, ba, be, ból, ből, hoz, hez, höz, tól, től, nak, nek, val, vel, ért, ul, ül, ként, án, én, ig, at, et, tat, tet, ott, ett, atlan, etlen, talan, telen, ál, él, oz, ez, öd, ed, kor, ra, and re. The morphDecompose function finds the longest matching prefix and suffix, extracting the remaining middle as the root, with a minimum root length of 2 characters to prevent over-fragmentation.

The trainBPE method implements Byte Pair Encoding training: counting adjacent byte-pair frequencies in the corpus, sorting by frequency, registering the top pairs as new tokens with priority indices. The encode method performs multi-level tokenization: first checking for special tokens, then whitespace handling, then morphological decomposition, then dictionary lookup, then BPE fallback for unknown subwords, and finally per-byte fallback using hexadecimal <XX> format. The decode method reverses encoding by converting hex tokens to bytes, skipping special tokens, and concatenating regular token strings. Batch operations (encodeBatch, batchDetokenize), coverage metrics, anchor-aware tokenization, and binary serialization (saveVocab/loadVocab) round out the module.

### src/optimizer/sfd.zig — Stochastic Fractal Descent Optimizer

The sfd.zig file is a roughly 2,000-line optimization module that defines its own local Tensor type and implements multiple optimization algorithms. The SFD optimizer is the primary algorithm, functioning as an Adam-like optimizer with Fisher diagonal preconditioning. It maintains first-moment (mean) and second-moment (variance) running averages of gradients, applies bias correction, and scales updates by the inverse square root of the Fisher information diagonal approximation.

The KFACBlock implements Kronecker-Factored Approximate Curvature, storing Kronecker factors A and G (input and gradient covariance matrices) and providing efficient natural gradient updates by inverting the smaller factors instead of the full Fisher matrix. The SpectralNormalizer constrains weight matrix spectral norms using power iteration. The GradientFlowController monitors gradient magnitudes across layers and applies per-layer scaling to prevent vanishing or exploding gradients. The MARSVarianceReducer implements variance-reduced stochastic gradient estimation using control variates. The ReversibleOptimizerState stores optimizer state in a way that allows exact reversal of optimization steps for the RSF architecture.

The LRScheduler provides six learning rate schedule types: constant, linear warmup with decay, cosine annealing, step decay, exponential decay, and one-cycle (combining warmup and cosine decay). The MixedPrecisionTrainer manages training with mixed f16/f32 precision, maintaining master weights in f32 while computing gradients in f16 for speed. The B200MemoryManager provides GPU memory management specific to NVIDIA B200 hardware, tracking allocations and implementing memory pooling for the 80GB HBM3e memory. The B200KernelOptimizer fuses sequences of small CUDA kernel launches into larger fused kernels to reduce launch overhead. The BayesianOptimizer uses Gaussian Process regression with an RBF kernel to optimize hyperparameters, with an acquisition function (expected improvement) guiding the search. The SophiaSOAPOptimizer implements the Sophia second-order optimizer with stochastic diagonal Hessian estimation.

### src/index/ssi.zig — Sparse Segment Index

The ssi.zig file implements a hash-based tree structure for semantic search. The Node struct contains a key (u64 hash), a value as a token ID array, a score (f64), and a position (usize). The CollisionNode handles hash collisions by chaining multiple key-value pairs at the same hash bucket. The Segment groups nodes into segments with a segment_id and item count. The SSI struct manages an array of hash buckets, each potentially containing chains of CollisionNodes.

Insertion via addSequence computes a Wyhash of the token array, routes to a bucket, and chains on collision. Retrieval via retrieveTopK takes a query token array, hashes it, scans buckets for candidates, computes Hamming distance between query and candidate hashes, maintains a min-heap of the top-K closest results, and returns them sorted by distance. The merge method combines two SSI instances by iterating all entries from the source and inserting them into the target. The split method divides an SSI into two halves based on hash value ranges. Binary serialization (serialize/deserialize) writes bucket counts, node data, and collision chains in a compact format. The exportToTensor and importFromTensor methods convert the index data to and from Tensor format for integration with the neural pipeline.

### src/ranker/ranker.zig — Multi-Criteria Ranking System

The ranker.zig file implements a ranking system that combines multiple scoring signals. The Ranker struct stores ngram_weights (an array of f64 weights for n-grams of different sizes), lsh_hash_functions (parameters for locality-sensitive hashing via MinHash), and a seed for deterministic behavior. The computeNgramScore function slides a window of configurable size across a token sequence, accumulating weighted scores for each n-gram found in the weights table.

The computeLSHSignature generates MinHash signatures by applying each hash function to every token and taking the minimum hash value per function, producing a compact fingerprint for approximate set similarity. The jaccardSimilarity function computes exact Jaccard similarity between two token sets, while cosineSimilarity computes the cosine of the angle between two floating-point vectors. The rank method combines multiple scoring criteria with configurable weights: n-gram relevance, LSH similarity to a query, token diversity (ratio of unique tokens to total), anchor proximity (distance to the nearest anchor token), and positional decay (exponential decay based on document position). All scores are normalized to [0,1] before weighted combination. The rankBatch method applies ranking to multiple documents and returns them sorted by score.

---

## 5. Entry Points

### src/main.zig — CPU Training Entry Point

The main.zig file is the primary entry point for CPU-based training, validation, and interactive inference. It imports modules from across the entire codebase and defines type aliases for commonly used types. The MainConfig struct contains dozens of compile-time constants: DEFAULT_EMBEDDING_DIM (128), DEFAULT_RSF_LAYERS (4), DEFAULT_BATCH_SIZE (16), DEFAULT_NUM_EPOCHS (10), DEFAULT_LEARNING_RATE (0.001), DEFAULT_GRADIENT_CLIP_NORM (5.0), DEFAULT_SEQUENCE_LENGTH (64), DEFAULT_TOP_K (5), ranker parameters (RANKER_NGRAM_SIZE at 10, RANKER_LSH_TABLES at 16, RANKER_SEED at 42), PRNG seeds (PRNG_SEED_FORWARD at 54321, PRNG_SEED_VALIDATION at 12345, PRNG_SEED_GRADIENT at 99999), and file format magic numbers (FILE_MAGIC_RSF as 0x4A524653, FILE_MAGIC_MGT as 0x4A4D4754, FILE_MAGIC_RANKER as 0x4A524E4B).

The Config struct holds runtime-configurable fields parsed from command-line arguments via parseArgs: embedding_dim, rsf_layers, batch_size, num_epochs, learning_rate, models_dir, vocab_file, dataset_path, sample_limit, gradient_clip_norm, sequence_length, top_k, and mode. The runKgruTest function validates five core components: RSF forward/inverse round-trip accuracy, MGT tokenization/detokenization consistency, SSI insert/retrieve correctness, Ranker scoring verification, and Tensor arithmetic properties. The training pipeline initializes RSF, MGT, and a projection matrix, loads training data, and runs epochs of forward pass through RSF, projection to vocabulary space, softmax loss computation, backpropagation, and SFD optimizer parameter updates, with periodic validation and checkpoint saving. An interactive REPL mode tokenizes user input, runs it through the trained model, and returns generated text.

### src/main_gpu.zig — Single-GPU Training Entry Point

The main_gpu.zig file targets single H100/B200 GPU training with Futhark acceleration. It initializes a Futhark context, creates an RSFAccelerator from the accelerator interface, loads training data, and runs the training loop using GPU-accelerated forward and backward passes. Each epoch iterates over batches, uploading data to the GPU as FutharkArray2DF16, calling the training step entry point (which performs the forward pass, loss computation, and backward pass entirely on the GPU), and downloading the scalar loss value. Checkpoints are saved after each epoch by downloading weight matrices from the GPU and serializing them.

### src/main_distributed.zig — Multi-GPU Distributed Training Entry Point

The main_distributed.zig file orchestrates multi-GPU training across multiple nodes. It initializes NCCL by having rank 0 generate a unique ID and broadcasting it to all ranks via file-based exchange (writing the ID to a shared filesystem path). Each rank initializes its own GPU, creates a DistributedTrainer, and enters the training loop. On each batch, the forward pass and loss computation are performed locally, gradients are computed via backpropagation, and then ncclAllReduce averages the gradients across all ranks. The averaged gradients are applied to update local parameters, ensuring all ranks maintain synchronized weights. A QuantumTrainingConfig struct enables optional quantum-GPU hybrid training, where specific subgraphs are offloaded to IBM Quantum hardware via the QuantumTaskAdapter during training.

### src/main_distributed_futhark.zig — Futhark Distributed Training Entry Point

The main_distributed_futhark.zig file combines Futhark GPU acceleration with NCCL-based distributed training. It uses PID-based NCCL file exchange (writing the unique ID to /tmp/nccl_id_{pid}), initializes a Futhark context on each rank's GPU, and runs the training loop with Futhark-accelerated forward/backward passes and NCCL gradient synchronization. Per-epoch checkpoints are serialized with rank-specific filenames, and the training loop logs per-batch loss values to stdout for monitoring.

### src/inference_server_main.zig — Inference Server Entry Point

The inference_server_main.zig file launches the HTTP inference server. It creates a ServerConfig with default values (port 8080, host "0.0.0.0", max_connections 100, timeout 30 seconds, batch_size 32), parses command-line arguments to override defaults, initializes the InferenceServer with the config and an allocator, and calls server.start() to begin accepting connections.

### src/bench_deps.zig and src/wasm_deps.zig — Dependency Re-Export Wrappers

The bench_deps.zig file re-exports SSI, RSF, Tensor, and RankedSegment types for use in benchmark executables, providing a single import point. Similarly, wasm_deps.zig re-exports MGT, RSF, Tensor, ModelFormat, and the importModel function for use in the WebAssembly build target.

---

## 6. API and Inference

### src/api/inference_server.zig — HTTP Inference Server

The inference_server.zig file implements a full HTTP server for model inference. The ServerConfig struct contains port, host, max_connections, timeout_seconds, batch_size, model_path, rate_limit_requests, rate_limit_window_seconds, and api_key fields. The RateLimiter struct uses a HashMap to track per-IP request counts within a sliding 60-second window, cleaning expired entries on each check. Bearer token authentication reads the JAIDE_API_KEY environment variable and validates Authorization headers via checkAuthorization.

The handleStreamConnection method manages the request-response cycle using an ArenaAllocator for per-connection memory that is freed en masse when the connection closes. The server manually parses HTTP requests from a buffer, dispatching to two endpoints: GET /v1/health returns uptime, model status, and version information as JSON; POST /v1/inference accepts a JSON body with text, max_tokens, and optional return_embeddings fields, tokenizes the input via MGT, runs it through RSF to generate embeddings, and returns the results as JSON. A BatchInferenceRequest variant processes multiple inputs in a single request for throughput optimization.

### src/wasm/wasm_bindings.zig — WebAssembly Bindings

The wasm_bindings.zig file provides the WASM API for running JAIDE in a browser. It maintains global state (MGT tokenizer, RSF model, and Tensor instances) behind a global mutex for thread safety. Exported functions include jaide_encode (tokenizing text and returning token IDs), jaide_decode (detokenizing IDs back to text), jaide_inference (running a full encode-forward-decode pipeline), and jaide_batch_inference (processing multiple texts). The exports use comptime conditional compilation to only generate the WASM export table when targeting a WASM architecture, keeping the functions available as normal Zig code for testing on native platforms.

---

## 7. Distributed Training

### src/distributed/distributed_trainer.zig — Distributed Training Engine

The distributed_trainer.zig file is a substantial 1,859-line module that contains a complete self-contained tensor implementation alongside the distributed training logic. It defines its own Shape struct (with dims, strides, and helper methods), a TensorData union (dense f32 array or sparse index-value pairs), and a COW-enabled Tensor with atomic refcount and mutex. This standalone tensor implementation includes a full suite of linear algebra: QR decomposition (Gram-Schmidt), SVD (one-sided Jacobi), eigendecomposition (iterative QR), Cholesky factorization, LU decomposition with partial pivoting, matrix inverse (Gauss-Jordan), determinant, and solve (LU with forward/backward substitution).

The DistributedTrainer struct manages NCCL communicators, GPU device assignments, and training state. The train method implements the distributed training loop: each rank processes its local batch (forward pass through RSF layers, cross-entropy loss computation, backward pass), then calls ncclAllReduce to average gradients across all ranks. Parameter updates are applied identically on each rank using the averaged gradients. Checkpoint serialization writes model weights, optimizer state, and training metadata to files with rank-specific naming. The load method restores from checkpoints, enabling training resumption after interruptions.

### src/distributed/distributed_trainer_futhark.zig — Futhark Distributed Trainer

The distributed_trainer_futhark.zig file implements distributed training using Futhark GPU kernels for computation. It manages f16 pinned memory for efficient CPU-GPU data transfer, uses FutharkArray2DF16 for GPU-side storage, and implements one-hot encoding for label preparation. The training loop calls Futhark entry points for forward pass, loss computation, and backward pass, then uses NCCL all-reduce for gradient synchronization. Checkpoints serialize the Futhark GPU arrays by downloading them to host memory and writing in a binary format that records shapes and raw f16 data.

### src/distributed/gpu_coordinator.zig — GPU Coordination Layer

The gpu_coordinator.zig file wraps CUDA and NCCL APIs into a high-level coordination interface. The GPUCoordinator struct stores rank, world_size, NCCL communicator, CUDA stream, and device ID. The init method queries CUDA devices, sets the device for this rank (rank modulo device count), creates a CUDA stream, and initializes the NCCL communicator. Memory operations include allocDeviceMemory (cudaMalloc), freeDeviceMemory (cudaFree), copyHostToDevice and copyDeviceToHost (cudaMemcpy). Collective operations include allReduceFloat32 and allReduceFloat16 (ncclAllReduce with ncclSum), broadcastFloat32 (ncclBroadcast with a root rank), synchronize (cudaStreamSynchronize), and barrier (all-reduce on a dummy value followed by sync).

### src/distributed/modal_gpu.zig — Modal Cloud GPU Client

The modal_gpu.zig file provides an HTTP client for deploying training jobs to the Modal cloud platform. The ModalGPUClient stores an API token and HTTP client. The deployTrainingJob method sends a POST request to Modal's API with a JSON payload specifying B200 GPUs, gpu_count (default 8), image name "jaide-v40-training," model and dataset paths, batch_size 32, and epochs 10. The getJobStatus method polls a job status endpoint by job ID and returns the response.

### src/distributed/nccl_bindings.zig — NCCL and CUDA FFI Declarations

The nccl_bindings.zig file provides Zig-native type definitions and extern function declarations for NVIDIA's NCCL and the CUDA runtime API. It defines ncclResult_t (9 variants from ncclSuccess through ncclNumResults), ncclDataType_t (10 data types from ncclInt8 through ncclBfloat16), and ncclRedOp_t (6 reduction operations). The NCCL communicator is an opaque type, and ncclUniqueId is a 128-byte struct. Extern function declarations cover ncclGetUniqueId, ncclCommInitRank, ncclCommDestroy, ncclAllReduce, ncclBroadcast, ncclReduce, ncclAllGather, ncclReduceScatter, and ncclGetErrorString. CUDA declarations include cudaGetDeviceCount, cudaSetDevice, cudaMalloc, cudaFree, cudaMemcpy, cudaStreamCreate, cudaStreamSynchronize, and cudaGetErrorString, plus a cudaMemcpyKind struct with constants for host-to-device (1), device-to-host (2), and device-to-device (3).

---

## 8. Hardware Acceleration

### src/hw/accel/accel_interface.zig — Accelerator Interface

The accel_interface.zig file serves as the primary bridge between the ML pipeline and GPU backends. The FutharkContext struct wraps a Futhark runtime context, configuring device index 0, group size 256, number of groups 128, and tile size 32. The PinnedMemory struct manages CUDA pinned host memory via cudaHostAlloc/cudaFreeHost, with asSlice reinterpreting the raw pointer as a typed Zig slice. The FutharkArray2DF16 represents a 2D f16 array on the GPU, supporting construction from Zig slices (new/newFromFlat/newZeros) and data retrieval back to host (values).

The RSFAccelerator struct holds a FutharkContext, two weight matrices and two velocity matrices (for momentum-based training), model dimension, and initialization flag. The forward method calls futhark_entry_rsf_forward with input and weight matrices. The trainingStep method performs a full iteration by invoking futhark_entry_training_step with inputs, targets, weights, velocities, learning rate, and momentum (the latter bit-cast from f16 to u16 for the C FFI), returning updated weights, velocities, and loss.

### src/hw/accel/cuda_bindings.zig — CUDA FFI Declarations

The cuda_bindings.zig file declares Zig extern bindings for the CUDA runtime API. It defines CudaError as an enum with variants for success and common error codes (invalid value, memory allocation failure, initialization error, launch failure, and a catch-all). Functions declared include cudaMalloc, cudaFree, cudaMemcpy, cudaMemset, cudaHostAlloc, cudaFreeHost, cudaSetDevice, cudaGetDevice, cudaGetDeviceCount, cudaStreamCreate, cudaStreamDestroy, cudaStreamSynchronize, cudaDeviceSynchronize, and cudaGetErrorString. A toError helper function maps CudaError values to Zig error types.

### src/hw/accel/fractal_lpu.zig — Fractal Language Processing Unit

The fractal_lpu.zig file implements a custom accelerator abstraction based on fractal hierarchy. The FractalDimensionConfig struct specifies min_depth, max_depth, branching_factor, and scale_ratio. The ComputeUnit struct models an individual processing element with local_memory, instruction_pointer, and utilization. The FractalTile struct is the recursive building block: each tile contains a list of child tiles, a list of compute units, a level number, and a computed fractal dimension. Tiles can be subdivided (creating child tiles with proportionally fewer compute units) or merged (combining children back into the parent).

The FractalLPU struct orchestrates the tile hierarchy. The init method creates a root tile and recursively subdivides it to the configured depth. The processInput method routes data through the tile hierarchy, with each tile applying its compute units to the input and passing results to child tiles for further processing. The computeFractalDimension method uses box-counting to estimate the fractal dimension of the tile hierarchy's processing pattern. Load balancing redistributes work across compute units based on their utilization metrics.

### src/hw/accel/futhark_bindings.zig — Futhark Runtime Bindings

The futhark_bindings.zig file declares Zig extern bindings for the Futhark runtime library. It defines opaque types for the Futhark context and configuration, and for 2D f16 arrays. Function declarations cover context management (futhark_context_config_new, futhark_context_new, futhark_context_free, futhark_context_sync), configuration (futhark_context_config_set_device, futhark_context_config_set_default_group_size, futhark_context_config_set_default_num_groups, futhark_context_config_set_default_tile_size), array management (futhark_new_f16_2d, futhark_free_f16_2d, futhark_values_f16_2d, futhark_shape_f16_2d), and entry points (futhark_entry_rsf_forward, futhark_entry_training_step, futhark_entry_rsf_backward, futhark_entry_scale_weights).

### src/hw/accel/futhark_kernels.fut — Futhark GPU Kernel Library

The futhark_kernels.fut file is a comprehensive GPU kernel library written in Futhark's functional array language. It implements matrix multiplication (matmul) using Futhark's map-reduce pattern over rows and columns. Activation functions include sigmoid, tanh, relu, gelu (Gaussian Error Linear Unit using the tanh approximation), and swish (x * sigmoid(x)). The RSF kernels implement rsf_scale_forward (computing exp(clip(W*x + b))), rsf_translate_forward (W*x + b), and full rsf_forward combining both with element-wise multiply and add. Loss functions include mean_squared_error and cross_entropy_loss.

SSI kernels implement parallel hash computation and similarity search using Futhark's map and reduce_by_key primitives. RGPU graph kernels implement parallel edge weight updates and node state propagation. Quantum simulation kernels implement Hadamard, CNOT, and rotation gates operating on state vectors. Utility kernels provide vector_add, vector_scale, softmax (with numerical stability via max subtraction), and layer_norm (computing mean, variance, and normalizing with epsilon).

### src/hw/accel/main.fut — Futhark Training Pipeline

The main.fut file implements the half-precision RSF training pipeline in Futhark. It defines matmul_f16 for f16 matrix multiplication, rsf_forward_f16 for the forward pass (computing scale via exp(clip(W_s * x + b_s)) and translation via W_t * x + b_t), rsf_backward_f16 for gradient computation through the coupling layer, and training_step as the top-level entry point. The training step performs a forward pass, computes MSE loss, backpropagates gradients, and updates weights using momentum SGD (v = momentum * v - lr * grad, w = w + v). Batch operations process multiple samples by mapping over the batch dimension.

---

## 9. Hardware Synthesis

### hw/fpga/top_level.v — FPGA System-on-Chip

The top_level.v file is a Verilog module implementing a system-on-chip for the iCE40-HX8K FPGA target. The top module instantiates four sub-modules connected via an AXI4-Lite bus: axi4_lite_slave (handling register reads/writes with address decoding, write/read state machines, and a 16-register bank), ssi_search_engine (implementing the SSI hash lookup in hardware with a 1024-entry hash table, Wyhash computation, and collision chain traversal), ranker_core (computing n-gram scores with pipelined multiply-accumulate units and parameterized weights), and memory_arbiter (coordinating memory access between the SSI and ranker modules using round-robin arbitration with priority override). The clock and reset distribution uses a PLL for clock generation from the external oscillator.

### hw/asic/synthesis.tcl — ASIC Synthesis Script

The synthesis.tcl file is a Synopsys Design Compiler script targeting TSMC 28nm technology. It reads the Verilog source from hw/fpga/top_level.v, sets the target library to tcbn28hpmbwp30p140_ccs.db (a TSMC 28nm high-performance cell library), defines the clock at 100 MHz with a 50% duty cycle, sets input/output delays at 30% of the clock period, specifies maximum area constraint, applies compile_ultra with timing-driven and scan-insertion options, generates timing reports, area reports, and power estimates, and writes out the synthesized netlist in Verilog format.

### hw/asic/floorplan.tcl — ASIC Floorplanning

The floorplan.tcl file is an IC Compiler floorplanning script. It initializes the design, creates a rectangular floorplan with a core utilization of 70%, adds power distribution (VDD and VSS rings and straps with configurable width and spacing), places I/O pins on the die perimeter (clock on the south edge, reset on the west, data bus on the east, address on the north), performs global placement, creates clock tree synthesis constraints, runs routing, performs timing analysis with setup and hold checks, and generates the final layout for tape-out.

---

## 10. Fuzz Testing

### hw/fuzz/fuzz_memory.zig — Memory Allocator Fuzzer

The fuzz_memory.zig file stress-tests the project's memory allocators through randomized operations. The FuzzConfig struct specifies 10,000 iterations and seed 42. Each iteration randomly selects one of three operations: allocate (requesting a random-sized block between 1 and 65536 bytes and verifying the returned pointer is valid), free (releasing a randomly chosen previously-allocated block and verifying no double-free occurs), or reallocate (resizing a random existing allocation and verifying data preservation up to the minimum of old and new sizes). The fuzzer tracks all active allocations in an ArrayList, verifies that freed memory is properly zeroed (due to the security zeroing policy), and reports total allocations, frees, reallocations, and any failures at the end.

### hw/fuzz/fuzz_ssi.zig — SSI Index Fuzzer

The fuzz_ssi.zig file stress-tests the SSI data structure with 5,000 iterations. Each iteration randomly chooses between inserting a document (generating a random-length token array with values modulo 50,000, a sequential position, and random anchor status) and querying (generating random tokens and retrieving the top-10 results). Progress reports every 500 iterations show the current document count. Final statistics include successful inserts, queries, failures, and total tokens indexed.

### hw/fuzz/fuzz_tensor.zig — Tensor Operations Fuzzer

The fuzz_tensor.zig file stress-tests the Tensor with 5,000 iterations. Each iteration creates a tensor with random rank (1-4) and random dimension sizes (1-256), skipping configurations exceeding one million total elements. The tensor is filled with random floats in [-1, 1], then a random operation is applied: sum reduction, max reduction, L2 norm, or scalar multiplication. Non-finite results count as failures. If the failure rate exceeds 10%, the fuzzer returns HighFailureRate error.

---

## 11. Zero-Knowledge Circuits

### src/zk/inference_trace.circom — ZK Inference Verification Circuit

The inference_trace.circom file defines Circom arithmetic circuits for zero-knowledge proof generation of JAIDE inference traces. The PoseidonChain template implements a chain of Poseidon hash computations, taking an array of inputs and producing a single hash output by iteratively hashing pairs of values. The RSFLayerComputation template models a single RSF coupling layer in the arithmetic circuit, with constraints for the scale and translation sub-network computations and element-wise operations.

The MerkleVerification template verifies Merkle tree membership proofs within the circuit, taking a leaf value, a proof path of sibling hashes, and path direction bits, and constraining that the computed root matches the expected public root. The DifferentialPrivacyVerification template constrains that added noise satisfies the configured privacy budget without revealing the actual noise values. The ProofAggregation template combines multiple individual inference proofs into a single aggregate proof by constructing a Merkle tree over all individual proof commitments. The main component wires these templates together: it takes inference inputs, weights, and the computed output as private signals, the commitment hash and privacy budget as public signals, and constrains that the inference computation is correct, the commitment matches, and the privacy budget is satisfied.

---

## 12. WebAssembly Target

### src/wasm/wasm_bindings.zig — Browser-Side JAIDE

The WASM bindings file provides the API for running JAIDE inference directly in a web browser. It maintains global state for an MGT tokenizer, RSF model, and working tensors behind a mutex. Exported WASM functions include jaide_init (loading model data from a memory buffer), jaide_encode (tokenizing text), jaide_decode (detokenizing IDs), jaide_inference (full encode-forward-decode pipeline), and jaide_batch_inference (processing multiple texts). The exports use comptime target detection to only generate WASM export symbols when compiling for wasm32-freestanding, keeping the code testable on native platforms.

---

## 13. Test Suite

### src/tests/stress_tensor_refcount.zig — Multi-Threaded Stress Test

The stress_tensor_refcount.zig file verifies the thread-safety of Tensor's reference counting mechanism. The TestConfig struct defines three parameters: num_threads, ops_per_thread, and num_tensors. The ThreadContext carries shared tensors, thread ID, an atomic barrier, and operation count. The threadWorker function synchronizes all threads to start simultaneously using the atomic barrier, then performs randomized operations on shared tensors: 50% probability of single retain/release, 25% of double retain/release, 15% of dual-tensor retain/release, and 10% of burst (5 retains followed by 5 releases). Random Thread.yield() calls increase scheduling variability.

The runStressTest function allocates 64x64 tensors, verifies initial refcounts of 1, spawns all workers, waits for completion, measures throughput, and critically verifies that every tensor's refcount has returned to exactly 1 — proving all retains were balanced by releases despite concurrent chaos. The default configuration uses 12 threads, 15,000 operations per thread (180,000 total), and 8 shared tensors. Three unit tests provide quick smoke testing at reduced scale.

---

## 14. Python Scripts

### src/scripts/modal_distributed_train.py — Modal Distributed Training

The modal_distributed_train.py file defines a Modal application for distributed GPU training on 8x NVIDIA B200 GPUs. It creates a Modal Image based on Debian Slim with CUDA runtime libraries, the Zig compiler, Python dependencies (numpy, torch for data loading only), and the JAIDE source code copied into the container. A Modal Volume at /checkpoints persists training checkpoints across runs. The main training function uses Modal's gpu="B200" with count=8, compiles the Zig distributed trainer inside the container, downloads the SZTAKI-HLT/HunSum-1 dataset (1.1M Hungarian article-summary pairs), converts it to the JAIDE binary format, and launches the distributed training process with MPI-style rank assignment. Training logs are written to the volume alongside checkpoints.

### src/scripts/modal_train.py — Modal Single-GPU Training

The modal_train.py file defines a simpler Modal application for single-GPU training. It uses typed parameters for configuration (embedding_dim, rsf_layers, batch_size, epochs, learning_rate) with sensible defaults. The training function compiles the JAIDE main binary, prepares the dataset, and runs training on a single B200 GPU. Checkpoints are saved to a Modal Volume and can be downloaded after training completes.

### src/scripts/modal_inference.py — Modal Inference

The modal_inference.py file defines a Modal application for running inference with a trained model. It loads the latest checkpoint from the Modal Volume, compiles the inference server binary, and exposes an HTTP endpoint that accepts text input and returns model predictions. The auto-model selection logic finds the most recent checkpoint by timestamp in the /checkpoints directory.

### src/scripts/modal_setup.sh — Modal Environment Setup

The modal_setup.sh script automates Modal CLI installation and authentication. It installs the modal Python package via pip, runs modal token set to configure API credentials from environment variables, and verifies the setup by running modal app list to confirm connectivity to Modal's servers.

### src/scripts/check_proofs_all.py — Multi-Framework Proof Checker

The check_proofs_all.py file orchestrates type-checking across all formal verification frameworks. It defines checker functions for each framework: run_agda (invoking agda --safe on each .agda file), run_lean (invoking lake build in the lean4 directory), run_isabelle (invoking isabelle build on theory files), run_viper (invoking silicon on .vpr files), run_tla (invoking tlc on .tla files), and run_spin (invoking spin -a, gcc, and ./pan -a on .pml files). Each checker captures stdout/stderr, reports pass/fail, and accumulates results. The main function runs all checkers and prints a summary with total/passed/failed counts.

### src/scripts/execution_trace.py — System Architecture Mapper

The execution_trace.py file maps the JAIDE system architecture by analyzing source file imports and generating a dependency graph. It parses Zig @import statements and Python import statements, builds a directed graph of module dependencies, computes topological ordering, and identifies circular dependencies. The output includes a textual dependency tree and statistics on module coupling (fan-in, fan-out) and cohesion metrics.

### src/scripts/generate_proof_skeleton.py — Proof Scaffolding Generator

The generate_proof_skeleton.py file automatically generates proof skeleton files from Zig source code. It parses Zig struct definitions, function signatures, and error sets, then generates corresponding proof obligations in multiple frameworks: Agda modules with data type declarations and theorem stubs, Lean 4 structures with theorem statements and sorry placeholders, Isabelle theories with datatype and lemma declarations, and TLA+ specifications with variable and invariant definitions. This accelerates the verification workflow by automatically creating the boilerplate that proof engineers then fill in with actual proofs.

### src/scripts/verify_coverage.sh — Verification Coverage Reporter

The verify_coverage.sh script measures how much of the Zig codebase is covered by formal verification. It counts lines of code in each Zig module, counts corresponding lines of proof across all verification frameworks, and computes a coverage ratio. Target thresholds are defined per module (e.g., tensor.zig should have at least 80% proof coverage, memory.zig at least 70%). The script reports per-module and aggregate coverage percentages and flags modules that fall below their targets.

---

## 15. Top-Level Scripts

### scripts/bootstrap_verification_libs.sh — Verification Library Setup

The bootstrap_verification_libs.sh script performs one-time installation of all verification framework dependencies. For Lean 4, it clones and builds Mathlib4. For Isabelle, it downloads the Archive of Formal Proofs (AFP). For Agda, it installs the standard library and cubical library. For Viper, it downloads the Silicon and Carbon verifiers. For SPIN, it installs the SPIN model checker. For TLA+, it downloads the TLC model checker JAR file. Each section checks whether the dependency is already installed before proceeding.

### scripts/fpga_synthesis.sh — FPGA Synthesis Pipeline

The fpga_synthesis.sh script implements the full FPGA build pipeline for the iCE40-HX8K target. It first runs Clash (Haskell-to-Verilog compiler) if any .clash files are present, then runs Yosys for synthesis (reading Verilog, running synth_ice40, and writing a JSON netlist), nextpnr-ice40 for place-and-route (targeting the ct256 package with timing-driven placement), icepack for bitstream generation, and optionally iceprog for programming the FPGA via USB. Timing reports are generated at each stage.

### scripts/profile_edge_fractal.sh — Edge Device Profiling

The profile_edge_fractal.sh script profiles JAIDE performance on edge devices. It compiles the Zig binary with release-fast optimization, runs inference benchmarks while capturing power consumption (via powertop or platform-specific tools), measures latency percentiles (p50, p95, p99), throughput in tokens per second, and memory footprint. Results are output in a structured format for comparison across device configurations.

### scripts/run_profiling.sh — Comprehensive Profiling Suite

The run_profiling.sh script runs a comprehensive profiling suite. It executes perf stat for hardware counter collection (cycles, instructions, cache misses, branch mispredictions), perf record with call-graph capture for flamegraph generation, Valgrind's callgrind for instruction-level profiling, Valgrind's massif for heap profiling, and a custom stress test measuring throughput under load. Results are collected in a timestamped directory with flamegraph SVGs generated via stackcollapse-perf.pl and flamegraph.pl.

### scripts/verify_all.sh — Master Verification Script

The scripts/verify_all.sh script is the top-level entry point for running all verifications. It calls the framework-specific scripts in sequence, collects results, and produces a unified report with pass/fail status for each verification target. A non-zero exit code is returned if any verification fails, enabling CI/CD integration.

### src/verification/verify_all.sh — Verification Suite Runner

The src/verification/verify_all.sh script coordinates verification across all six formal methods frameworks. It initializes counters, defines a run_check function that executes a command and logs results, then processes each framework: Agda (agda --safe on 6 core files), Lean 4 (lake build), Isabelle (isabelle build on 4 theories), Viper (silicon on 2 files), TLA+ (tlc on 2 specs), and SPIN (spin -a, gcc, ./pan -a pipeline). Each framework section is skipped with a message if the tool is not installed. The summary prints total/passed/failed counts and exits with code 1 if any checks failed.

---

## 16. Formal Verification Suite — Agda (21 files)

### src/verification/agda/Tensor.agda

Tensor.agda serves as the foundational module for tensor operations, building upon Agda's built-in floating-point primitives. It defines custom infix operators for floating-point addition, multiplication, subtraction, and division, along with constants for zero, one, and epsilon. Core data types include TensorError (an enumeration of OutOfBounds, ShapeMismatch, InvalidAxis, Overflow, AllocationFailed, and DivisionByZero), COWState (Exclusive and Shared), MutexState (tracking lock status and owner), and RefCount (wrapping a natural number). The Shape record carries a vector of dimension sizes with a proof that every dimension is strictly positive. The Tensor record bundles data, shape, reference count, COW state, and mutex.

The file provides tensor operations and their correctness theorems. Initialization creates tensors filled with zeros or a given value. Copy-on-write is implemented through tensor-retain, tensor-mark-shared, and tensor-ensure-writable, with theorems proving that ensure-writable always produces an Exclusive tensor, copying yields a fresh refcount of one, and data is preserved. Arithmetic operations via zipWith include tensor-add, tensor-sub, tensor-mul, and tensor-div (with division-by-zero error handling). Commutativity and associativity of addition and multiplication, distributivity of scalar multiplication, and correctness of flatten/unflatten as inverses are all proven. COW isolation theorems prove that ensuring writability on a view always produces Exclusive ownership, a shared tensor made writable receives fresh resources, and creating views preserves Shared state.

### src/verification/agda/TensorVerification.agda

TensorVerification.agda extends the foundation with operation-specific proofs. It proves that tensor initialization produces a tensor whose data vector contains all zeros, that reshape preserves total element count, that broadcasting maintains the broadcasting compatibility relation, and that slice produces a tensor whose dimensions match the requested sub-range. Matrix multiplication shape correctness is proven: given inputs of shape [M,K] and [K,N], the output shape is [M,N]. The convolution output size formula is verified: (input_size - kernel_size + 2*padding) / stride + 1. Reduction operations are shown to produce scalars (single-element tensors). The file uses structural induction on dimension lists, vector element-wise reasoning, and arithmetic properties from the standard library.

### src/verification/agda/TensorComplete.agda

TensorComplete.agda provides the most comprehensive tensor verification, combining shape algebra, element-wise operation proofs, and memory safety guarantees into a single module. It proves that shape validation rejects zero dimensions, that stride computation is correct for row-major layout, that total size computation is associative over shape concatenation, and that reshape-round-trip (reshape then reshape back) preserves the original shape when total sizes match. SIMD vectorization safety is addressed by proving that the vector loop processes exactly floor(n/4)*4 elements and the scalar tail handles the remaining n mod 4 elements, together covering all n elements. The copy-on-write invariant is proven: after ensure-writable, the tensor has exclusive ownership and its data is a deep copy, meaning modifications cannot affect other tensors that previously shared the same data.

### src/verification/agda/TensorCompleteExpanded.agda

TensorCompleteExpanded.agda is the largest Agda file, providing exhaustive proofs for every tensor operation. It defines a VerificationResult type tracking operation name, property verified, and proof status. For each arithmetic operation, it proves commutativity, associativity, identity element existence, and (where applicable) distributivity. For matrix multiplication, it proves associativity ((A*B)*C = A*(B*C)), transpose-multiplication duality ((A*B)^T = B^T*A^T), and identity matrix properties. For SVD, it proves that U and V are orthogonal (U^T*U = I, V^T*V = I) and that S contains non-negative singular values in non-increasing order. For QR decomposition, it proves Q is orthogonal and R is upper triangular. For eigendecomposition, it proves the eigenvalue equation (A*v = lambda*v). Cholesky is verified as producing a lower triangular L such that L*L^T equals the original matrix. LU decomposition is verified with P*A = L*U where P is a permutation matrix.

### src/verification/agda/TensorArithmeticLemmas.agda

TensorArithmeticLemmas.agda provides specialized arithmetic proofs. It proves that element-wise addition commutes and associates, that scalar multiplication distributes over addition, that the zero tensor is the additive identity, and that negation produces the additive inverse. For floating-point arithmetic specifically, it proves that operations are deterministic (same inputs always produce the same outputs) even though they may not be associative due to rounding, and establishes bounds on rounding error for each operation. The epsilon-ball property shows that the difference between exact and floating-point arithmetic results is bounded by machine epsilon times the magnitude of the operands.

### src/verification/agda/TensorMatrixLemmas.agda

TensorMatrixLemmas.agda proves matrix-specific properties. Matrix multiplication associativity is proven by showing element-wise equivalence: the (i,j) element of (A*B)*C equals the sum over k of (sum over l of A[i,l]*B[l,k])*C[k,j], which by distributivity of multiplication over addition equals the sum over l of A[i,l]*(sum over k of B[l,k]*C[k,j]), which is the (i,j) element of A*(B*C). The trace of a product is shown to be symmetric: tr(A*B) = tr(B*A). The determinant of a product equals the product of determinants. Matrix inverse properties include (A^-1)^-1 = A and (A*B)^-1 = B^-1*A^-1.

### src/verification/agda/TensorShapeLemmas.agda

TensorShapeLemmas.agda focuses on shape algebra. It proves that shape products are associative and commutative, that broadcasting is symmetric (if A can broadcast to B, then B can broadcast to A when sizes match), that reshape preserves total element count, and that concatenation along an axis produces the correct output shape. Stride computation correctness is verified by induction on the dimension list, showing that strides[i] = product of dims[i+1..n] for row-major order.

### src/verification/agda/TensorVectorLemmas.agda

TensorVectorLemmas.agda provides vector-specific proofs. Dot product commutativity and bilinearity are proven. The Cauchy-Schwarz inequality is stated. Vector normalization produces a unit vector (norm = 1). Cross product anti-commutativity (a × b = -(b × a)) and the Jacobi identity are verified. Vector projection properties show that the projection of a onto b is parallel to b and orthogonal to the rejection component.

### src/verification/agda/Memory.agda

Memory.agda formalizes the memory management subsystem. It defines allocator types (Arena, Pool, Slab, Buddy) as an enumeration, memory blocks as records with address, size, and allocator provenance, and allocation results as Either of error or block. The arena allocator model proves that allocations never overlap (two blocks from the same arena have disjoint address ranges), that total allocated size never exceeds arena capacity, and that arena reset frees all blocks simultaneously. The pool allocator model proves that all blocks have identical sizes, that the free list maintains LIFO ordering, and that double-free is detectable. The buddy allocator model proves that block sizes are always powers of two, that buddy pairs are correctly identified (XOR of address with size), and that merge-on-free produces the correct parent block.

### src/verification/agda/MemoryAllocators.agda

MemoryAllocators.agda extends the memory formalization with detailed allocator-specific invariants. For the slab allocator, it proves that slab sizes are fixed at construction time, that the object freelist correctly recycles within slabs, and that the slab list maintains non-decreasing address order. For the buddy allocator, it proves the fundamental buddy property: that after splitting a block of size 2^k into two blocks of size 2^(k-1), freeing both buddies and merging produces the original block. Security zeroing is proven: after deallocation, every byte of the freed block is set to zero before the block is returned to the freelist, ensuring no data leakage between allocations.

### src/verification/agda/MemoryVerification.agda

MemoryVerification.agda verifies cross-cutting memory properties. It proves that all six allocators respect alignment requirements (returned addresses are multiples of the requested alignment), that concurrent allocation and deallocation using mutex-protected allocators is serializable (the sequence of allocations and frees produces the same result regardless of thread interleaving), and that memory statistics tracking is accurate (total_allocated equals the sum of all active allocation sizes, and peak_usage is the maximum of all total_allocated values observed). The proof of leak detection shows that if deinit is called with any active allocations remaining, the leak count equals the number of unfreed blocks.

### src/verification/agda/Types.agda

Types.agda formalizes the core type system. It defines FixedPoint16, FixedPoint32, and FixedPoint64 as records with natural number values and proves arithmetic properties: commutativity and associativity of addition for all three sizes, commutativity of multiplication for FP32, and the existence of additive identity. The PRNG formalization models xoshiro256** as a state transition function on a 4-tuple of u64 values and proves that the period is 2^256-1 (by showing the state transition function is a bijection on the non-zero state space). BitSet operations are verified: set followed by isSet returns true, unset followed by isSet returns false, and count equals the number of set bits.

### src/verification/agda/TypesVerification.agda

TypesVerification.agda provides extended verification of the type system. It proves that fixed-point multiplication distributes over addition for all three precisions, that conversion between fixed-point and floating-point round-trips with bounded error (the error is at most half the least significant bit), and that clamp is idempotent. The ContextWindow is verified to maintain FIFO ordering and bounded size. RankedSegment comparison is shown to define a total order.

### src/verification/agda/RSF.agda

RSF.agda formalizes the Reversible Scale Flow architecture. It defines the RSFLayer with weight and bias matrices and proves the fundamental reversibility property: for any input pair (x1, x2), applying forward and then inverse produces exactly (x1, x2). The proof proceeds by algebraic manipulation, showing that the inverse correctly undoes each step of the forward computation. Xavier initialization bounds are proven to maintain activation variance: the variance of the output of a randomly initialized layer equals the variance of its input (up to a factor of 1 ± epsilon for finite sample sizes). Gradient computation is verified by proving that the backward method computes the exact Jacobian of the forward function via the chain rule.

### src/verification/agda/RSFVerification.agda

RSFVerification.agda extends the RSF proofs. It proves that multi-layer RSF maintains reversibility (composing reversible layers preserves reversibility by induction), that gradient clipping preserves the gradient direction (clipped gradient is parallel to the original), that save/load preserves model parameters exactly (round-trip serialization is the identity function on model state), and that batch processing is equivalent to sequential processing (processing a batch produces the same results as processing each element individually, since all operations are element-wise or matrix-level without cross-sample interaction).

### src/verification/agda/RSF_Processor_Complete.agda

RSF_Processor_Complete.agda provides the most comprehensive RSF verification. It formalizes the RSF processing pipeline as a sequence of stages: tokenization, embedding, RSF forward pass, projection, and output. The pipeline correctness theorem proves that the composition of all stages produces a valid probability distribution (all outputs are non-negative and sum to one, given that the final softmax is included). The numerical stability property shows that using max-subtracted softmax prevents overflow for any input whose elements are bounded by the clipping range. The training convergence theorem states that under certain conditions on the learning rate and data distribution, the loss function decreases monotonically (modulo stochastic noise).

### src/verification/agda/Optimizer.agda

Optimizer.agda formalizes the SFD optimizer. It models the optimizer state as a triple of parameter vector, first-moment estimate, and second-moment estimate. The update rule is formalized as: first_moment = beta1 * first_moment + (1-beta1) * gradient, second_moment = beta2 * second_moment + (1-beta2) * gradient^2, corrected_first = first_moment / (1-beta1^t), corrected_second = second_moment / (1-beta2^t), and parameter = parameter - learning_rate * corrected_first / (sqrt(corrected_second) + epsilon). The proof shows that the bias-corrected estimates converge to the true first and second moments as t approaches infinity.

### src/verification/agda/SFDVerification.agda

SFDVerification.agda extends the optimizer proofs. It proves that the learning rate scheduler produces monotonically decreasing learning rates for the exponential and step-decay schedules, that cosine annealing stays within [min_lr, max_lr], and that one-cycle reaches peak_lr at the midpoint. Gradient clipping is shown to preserve the descent direction while bounding the step size. The spectral normalization proof shows that after normalization, the weight matrix has spectral norm exactly 1.

### src/verification/agda/MGTVerification.agda

MGTVerification.agda formalizes the tokenizer. It proves that encoding followed by decoding produces the original text (round-trip property) for in-vocabulary texts, that the vocabulary maintains the bijection invariant (token_to_id and id_to_token are mutual inverses), that BPE merge order is deterministic (the same corpus always produces the same merge sequence), and that morphological decomposition is sound (the concatenation of prefix + root + suffix equals the original word).

### src/verification/agda/Tokenizer.agda

Tokenizer.agda provides foundational tokenizer definitions. It defines a MorphemeNode record (text, unique_id, frequency) and a MorphemeGraph record (vertices, edges, counts with validity proofs). The add-morpheme function maintains the invariant that vertex_count equals the length of the vertices list. The tokenize function performs greedy matching of morphemes against input text, and the tokenize-produces-valid-ids theorem proves that all output token IDs are valid indices into the graph's vertex list.

### src/verification/agda/Tokenizer_MGT_Complete.agda

Tokenizer_MGT_Complete.agda provides the most comprehensive tokenizer verification. The TokenizerState bundles a graph with vocabulary and validity proofs. GraphConsistency ensures all edge endpoints are valid vertex indices. The initial-graph-consistency lemma proves the empty graph is consistent. The add-morpheme-preserves-consistency lemma proves adding a morpheme preserves consistency. The TokenizerInvariant bundles four properties: vocabulary size correctness, graph count correctness, and graph consistency. The build-vocab-preserves-invariant lemma proves by induction that the invariant is maintained throughout training. The tokenizer determinism theorem proves determinism for any state satisfying the invariant.

---

## 17. Formal Verification Suite — Lean 4 (17 files)

### src/verification/lean4/Tensor.lean

Tensor.lean establishes the foundational formal model in Lean 4, importing from Mathlib. It defines TensorError, COWState, MutexState, and RefCount types. The core TensorSpec structure is parameterized by shape and carries a data vector, refcount, COW state, and mutex. Operations include initialization, retain, mark-shared, ensure-writable (with deep copy), view creation, release, element access, fill, pointwise arithmetic, and reductions. Theorems cover shapeSizePositive, ensureWritableExclusive, COW fresh-resources guarantees, commutativity and associativity of arithmetic, distributivity of scalar multiplication, fill uniformity, summation linearity, and broadcast symmetry. Proof tactics include induction, simp, omega, cases, ext, and ring.

### src/verification/lean4/TensorVerification.lean

TensorVerification.lean proves operational properties: reshape preserves element count (using omega and list length reasoning), element access returns the correct value (by unfolding the flat index computation), fill creates uniform values, pointwise operations preserve shape, and reduction produces a single-element tensor. Matrix multiplication output shape is verified as [m, n] given inputs [m, k] and [k, n].

### src/verification/lean4/TensorComplete.lean

TensorComplete.lean unifies shape algebra, SIMD safety, and COW correctness. SIMD loop safety proves that vector-width processing plus scalar tail covers all elements exactly. COW copy isolation proves that modifications after ensure-writable cannot affect shared views. Shape concatenation distributes over size computation. The complete verification bundle ties all properties together into a single theorem parameterized by arbitrary tensor configurations.

### src/verification/lean4/Memory.lean

Memory.lean formalizes memory allocation. It defines AllocatorType, MemoryBlock, AllocationResult, and AllocatorState structures. The arena allocator model proves monotonic growth (arena pointer never decreases) and bulk-free correctness (reset frees all). Pool allocator proofs cover fixed-size allocation, free-list integrity, and no-overlap guarantees. Security zeroing after free is modeled and proven to clear all bytes.

### src/verification/lean4/Memory_Complete.lean

Memory_Complete.lean provides exhaustive memory verification. It proves mutual exclusion for mutex-protected allocators (at most one thread holds the lock), leak detection accuracy (leak count equals unfreed blocks), and alignment correctness (all returned addresses satisfy the requested alignment). Cross-allocator proofs show that blocks from different allocator instances never overlap.

### src/verification/lean4/MemoryVerification.lean

MemoryVerification.lean bridges memory models with the Zig implementation. It defines a state machine for each allocator, where states are memory configurations and transitions are alloc/free operations. The safety property proves that no transition sequence produces an invalid state (use-after-free, double-free, or buffer overflow). The liveness property proves that a free always eventually succeeds (no permanent memory exhaustion in the buddy allocator due to fragmentation), given that the total allocation request is bounded by the initial capacity.

### src/verification/lean4/Types.lean

Types.lean formalizes fixed-point arithmetic. FixedPoint32 uses a 16.16 format with multiplication via intermediate 64-bit computation. Proofs establish addition commutativity and associativity, multiplication commutativity, zero identity, and bounded rounding error for fixed-float conversion (error ≤ 2^-16). The PRNG xoshiro256** is modeled as a deterministic function with theorems for period lower bound and uniform distribution of output bits.

### src/verification/lean4/TypesVerification.lean

TypesVerification.lean extends type proofs with distributivity of FP32 multiplication over addition, idempotence of clamp, and correctness of GCD computation (the result divides both inputs). Complex fixed-point arithmetic (complex_add, complex_mul) is proven to satisfy commutativity and the correct algebraic structure.

### src/verification/lean4/TypeTheory_Complete.lean

TypeTheory_Complete.lean formalizes the dependent type system. It defines TypeKind with all 21 constructors, the recursive Type structure, and capture-avoiding substitution. The substitution lemma proves that substituting a variable not free in a type produces the type unchanged. Type checking decidability is stated: for any context, term, and type, either a derivation exists or a counterexample can be constructed. The Pi-type introduction and elimination rules are proven sound.

### src/verification/lean4/RSFVerified.lean

RSFVerified.lean proves RSF network properties in Lean 4. Reversibility is proven by showing forward composed with inverse equals identity. The Jacobian determinant of the coupling layer is shown to be the product of the scale factors, enabling exact log-determinant computation. Weight initialization variance is bounded, and gradient norm clipping is shown to preserve descent direction while bounding step size to clip_max * gradient_direction.

### src/verification/lean4/SFDVerification.lean

SFDVerification.lean formalizes the SFD optimizer. The Adam update rule is expressed as a recurrence, and bias correction convergence is proven. Learning rate schedule monotonicity is proven for exponential and step-decay schedules. Fisher information preconditioning is shown to reduce the effective condition number of the optimization landscape.

### src/verification/lean4/MGTVerification.lean

MGTVerification.lean proves tokenizer correctness. The vocabulary bijection invariant (token_to_id and id_to_token are mutual inverses) is maintained by addVocabWord and removeVocabWord. BPE training determinism is proven. Encode-decode round-trip is proven for in-vocabulary text. Morphological decomposition is verified: prefix + root + suffix equals the original word when decomposition succeeds.

### src/verification/lean4/SSI_Index_Complete.lean

SSI_Index_Complete.lean formalizes the Sparse Segment Index. Insertion correctness proves that after addSequence, the entry can be retrieved by its hash key. Collision handling proves that chained entries are all preserved and distinguishable. Retrieval completeness proves that retrieveTopK returns the K nearest entries by Hamming distance, or all entries if fewer than K exist. Merge correctness proves that merging two indices produces an index containing exactly the union of both inputs.

### src/verification/lean4/Optimizer_SFD_Complete.lean

Optimizer_SFD_Complete.lean provides exhaustive optimizer verification. Each of the six learning rate schedules is verified to produce values within [min_lr, max_lr]. Mixed precision training is shown to preserve numerical correctness within a bounded error proportional to the f16 epsilon. The spectral normalizer's power iteration is shown to converge to the dominant singular value. The KFAC approximation error is bounded in terms of the Kronecker product approximation error of the Fisher matrix.

### src/verification/lean4/crev_pipeline.lean

The crev_pipeline.lean file formalizes the CREV knowledge extraction pipeline. It implements SHA-256 from scratch in Lean 4 (including Ch, Maj, sigma functions, round constants, message schedule expansion, compression, and padding), with proofs that the output is always 32 bytes and that the hash function is deterministic. ExtractionStage progression is verified with injective toString, round-trip fromString/toString, and ordinal consistency. RelationalTriplet initialization clamps confidence, computes identity hash, and clone is proven to be the identity function. Triplet equality is proven reflexive, symmetric, and transitive. Knowledge graph conversion produces edges with strength equal to clamped confidence.

### src/verification/lean4/rgpu.lean

The rgpu.lean file formalizes the RGPU mesh architecture. Processing core state transitions are modeled as a finite state machine with proven reachability from idle to any other state. XY routing correctness is proven: messages always reach their destination in at most width + height - 2 hops with no routing loops. Graph isomorphism via canonical forms is proven sound (isomorphic graphs produce equal canonical hashes). Power gating is shown to be safe: a gated core can always be woken within bounded latency.

### src/verification/lean4/surprise_memory.lean

The surprise_memory.lean file formalizes the surprise memory manager. The surprise score computation is shown to be monotonically decreasing as entries become less novel (more similar to existing memory contents). Eviction correctness proves that the evicted entry always has the minimum surprise score. Memory bounds prove that the store never exceeds its capacity. Thread safety is modeled through a mutex-based linearizability argument.

---

## 18. Formal Verification Suite — Isabelle/HOL (8 files)

### src/verification/isabelle/Tensor.thy

Tensor.thy defines tensor types in Isabelle/HOL using algebraic datatypes. TensorShape is a list of natural numbers. TensorData wraps a list of natural numbers. COW_State distinguishes COW_Exclusive and COW_Shared. MutexState records lock status and owner. TensorSpec bundles data, shape, COW state, mutex, and refcount. Operations include tensor_init, tensor_retain, tensor_mark_shared, tensor_ensure_writable, tensor_view, tensor_release, tensor_add, tensor_sub, tensor_mul, tensor_get, and tensor_set. Proofs establish that ensure_writable produces Exclusive state, that retain increments refcount, that view creates shared copies, and that arithmetic is commutative and associative.

### src/verification/isabelle/TensorComplete.thy and Tensor_Complete.thy

TensorComplete.thy and Tensor_Complete.thy provide comprehensive tensor verification. They prove shape_size_positive (product of positive dimensions is positive), reshape_preserves_size, broadcast_symmetric, SIMD loop coverage (vector loop plus scalar tail processes all elements), COW isolation (ensure_writable produces independent data), and matrix multiplication associativity. The two files differ slightly in their formalization approach: TensorComplete uses primrec definitions while Tensor_Complete uses function definitions with pattern matching.

### src/verification/isabelle/Types.thy

Types.thy defines fixed-point types as algebraic datatypes: FP16 (FixedPoint16 wrapping an int), FP32, and FP64. Operations include fp_add, fp_sub, fp_mul with correct scaling (FP16 uses div 256, FP32 uses div 65536, FP64 uses div 4294967296). BitSet operations (set, unset, test, count) are defined with correctness proofs. Vector operations include dot product with commutativity proof. Factorial and binomial coefficient properties are verified. Cache line size is defined as CACHE_LINE_SIZE = 128.

### src/verification/isabelle/TypesVerification.thy

TypesVerification.thy focuses on algebraic properties of fixed-point arithmetic. For FP16: addition commutativity, associativity, and zero identity. For FP32: addition and multiplication commutativity and associativity, and distributivity of multiplication over addition. For FP64: addition commutativity and associativity. Complex fixed-point arithmetic (two FP32 components) proves complex addition commutativity and associativity, and correct complex multiplication formula (ac-bd, ad+bc). Clamp is proven to respect bounds and be idempotent. Pascal's rule for binomial coefficients is verified.

### src/verification/isabelle/Memory.thy

Memory.thy formalizes memory management. It defines AllocatorType, MemoryBlock, AllocationResult, and MemoryState with allocated/free block lists. Arena allocation proves monotonic growth and bulk-free correctness. Pool allocation proves fixed-size blocks and no-overlap. Security zeroing proves that freed blocks contain all zeros. Alignment correctness proves returned addresses are multiples of the requested alignment.

### src/verification/isabelle/RSF.thy

RSF.thy formalizes the RSF layer. It defines RSFLayerConfig, RSFWeights (scale and translation weight/bias matrices), and the forward/inverse functions. The reversibility theorem proves forward composed with inverse equals identity using algebraic manipulation of the coupling layer equations. Xavier initialization is shown to maintain unit variance. Gradient computation correctness is proven by symbolic differentiation of the forward function.

### src/verification/isabelle/IO_Complete.thy

IO_Complete.thy formalizes the I/O subsystem. It models the DurableWriter as a sequence of operations (write-to-temp, fsync, rename) and proves crash safety: if a crash occurs at any point during the sequence, the original file remains intact (either the old file exists or the new file is complete, never a partial write). BufferedWriter flush correctness proves that after flush, all previously written data has reached the underlying file. The Wyhash function is modeled and proven to produce uniformly distributed outputs for uniformly distributed inputs.

---

## 19. Formal Verification Suite — TLA+ (7 files)

### src/verification/tla/TensorSpec.tla

TensorSpec.tla specifies the tensor allocation state machine in TLA+. Variables include allocated_tensors, free_list, total_memory, and reference_counts. The Init predicate sets all to empty/zero. Actions include Allocate (creating a tensor with refcount 1), Retain (incrementing refcount), Release (decrementing refcount and freeing when zero), and COWCopy (creating a deep copy when refcount > 1). The type invariant asserts all refcounts are positive. The safety invariant asserts no two tensors share overlapping memory unless they are COW views of the same data with refcount > 1. Temporal properties include liveness (any allocation request is eventually satisfied or rejected) and fairness (the allocator does not starve any requester).

### src/verification/tla/TensorComplete.tla

TensorComplete.tla extends the tensor specification with additional state variables for SIMD processing state, matrix multiplication intermediate results, and serialization buffers. It proves that SIMD processing produces the same results as scalar processing (equivalence invariant), that matrix multiplication intermediate memory is bounded by O(M*K + K*N), and that serialization followed by deserialization preserves tensor equality.

### src/verification/tla/MemorySpec.tla

MemorySpec.tla specifies memory allocator behavior. Variables track heap state, free lists per allocator, and allocation history. Actions model alloc, free, and coalesce operations. The no-leak invariant asserts that the sum of allocated memory plus free memory equals total capacity at all times. The no-double-free invariant asserts that free is only called on currently-allocated blocks. The alignment invariant asserts that all allocated addresses satisfy alignment constraints.

### src/verification/tla/IPC_Liveness.tla and IPC_Liveness.cfg

IPC_Liveness.tla specifies inter-process communication between the Python server and the Zig binary. Variables include message_queue, server_state, binary_state, and timeout_counter. Actions model SendMessage, ReceiveMessage, ProcessMessage, SendResponse, and Timeout. The liveness property asserts that every sent message eventually receives a response (under fairness assumptions). The cfg file configures the TLC model checker with specific constant values: MAX_QUEUE_SIZE = 10, TIMEOUT_LIMIT = 30, and the initial state specification.

### src/verification/tlaplus/DistributedTraining.tla

DistributedTraining.tla specifies the distributed training protocol. Variables include per-rank state (parameters, gradients, sync_status), the NCCL communicator state, and the global step counter. Actions model LocalForward, LocalBackward, AllReduceStart, AllReduceComplete, and ParameterUpdate. The consistency invariant asserts that after AllReduceComplete, all ranks have identical parameters. The progress property asserts that the global step counter increases monotonically. The deadlock-freedom property asserts that the system always has an enabled action.

### src/verification/tlaplus/TypesVerification.tla

TypesVerification.tla specifies fixed-point arithmetic as a state machine. Variables include fp16_value, fp32_value, and fp64_value. Operations model addition, subtraction, multiplication, and conversion. Invariants assert that values remain within their representable ranges, that addition commutes, and that conversion preserves value within rounding bounds.

---

## 20. Formal Verification Suite — Viper (4 files)

### src/verification/viper/Tensor.vpr

Tensor.vpr uses Viper's separation logic to verify tensor operations. The Tensor predicate asserts full permission to data, layout, device, and ownership fields, with constraints on data length matching shape size and non-negative values. The tensorCreate method allocates a new tensor with replicated zero data, ROW_MAJOR layout, CPU device, and OWNED ownership, with postconditions specifying the initial state. The tensorMap method transforms tensor data using a scaling factor, maintaining the Tensor predicate throughout.

### src/verification/viper/TensorMemory.vpr

TensorMemory.vpr verifies the interaction between tensors and memory management. It proves that tensor allocation always obtains memory from a valid allocator, that tensor deallocation returns memory to the correct allocator, and that reference counting correctly tracks shared ownership using permission fractions. The COW copy operation is verified to transfer exclusive write permission to the new copy while retaining read permission on the original.

### src/verification/viper/Memory.vpr

Memory.vpr formalizes memory allocator contracts in separation logic. Each allocator method has preconditions specifying required permissions and postconditions specifying granted permissions. Alloc requires no permissions and grants full permission to the returned block. Free requires full permission to the block and removes it from the permission set. The no-overlap property is encoded as a separating conjunction: allocated blocks are pairwise disjoint.

### src/verification/viper/TypesVerification.vpr

TypesVerification.vpr verifies type operations using separation logic. Fixed-point arithmetic is verified with pre/postconditions asserting overflow safety. The PRNG is verified to maintain its state invariant across calls. BitSet operations are verified to maintain the backing array's permission consistency.

---

## 21. Formal Verification Suite — SPIN (5 files)

### src/verification/spin/TensorModel.pml

TensorModel.pml models tensor operations as concurrent processes in Promela. Global variables represent tensor refcounts, data arrays, and COW flags. Processes model concurrent tensor users that non-deterministically retain, release, read, and write tensors. The COW protocol is modeled: before writing, a process checks the refcount and performs a copy if shared. LTL properties assert that refcounts never become negative, that data is never corrupted by concurrent writes (mutual exclusion via COW), and that all tensors are eventually freed (no memory leaks).

### src/verification/spin/TensorComplete.pml

TensorComplete.pml extends the tensor model with SIMD processing, matrix operations, and serialization state. It verifies that SIMD and scalar processing produce identical results by running both in parallel and checking equality after each operation. Matrix multiplication is modeled as a multi-step process with assertions on intermediate shapes. Serialization is modeled with assertions that deserialized state matches original state.

### src/verification/spin/MemoryModel.pml

MemoryModel.pml models the memory subsystem with concurrent allocator users. Processes perform interleaved alloc/free operations. The model checks for deadlocks (no state where all processes are blocked), for memory leaks (all allocated blocks are eventually freed), for double-free detection (freeing a free block is rejected), and for use-after-free detection (accessing a freed block is rejected). The SPIN model checker exhaustively explores all interleavings to verify these properties.

### src/verification/spin/GPUSync.pml

GPUSync.pml models GPU synchronization primitives. Processes represent multiple GPU ranks performing NCCL collective operations. The model verifies that allReduce produces consistent results across all ranks (all ranks end up with the same value), that barrier synchronization ensures no rank proceeds past the barrier until all have arrived, and that the system is deadlock-free (no configuration of ranks can permanently block). The NCCL communicator state is modeled with channels for message passing between rank processes.

### src/verification/spin/TypesVerification.pml

TypesVerification.pml models the type system's runtime behavior. Processes model concurrent fixed-point arithmetic operations and verify that results are deterministic regardless of execution order. The PRNG state machine is modeled and verified to produce a complete cycle before repeating.

---

## 22. Formal Verification Suite — Semantics (6 files)

### src/verification/semantics/TensorModel.agda, .lean, .thy, .tla, .pml, .vpr

The semantics directory contains six files, each implementing the same tensor model in a different verification language. These serve as reference implementations that establish a common semantic foundation across all six frameworks. Each file defines the same core operations (create, retain, release, add, multiply, reshape, transpose) with the same semantics, ensuring that the properties proven in one framework are consistent with those proven in the others. This cross-framework consistency is critical for the verification suite's integrity: if the tensor model means something different in Agda than it does in Lean 4, proofs in the two frameworks could be contradictory.

The Agda version uses dependent types and constructive proofs. The Lean 4 version uses Mathlib tactics and simp lemmas. The Isabelle/HOL version uses algebraic datatypes and sledgehammer-assisted proofs. The TLA+ version uses state machines and temporal logic. The Promela version uses processes and LTL properties. The Viper version uses separation logic predicates and permissions. Together, they establish that the tensor abstraction is consistently formalized across all six frameworks, enabling confident composition of verification results.

---

## 23. Model Binaries

The models/ directory (and its mirror in src/models/) contains four pre-trained model binary files:

rsf_trained.bin contains the trained RSF network weights, serialized using the model_io.zig binary format with MAGIC_HEADER "JAIDE40\x00", version number, JSON metadata, and SHA-256 integrity checksum. The weights include s_weight, t_weight, s_bias, and t_bias tensors for each RSF layer.

mgt_vocab.bin contains the trained MGT vocabulary, including token-to-ID mappings, BPE merge rules with priorities, morphological prefix/suffix/root lists, and anchor token definitions, all serialized in the binary vocab format.

optimizer_state.bin contains the SFD optimizer checkpoint, including first-moment and second-moment estimates for all parameters, the current learning rate, epoch counter, and global step count.

ranker_weights.bin contains the trained ranker parameters, including n-gram weights, LSH hash function parameters, and the random seed, serialized in the ranker binary format.

---

## 24. File Count Summary

| Category | Files | Approximate Lines |
|---|---|---|
| Zig Core | 5 | ~9,500 |
| Zig Core Relational | 25 | ~22,000 |
| Zig Processor/Tokenizer/Optimizer/Index/Ranker | 5 | ~7,500 |
| Zig Entry Points | 6 | ~4,500 |
| Zig API/Server | 2 | ~1,500 |
| Zig Distributed | 5 | ~5,000 |
| Zig HW/Accel | 4 | ~2,500 |
| Zig WASM | 2 | ~600 |
| Zig Tests | 1 | ~320 |
| Zig Fuzz | 3 | ~500 |
| Zig Build | 2 | ~250 |
| Futhark GPU Kernels | 2 | ~800 |
| ZK Circuit (Circom) | 1 | ~200 |
| FPGA Verilog | 1 | ~400 |
| Verification (Agda) | 21 | ~7,800 |
| Verification (Lean 4) | 17 | ~7,700 |
| Verification (Isabelle) | 8 | ~3,700 |
| Verification (TLA+) | 7 | ~1,400 |
| Verification (Viper) | 4 | ~2,400 |
| Verification (SPIN) | 5 | ~1,900 |
| Verification (Semantics) | 6 | ~1,300 |
| Python Scripts | 8 | ~2,500 |
| Shell Scripts | 8 | ~1,200 |
| ASIC/FPGA Scripts | 2 | ~300 |
| Config Files | 2 | ~50 |
| Top-Level Python | 3 | ~350 |
| Model Binaries | 4 | (binary) |
| **Total** | **~160+** | **~86,000+** |
