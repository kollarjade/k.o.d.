let matmul_tiled [m][n][k] (a: [m][k]f32) (b: [k][n]f32): [m][n]f32 =
  let tile_size = 16i64
  in map (\i ->
    map (\j ->
      let tiles = (k + tile_size - 1) / tile_size
      in reduce (+) 0f32
        (map (\t ->
          let base = t * tile_size
          let tile_k = i64.min tile_size (k - base)
          in reduce (+) 0f32
            (map (\kk ->
              a[i, base + kk] * b[base + kk, j]
            ) (iota tile_k))
        ) (iota tiles))
    ) (iota n)
  ) (iota m)

let batched_matmul [b][m][n][k] (a: [b][m][k]f32) (c: [b][k][n]f32): [b][m][n]f32 =
  map2 (\a_mat c_mat -> matmul_tiled a_mat c_mat) a c

let dot_product [n] (a: [n]f32) (b: [n]f32): f32 =
  reduce (+) 0f32 (map2 (*) a b)

let softmax [n] (x: [n]f32): [n]f32 =
  let max_val = reduce f32.max (-f32.inf) x
  let exp_x = map (\xi -> f32.exp (xi - max_val)) x
  let sum = reduce (+) 0f32 exp_x
  in map (/ sum) exp_x

let layer_norm [n] (x: [n]f32) (gamma: [n]f32) (beta: [n]f32) (eps: f32): [n]f32 =
  let mean = (reduce (+) 0f32 x) / f32.i64 n
  let variance = (reduce (+) 0f32 (map (\xi -> (xi - mean) * (xi - mean)) x)) / f32.i64 n
  let std_dev = f32.sqrt (variance + eps)
  in map3 (\xi g b -> g * ((xi - mean) / std_dev) + b) x gamma beta

let relu [n] (x: [n]f32): [n]f32 =
  map (\xi -> f32.max 0f32 xi) x

let gelu [n] (x: [n]f32): [n]f32 =
  let sqrt_2_over_pi = 0.7978845608f32
  in map (\xi ->
    let cdf = 0.5f32 * (1.0f32 + f32.tanh (sqrt_2_over_pi * (xi + 0.044715f32 * xi * xi * xi)))
    in xi * cdf
  ) x

let spectral_clip [n] (fisher: [n]f32) (clip_val: f32): [n]f32 =
  map (\f -> f32.max f clip_val) fisher

let batch_reduce [b][n] (gradients: [b][n]f32): [n]f32 =
  reduce_comm (\a b -> map2 (+) a b) (replicate n 0f32) gradients

let score_segments [n] (query_hash: u64) (segment_hashes: [n]u64) (base_scores: [n]f32): [n]f32 =
  map2 (\hash score ->
    let match_bonus = if hash == query_hash then 1.0f32 else 0.0f32
    in score + match_bonus
  ) segment_hashes base_scores

let topk [n] (k: i64) (scores: [n]f32) (indices: [n]i64): ([k]f32, [k]i64) =
  let sorted_pairs = zip scores indices
                      |> radix_sort_by_key (\p -> -p.0) f32.num_bits f32.get_bit
  let top = take k sorted_pairs
  in (map (.0) top, map (.1) top)

let rsf_scatter [n] (x: [n]f32) (indices: [n]i64): [n]f32 =
  let half = n / 2
  in map (\i ->
    if i < half then
      let j = indices[i] % half
      in x[j] + x[j + half]
    else
      let j = indices[i - half] % half
      in x[j] - x[j + half]
  ) (iota n)

let rsf_flow [n] (x: [n]f32) (s_weight: [n]f32) (t_weight: [n]f32) (s_bias: [n]f32) (t_bias: [n]f32): [n]f32 =
  let half = n / 2
  let x_s = map (\i -> x[i] * s_weight[i] + s_bias[i]) (iota half)
  let x_t = map (\i -> x[i + half] * t_weight[i] + t_bias[i]) (iota half)
  let combined = map2 (+) x_s x_t
  in scatter (replicate n 0f32) (iota n) (map (\i -> if i < half then combined[i] else combined[i - half]) (iota n))

let rsf_forward_layer [n] (x: [n]f32) (s_weight: [n]f32) (t_weight: [n]f32) (s_bias: [n]f32) (t_bias: [n]f32) (perm_indices: [n]i64): [n]f32 =
  let scattered = rsf_scatter x perm_indices
  in rsf_flow scattered s_weight t_weight s_bias t_bias

let rsf_backward_scatter [n] (grad: [n]f32) (indices: [n]i64): [n]f32 =
  let half = n / 2
  in map (\i ->
    if i < half then
      let j = indices[i] % half
      in grad[j] + grad[j + half]
    else
      let j = indices[i - half] % half
      in grad[j] - grad[j + half]
  ) (iota n)

let rsf_backward_flow [n] (grad_out: [n]f32) (x: [n]f32) (s_weight: [n]f32) (t_weight: [n]f32): ([n]f32, [n]f32, [n]f32, [n]f32, [n]f32) =
  let half = n / 2
  let grad_s_bias = map (\i -> grad_out[i]) (iota half)
  let grad_t_bias = map (\i -> grad_out[i + half]) (iota half)
  let grad_s_weight = map2 (\g xi -> g * xi) grad_s_bias (map (\i -> x[i]) (iota half))
  let grad_t_weight = map2 (\g xi -> g * xi) grad_t_bias (map (\i -> x[i + half]) (iota half))
  let grad_x_s = map2 (*) grad_s_bias s_weight
  let grad_x_t = map2 (*) grad_t_bias t_weight
  let grad_x = map (\i -> if i < half then grad_x_s[i] else grad_x_t[i - half]) (iota n)
  let grad_s_weight_full = map (\i -> if i < half then grad_s_weight[i] else 0f32) (iota n)
  let grad_t_weight_full = map (\i -> if i < half then grad_t_weight[i] else 0f32) (iota n)
  let grad_s_bias_full = map (\i -> if i < half then grad_s_bias[i] else 0f32) (iota n)
  let grad_t_bias_full = map (\i -> if i < half then grad_t_bias[i] else 0f32) (iota n)
  in (grad_x, grad_s_weight_full, grad_t_weight_full, grad_s_bias_full, grad_t_bias_full)

let rsf_backward_layer [n] (grad_out: [n]f32) (x: [n]f32) (s_weight: [n]f32) (t_weight: [n]f32) (perm_indices: [n]i64): ([n]f32, [n]f32, [n]f32, [n]f32, [n]f32) =
  let (grad_x_flow, grad_s_w, grad_t_w, grad_s_b, grad_t_b) = rsf_backward_flow grad_out x s_weight t_weight
  let grad_x = rsf_backward_scatter grad_x_flow perm_indices
  in (grad_x, grad_s_w, grad_t_w, grad_s_b, grad_t_b)

let hash_sequence [m] (tokens: [m]u32): u64 =
  let multiplier = 31u64
  in reduce (\h t -> h * multiplier + u64.u32 t) 0u64 tokens

let ssi_hash_insert [n] (hashes: [n]u64) (new_hash: u64): [n+1]u64 =
  let pos = reduce (+) 0i64 (map (\h -> if h < new_hash then 1i64 else 0i64) hashes)
  let left = hashes[0:pos]
  let right = hashes[pos:]
  in left ++ [new_hash] ++ right

let ssi_search [n][m] (tree_hashes: [n]u64) (query: [m]u32): i64 =
  let query_hash = hash_sequence query
  let distances = map (\h ->
    let diff = if h > query_hash then h - query_hash else query_hash - h
    in diff
  ) tree_hashes
  let pairs = zip (map f64.u64 distances) (iota n)
  let min_pair = reduce (\a b -> if a.0 < b.0 then a else b) (f64.inf, 0i64) pairs
  in min_pair.1

let ssi_retrieve_topk [n][m] (tree_hashes: [n]u64) (scores: [n]f32) (query: [m]u32) (k: i64): ([k]u64, [k]f32) =
  let query_hash = hash_sequence query
  let adjusted_scores = map2 (\h score ->
    let match_bonus = if h == query_hash then 10.0f32 else 0.0f32
    let proximity = 1.0f32 / (1.0f32 + f32.u64 (if h > query_hash then h - query_hash else query_hash - h))
    in score + match_bonus + proximity
  ) tree_hashes scores
  let sorted_indices = radix_sort_by_key (\i -> -adjusted_scores[i]) f32.num_bits f32.get_bit (iota n)
  let top_indices = take k sorted_indices
  let top_hashes = map (\i -> tree_hashes[i]) top_indices
  let top_scores = map (\i -> adjusted_scores[i]) top_indices
  in (top_hashes, top_scores)

let ssi_compute_similarity [m] (query: [m]u32) (candidate: [m]u32): f32 =
  let matches = reduce (+) 0i64 (map2 (\q c -> if q == c then 1i64 else 0i64) query candidate)
  let max_len = i64.max (length query) (length candidate)
  in f32.i64 matches / f32.i64 max_len

let ngram_hash [n] (tokens: [n]u32) (ngram_size: i64): []u64 =
  let num_ngrams = n - ngram_size + 1
  in map (\i ->
    let ngram = tokens[i:i+ngram_size]
    in hash_sequence ngram
  ) (iota num_ngrams)

let lsh_hash [n] (vec: [n]f32) (num_tables: i64) (seed: u64): [num_tables]u64 =
  map (\table_idx ->
    let table_seed = seed + u64.i64 table_idx
    let proj = reduce (+) 0f32 (map2 (\v i ->
      let pseudo_rand = f32.u64 ((table_seed + u64.i64 i) * 2654435761u64)
      in v * pseudo_rand
    ) vec (iota n))
    in if proj > 0f32 then 1u64 else 0u64
  ) (iota num_tables)

let fisher_diagonal_update [n] (fisher: [n]f32) (gradient: [n]f32) (decay: f32): [n]f32 =
  map2 (\f g -> decay * f + (1.0f32 - decay) * g * g) fisher gradient

let spectral_natural_gradient [n] (gradient: [n]f32) (fisher: [n]f32) (damping: f32): [n]f32 =
  map2 (\g f -> g / (f + damping)) gradient fisher

let attention [seq_len][d_model] (query: [seq_len][d_model]f32) (key: [seq_len][d_model]f32) (value: [seq_len][d_model]f32): [seq_len][d_model]f32 =
  let scores = map (\q -> map (\k -> dot_product q k) key) query
  let scaled_scores = map (\row -> map (/ f32.sqrt (f32.i64 d_model)) row) scores
  let attention_weights = map softmax scaled_scores
  in map (\weights -> reduce_comm (\acc pair -> map2 (+) acc (let (w, v) = pair in map (* w) v)) (replicate d_model 0f32) (zip weights value)) attention_weights

let conv1d [input_len][kernel_size] (input: [input_len]f32) (kernel: [kernel_size]f32): [input_len - kernel_size + 1]f32 =
  map (\i ->
    reduce (+) 0f32 (map2 (*) (input[i:i+kernel_size]) kernel)
  ) (iota (input_len - kernel_size + 1))

let maxpool1d [input_len] (input: [input_len]f32) (pool_size: i64): [input_len / pool_size]f32 =
  map (\i ->
    let pool_start = i * pool_size
    let pool_end = i64.min (pool_start + pool_size) input_len
    in reduce f32.max (-f32.inf) input[pool_start:pool_end]
  ) (iota (input_len / pool_size))

let elem_add [n] (a: [n]f32) (b: [n]f32): [n]f32 = map2 (+) a b
let elem_mul [n] (a: [n]f32) (b: [n]f32): [n]f32 = map2 (*) a b
let elem_div [n] (a: [n]f32) (b: [n]f32): [n]f32 = map2 (/) a b
let elem_sub [n] (a: [n]f32) (b: [n]f32): [n]f32 = map2 (-) a b

let scalar_add [n] (a: [n]f32) (s: f32): [n]f32 = map (+ s) a
let scalar_mul [n] (a: [n]f32) (s: f32): [n]f32 = map (* s) a
let scalar_div [n] (a: [n]f32) (s: f32): [n]f32 = map (/ s) a

let sum [n] (x: [n]f32): f32 = reduce (+) 0f32 x
let mean [n] (x: [n]f32): f32 = (reduce (+) 0f32 x) / f32.i64 n
let max [n] (x: [n]f32): f32 = reduce f32.max (-f32.inf) x
let min [n] (x: [n]f32): f32 = reduce f32.min f32.inf x

entry matmul [m][n][k] (a: [m][k]f32) (b: [k][n]f32): [m][n]f32 = matmul_tiled a b
entry batch_matmul [b][m][n][k] (a: [b][m][k]f32) (c: [b][k][n]f32): [b][m][n]f32 = batched_matmul a c
entry dot [n] (a: [n]f32) (b: [n]f32): f32 = dot_product a b

entry apply_softmax [n] (x: [n]f32): [n]f32 = softmax x
entry apply_layer_norm [n] (x: [n]f32) (gamma: [n]f32) (beta: [n]f32) (eps: f32): [n]f32 = layer_norm x gamma beta eps
entry apply_relu [n] (x: [n]f32): [n]f32 = relu x
entry apply_gelu [n] (x: [n]f32): [n]f32 = gelu x

entry clip_fisher [n] (fisher: [n]f32) (clip_val: f32): [n]f32 = spectral_clip fisher clip_val
entry reduce_gradients [b][n] (gradients: [b][n]f32): [n]f32 = batch_reduce gradients
entry update_fisher [n] (fisher: [n]f32) (grad: [n]f32) (decay: f32): [n]f32 = fisher_diagonal_update fisher grad decay
entry compute_natural_grad [n] (grad: [n]f32) (fisher: [n]f32) (damping: f32): [n]f32 = spectral_natural_gradient grad fisher damping

entry rank_segments [n] (query_hash: u64) (segment_hashes: [n]u64) (base_scores: [n]f32): [n]f32 = score_segments query_hash segment_hashes base_scores
entry select_topk [n] (k: i64) (scores: [n]f32): ([k]f32, [k]i64) = topk k scores (iota n)

entry rsf_forward [n] (x: [n]f32) (s_w: [n]f32) (t_w: [n]f32) (s_b: [n]f32) (t_b: [n]f32) (perm: [n]i64): [n]f32 = rsf_forward_layer x s_w t_w s_b t_b perm
entry rsf_backward [n] (grad: [n]f32) (x: [n]f32) (s_w: [n]f32) (t_w: [n]f32) (perm: [n]i64): ([n]f32, [n]f32, [n]f32, [n]f32, [n]f32) = rsf_backward_layer grad x s_w t_w perm

entry ssi_hash_tokens [m] (tokens: [m]u32): u64 = hash_sequence tokens
entry ssi_find_nearest [n][m] (tree: [n]u64) (query: [m]u32): i64 = ssi_search tree query
entry ssi_get_topk [n][m] (tree: [n]u64) (scores: [n]f32) (query: [m]u32) (k: i64): ([k]u64, [k]f32) = ssi_retrieve_topk tree scores query k
entry ssi_similarity [m] (query: [m]u32) (candidate: [m]u32): f32 = ssi_compute_similarity query candidate

entry compute_ngram_hashes [n] (tokens: [n]u32) (ngram_size: i64): []u64 = ngram_hash tokens ngram_size
entry compute_lsh [n] (vec: [n]f32) (num_tables: i64) (seed: u64): [num_tables]u64 = lsh_hash vec num_tables seed

entry compute_attention [seq_len][d_model] (query: [seq_len][d_model]f32) (key: [seq_len][d_model]f32) (value: [seq_len][d_model]f32): [seq_len][d_model]f32 = attention query key value

entry apply_conv1d [input_len][kernel_size] (input: [input_len]f32) (kernel: [kernel_size]f32): [input_len - kernel_size + 1]f32 = conv1d input kernel
entry apply_maxpool1d [input_len] (input: [input_len]f32) (pool_size: i64): [input_len / pool_size]f32 = maxpool1d input pool_size

entry add_arrays [n] (a: [n]f32) (b: [n]f32): [n]f32 = elem_add a b
entry mul_arrays [n] (a: [n]f32) (b: [n]f32): [n]f32 = elem_mul a b
entry div_arrays [n] (a: [n]f32) (b: [n]f32): [n]f32 = elem_div a b
entry sub_arrays [n] (a: [n]f32) (b: [n]f32): [n]f32 = elem_sub a b

entry add_scalar [n] (a: [n]f32) (s: f32): [n]f32 = scalar_add a s
entry mul_scalar [n] (a: [n]f32) (s: f32): [n]f32 = scalar_mul a s
entry div_scalar [n] (a: [n]f32) (s: f32): [n]f32 = scalar_div a s

entry array_sum [n] (x: [n]f32): f32 = sum x
entry array_mean [n] (x: [n]f32): f32 = mean x
entry array_max [n] (x: [n]f32): f32 = max x
entry array_min [n] (x: [n]f32): f32 = min x

type complex = {re: f32, im: f32}

let complex_add (a: complex) (b: complex): complex =
  {re = a.re + b.re, im = a.im + b.im}

let complex_sub (a: complex) (b: complex): complex =
  {re = a.re - b.re, im = a.im - b.im}

let complex_mul (a: complex) (b: complex): complex =
  {re = a.re * b.re - a.im * b.im, im = a.re * b.im + a.im * b.re}

let complex_conj (a: complex): complex =
  {re = a.re, im = -a.im}

let complex_abs (a: complex): f32 =
  f32.sqrt (a.re * a.re + a.im * a.im)

let complex_abs_sq (a: complex): f32 =
  a.re * a.re + a.im * a.im

let complex_scale (s: f32) (a: complex): complex =
  {re = s * a.re, im = s * a.im}

let complex_from_polar (r: f32) (theta: f32): complex =
  {re = r * f32.cos theta, im = r * f32.sin theta}

let complex_normalize (a: complex): complex =
  let mag = complex_abs a
  in if mag > 0f32 then complex_scale (1f32 / mag) a else a

let rgpu_edge_quality_to_weight (quality: i32): f32 =
  if quality == 0 then 0.25f32
  else if quality == 1 then 1.0f32
  else if quality == 2 then 0.75f32
  else if quality == 3 then 0.1f32
  else if quality == 4 then 0.5f32
  else 0.0f32

let rgpu_edge_quality_to_weight_batch [n] (qualities: [n]i32): [n]f32 =
  map rgpu_edge_quality_to_weight qualities

let rgpu_propagate_quality [n] (edge_sources: [n]i64) (edge_targets: [n]i64) (edge_qualities: [n]i32) (node_qualities: []i32) (iterations: i64): [n]i32 =
  let propagate_once qualities =
    map3 (\src tgt q ->
      let src_q = if src >= 0 && src < i64.i32 (length node_qualities) then node_qualities[src] else q
      let tgt_q = if tgt >= 0 && tgt < i64.i32 (length node_qualities) then node_qualities[tgt] else q
      let combined = i32.min src_q (i32.min tgt_q q)
      in if combined == 3 then 3 else i32.max q combined
    ) edge_sources edge_targets qualities
  in loop current_qualities = edge_qualities for _i < iterations do
    propagate_once current_qualities

let rgpu_compute_degree_sequence [num_nodes][num_edges] (edge_sources: [num_edges]i64) (edge_targets: [num_edges]i64): ([num_nodes]i64, [num_nodes]i64) =
  let out_degrees = 
    reduce_by_index (replicate num_nodes 0i64) (+) 0i64 edge_sources (replicate num_edges 1i64)
  let in_degrees = 
    reduce_by_index (replicate num_nodes 0i64) (+) 0i64 edge_targets (replicate num_edges 1i64)
  in (out_degrees, in_degrees)

let rgpu_canonical_form_signature [num_nodes][num_edges] (edge_sources: [num_edges]i64) (edge_targets: [num_edges]i64) (edge_qualities: [num_edges]i32): u64 =
  let (out_degrees, in_degrees) = rgpu_compute_degree_sequence edge_sources edge_targets
  let degree_hash = reduce (\h d -> h * 31u64 + u64.i64 d) 0u64 out_degrees
  let in_degree_hash = reduce (\h d -> h * 37u64 + u64.i64 d) 0u64 in_degrees
  let quality_hash = reduce (\h q -> h * 41u64 + u64.i32 q) 0u64 edge_qualities
  let node_count_hash = u64.i64 num_nodes * 1000003u64
  let edge_count_hash = u64.i64 num_edges * 999983u64
  in degree_hash ^ in_degree_hash ^ quality_hash ^ node_count_hash ^ edge_count_hash

let rgpu_compute_fractal_dimension [num_nodes][num_edges] (node_hashes: [num_nodes]u64) (edge_sources: [num_edges]i64) (edge_targets: [num_edges]i64): f32 =
  let box_sizes = [1i64, 2i64, 4i64, 8i64, 16i64]
  let box_counts = map (\box_size ->
    let boxes = map (\h -> h % u64.i64 box_size) node_hashes
    let sorted_boxes = radix_sort_by_key (\b -> b) 64 u64.get_bit boxes
    let unique_mask = map2 (\i sb -> if i == 0 || sb != sorted_boxes[i-1] then 1i64 else 0i64) (iota num_nodes) sorted_boxes
    in reduce (+) 0i64 unique_mask
  ) box_sizes
  let valid_indices = filter (\i -> box_counts[i] > 0) (iota 5)
  let valid_counts = map (\i -> box_counts[i]) valid_indices
  let valid_sizes = map (\i -> box_sizes[i]) valid_indices
  let n_valid = length valid_counts
  in if n_valid < 2 then 1.0f32 else
    let log_sizes = map (\s -> f32.log (f32.i64 s)) valid_sizes
    let log_counts = map (\c -> f32.log (f32.i64 c)) valid_counts
    let sum_x = reduce (+) 0f32 log_sizes
    let sum_y = reduce (+) 0f32 log_counts
    let sum_xy = reduce (+) 0f32 (map2 (*) log_sizes log_counts)
    let sum_x2 = reduce (+) 0f32 (map (\x -> x * x) log_sizes)
    let n = f32.i64 n_valid
    let denominator = n * sum_x2 - sum_x * sum_x
    in if f32.abs denominator < 1e-10f32 then 1.0f32 else
      let slope = (n * sum_xy - sum_x * sum_y) / denominator
      in f32.abs slope

let rgpu_update_edge_weights [n] (current_weights: [n]f32) (feedback: [n]f32) (learning_rate: f32): [n]f32 =
  map2 (\w f ->
    let delta = learning_rate * f
    let new_weight = w + delta
    in f32.max 0.0f32 (f32.min 1.0f32 new_weight)
  ) current_weights feedback

let rgpu_adaptive_weight [n] (base_weights: [n]f32) (temporal_factors: [n]f32) (spatial_factors: [n]f32) (semantic_factors: [n]f32): [n]f32 =
  map4 (\base temp spat sem ->
    let adaptive = base * temp * spat * sem
    in f32.max 0.0f32 (f32.min 1.0f32 adaptive)
  ) base_weights temporal_factors spatial_factors semantic_factors

let rgpu_propagate_weights [num_nodes][num_edges] (edge_weights: [num_edges]f32) (edge_sources: [num_edges]i64) (edge_targets: [num_edges]i64) (source_node: i64) (iterations: i64) (decay: f32): [num_edges]f32 =
  let propagate_once (weights: [num_edges]f32) (visited: [num_nodes]bool) (iteration: i64): ([num_edges]f32, [num_nodes]bool) =
    let decay_factor = f32.pow decay (f32.i64 iteration)
    let new_weights = map3 (\w src tgt ->
      let src_visited = if src >= 0 && src < num_nodes then visited[src] else false
      let tgt_visited = if tgt >= 0 && tgt < num_nodes then visited[tgt] else false
      in if src_visited || tgt_visited then w * decay_factor else w
    ) weights edge_sources edge_targets
    let newly_visited = map2 (\src tgt ->
      let src_v = if src >= 0 && src < num_nodes then visited[src] else false
      let tgt_v = if tgt >= 0 && tgt < num_nodes then visited[tgt] else false
      in src_v || tgt_v
    ) edge_sources edge_targets
    let new_visited_nodes = 
      reduce_by_index (copy visited) (||) false 
        (edge_targets ++ edge_sources) 
        (newly_visited ++ newly_visited)
    in (new_weights, new_visited_nodes)
  let initial_visited = map (\i -> i == source_node) (iota num_nodes)
  let (final_weights, _) = loop (weights, visited) = (edge_weights, initial_visited) for i < iterations do
    propagate_once weights visited i
  in final_weights

let rgpu_xy_route (src_x: i64) (src_y: i64) (dst_x: i64) (dst_y: i64) (grid_width: i64): (i64, []i64) =
  let dx = i64.abs (dst_x - src_x)
  let dy = i64.abs (dst_y - src_y)
  let total_hops = dx + dy
  let path_x_dir = if dst_x > src_x then 1i64 else -1i64
  let path_y_dir = if dst_y > src_y then 1i64 else -1i64
  let x_steps = map (\i -> 
    let current_x = src_x + (i + 1) * path_x_dir
    in src_y * grid_width + current_x
  ) (iota dx)
  let y_steps = map (\i ->
    let current_y = src_y + (i + 1) * path_y_dir
    in current_y * grid_width + dst_x
  ) (iota dy)
  in (total_hops, x_steps ++ y_steps)

let rgpu_route_cost (src_x: i64) (src_y: i64) (dst_x: i64) (dst_y: i64) (hop_cost: f32) (congestion_factor: f32): f32 =
  let dx = f32.i64 (i64.abs (dst_x - src_x))
  let dy = f32.i64 (i64.abs (dst_y - src_y))
  let base_cost = (dx + dy) * hop_cost
  let manhattan_distance = dx + dy
  let congestion_penalty = congestion_factor * manhattan_distance * manhattan_distance
  in base_cost + congestion_penalty

let rgpu_route_cost_batch [n] (src_xs: [n]i64) (src_ys: [n]i64) (dst_xs: [n]i64) (dst_ys: [n]i64) (hop_cost: f32) (congestion_factor: f32): [n]f32 =
  map4 (\sx sy dx dy -> rgpu_route_cost sx sy dx dy hop_cost congestion_factor) src_xs src_ys dst_xs dst_ys

let rgpu_balance_load [num_cores] (core_loads: [num_cores]f32): [num_cores]f32 =
  let total_load = reduce (+) 0f32 core_loads
  let avg_load = total_load / f32.i64 num_cores
  let max_deviation = 0.1f32 * avg_load
  in map (\load ->
    if load > avg_load + max_deviation then avg_load + max_deviation
    else if load < avg_load - max_deviation then avg_load - max_deviation
    else load
  ) core_loads

let rgpu_compute_core_utilization [num_cores] (cycles_active: [num_cores]i64) (cycles_idle: [num_cores]i64): [num_cores]f32 =
  map2 (\active idle ->
    let total = active + idle
    in if total > 0 then f32.i64 active / f32.i64 total else 0.0f32
  ) cycles_active cycles_idle

let rgpu_should_gate_core (utilization: f32) (low_threshold: f32) (current_power: f32) (power_budget: f32): bool =
  utilization < low_threshold && current_power > power_budget * 0.5f32

let rgpu_should_gate_core_batch [n] (utilizations: [n]f32) (low_threshold: f32) (current_power: f32) (power_budget: f32): [n]bool =
  map (\u -> rgpu_should_gate_core u low_threshold current_power power_budget) utilizations

let rgpu_power_budget_check [num_cores] (core_powers: [num_cores]f32) (power_budget: f32): (bool, f32, f32) =
  let total_power = reduce (+) 0f32 core_powers
  let headroom = power_budget - total_power
  let within_budget = total_power <= power_budget
  in (within_budget, total_power, headroom)

let rgpu_sparsity_mask [n] (workloads: [n]f32) (threshold: f32): [n]bool =
  map (\w -> w >= threshold) workloads

let rgpu_energy_savings [n] (workloads: [n]f32) (threshold: f32) (idle_power: f32) (active_power: f32): f32 =
  let mask = rgpu_sparsity_mask workloads threshold
  let inactive_count = reduce (+) 0i64 (map (\m -> if m then 0i64 else 1i64) mask)
  let savings_per_core = active_power - idle_power
  in f32.i64 inactive_count * savings_per_core

let rgpu_compute_sparsity_ratio [n] (workloads: [n]f32) (threshold: f32): f32 =
  let mask = rgpu_sparsity_mask workloads threshold
  let inactive_count = reduce (+) 0i64 (map (\m -> if m then 0i64 else 1i64) mask)
  in f32.i64 inactive_count / f32.i64 n

let rgpu_quantum_correlation (state1: complex) (state2: complex): complex =
  complex_mul state1 (complex_conj state2)

let rgpu_quantum_correlation_batch [n] (states1: [n]complex) (states2: [n]complex): [n]complex =
  map2 rgpu_quantum_correlation states1 states2

let rgpu_entangle_states (state1: complex) (state2: complex): complex =
  let sum_state = complex_add state1 state2
  let sqrt_2_inv = 1.0f32 / f32.sqrt 2.0f32
  in complex_scale sqrt_2_inv sum_state

let rgpu_entangle_states_batch [n] (states1: [n]complex) (states2: [n]complex): [n]complex =
  map2 rgpu_entangle_states states1 states2

let rgpu_measure_probability (state: complex): f32 =
  complex_abs_sq state

let rgpu_measure_probability_batch [n] (states: [n]complex): [n]f32 =
  map rgpu_measure_probability states

let rgpu_hadamard_transform (state: complex): complex =
  let one = {re = 1.0f32, im = 0.0f32}
  let sum_state = complex_add state one
  let sqrt_2_inv = 1.0f32 / f32.sqrt 2.0f32
  let transformed = complex_scale sqrt_2_inv sum_state
  in complex_normalize transformed

let rgpu_hadamard_transform_batch [n] (states: [n]complex): [n]complex =
  map rgpu_hadamard_transform states

let rgpu_phase_shift (state: complex) (theta: f32): complex =
  let rotation = complex_from_polar 1.0f32 theta
  in complex_mul state rotation

let rgpu_phase_shift_batch [n] (states: [n]complex) (thetas: [n]f32): [n]complex =
  map2 rgpu_phase_shift states thetas

let rgpu_pauli_x (state: complex): complex =
  {re = state.im, im = state.re}

let rgpu_pauli_y (state: complex): complex =
  {re = -state.im, im = state.re}

let rgpu_pauli_z (state: complex): complex =
  let pi_rotation = complex_from_polar 1.0f32 f32.pi
  in complex_mul state pi_rotation

let rgpu_cnot (control: complex) (target: complex): (complex, complex) =
  let should_flip = complex_abs control > 0.5f32
  let new_target = if should_flip then {re = target.im, im = target.re} else target
  in (control, new_target)

let rgpu_fractal_transform (state: complex) (depth: i64): complex =
  let transform_step (s: complex) (i: i64): complex =
    let scale_factor = 1.0f32 / f32.pow 2.0f32 (f32.i64 i)
    let phase = f32.atan2 s.im s.re
    let rotation = complex_from_polar scale_factor (phase * scale_factor)
    let sum_state = complex_add s rotation
    let norm_factor = 1.0f32 / f32.sqrt (1.0f32 + scale_factor * scale_factor)
    in complex_scale norm_factor sum_state
  in loop current = state for i < depth do transform_step current i

let rgpu_relational_and (state1: complex) (state2: complex): complex =
  complex_normalize (complex_mul state1 state2)

let rgpu_relational_or (state1: complex) (state2: complex): complex =
  let sqrt_2_inv = 1.0f32 / f32.sqrt 2.0f32
  let sum_state = complex_add state1 state2
  in complex_normalize (complex_scale sqrt_2_inv sum_state)

let rgpu_relational_xor (state1: complex) (state2: complex): complex =
  let sqrt_2_inv = 1.0f32 / f32.sqrt 2.0f32
  let diff_state = complex_sub state1 state2
  in complex_normalize (complex_scale sqrt_2_inv diff_state)

let rgpu_partition_nodes [num_nodes] (num_cores: i64): [num_nodes]i64 =
  let nodes_per_core = num_nodes / num_cores
  let remainder = num_nodes % num_cores
  in map (\i ->
    if i < remainder * (nodes_per_core + 1) then
      i / (nodes_per_core + 1)
    else
      remainder + (i - remainder * (nodes_per_core + 1)) / nodes_per_core
  ) (iota num_nodes)

let rgpu_compute_partition_boundaries [num_nodes] (num_cores: i64): [num_cores + 1]i64 =
  let nodes_per_core = num_nodes / num_cores
  let remainder = num_nodes % num_cores
  in map (\i ->
    if i == 0 then 0i64
    else if i <= remainder then i * (nodes_per_core + 1)
    else remainder * (nodes_per_core + 1) + (i - remainder) * nodes_per_core
  ) (iota (num_cores + 1))

let rgpu_distribute_edges [num_edges] (edge_sources: [num_edges]i64) (edge_targets: [num_edges]i64) (node_partitions: []i64): [num_edges]i64 =
  map2 (\src tgt ->
    let src_partition = if src >= 0 && src < i64.i32 (length node_partitions) then node_partitions[src] else 0i64
    let tgt_partition = if tgt >= 0 && tgt < i64.i32 (length node_partitions) then node_partitions[tgt] else 0i64
    in i64.min src_partition tgt_partition
  ) edge_sources edge_targets

let rgpu_count_cross_partition_edges [num_edges] (edge_sources: [num_edges]i64) (edge_targets: [num_edges]i64) (node_partitions: []i64): i64 =
  reduce (+) 0i64 (map2 (\src tgt ->
    let src_p = if src >= 0 && src < i64.i32 (length node_partitions) then node_partitions[src] else -1i64
    let tgt_p = if tgt >= 0 && tgt < i64.i32 (length node_partitions) then node_partitions[tgt] else -1i64
    in if src_p != tgt_p then 1i64 else 0i64
  ) edge_sources edge_targets)

let rgpu_compute_partition_load [num_edges] (edge_sources: [num_edges]i64) (edge_targets: [num_edges]i64) (node_partitions: []i64) (num_cores: i64): [num_cores]i64 =
  let edge_partitions = rgpu_distribute_edges edge_sources edge_targets node_partitions
  in reduce_by_index (replicate num_cores 0i64) (+) 0i64 edge_partitions (replicate num_edges 1i64)

let rgpu_noc_neighbors (core_id: i64) (grid_width: i64) (grid_height: i64): [4]i64 =
  let x = core_id % grid_width
  let y = core_id / grid_width
  let left = if x > 0 then core_id - 1 else -1i64
  let right = if x < grid_width - 1 then core_id + 1 else -1i64
  let up = if y > 0 then core_id - grid_width else -1i64
  let down = if y < grid_height - 1 then core_id + grid_width else -1i64
  in [left, right, up, down]

let rgpu_compute_core_position (core_id: i64) (grid_width: i64): (i64, i64) =
  (core_id % grid_width, core_id / grid_width)

let rgpu_core_id_from_position (x: i64) (y: i64) (grid_width: i64): i64 =
  y * grid_width + x

let rgpu_manhattan_distance (src_core: i64) (dst_core: i64) (grid_width: i64): i64 =
  let (src_x, src_y) = rgpu_compute_core_position src_core grid_width
  let (dst_x, dst_y) = rgpu_compute_core_position dst_core grid_width
  in i64.abs (dst_x - src_x) + i64.abs (dst_y - src_y)

let rgpu_message_latency (src_core: i64) (dst_core: i64) (grid_width: i64) (hop_latency: f32) (base_latency: f32): f32 =
  let hops = rgpu_manhattan_distance src_core dst_core grid_width
  in base_latency + f32.i64 hops * hop_latency

entry rgpu_quality_to_weight (quality: i32): f32 = rgpu_edge_quality_to_weight quality
entry rgpu_quality_to_weight_batch [n] (qualities: [n]i32): [n]f32 = rgpu_edge_quality_to_weight_batch qualities

entry rgpu_propagate_edge_quality [n] (edge_sources: [n]i64) (edge_targets: [n]i64) (edge_qualities: [n]i32) (node_qualities: []i32) (iterations: i64): [n]i32 =
  rgpu_propagate_quality edge_sources edge_targets edge_qualities node_qualities iterations

entry rgpu_degree_sequence [num_nodes][num_edges] (edge_sources: [num_edges]i64) (edge_targets: [num_edges]i64): ([num_nodes]i64, [num_nodes]i64) =
  rgpu_compute_degree_sequence edge_sources edge_targets

entry rgpu_canonical_signature [num_nodes][num_edges] (edge_sources: [num_edges]i64) (edge_targets: [num_edges]i64) (edge_qualities: [num_edges]i32): u64 =
  rgpu_canonical_form_signature edge_sources edge_targets edge_qualities

entry rgpu_fractal_dim [num_nodes][num_edges] (node_hashes: [num_nodes]u64) (edge_sources: [num_edges]i64) (edge_targets: [num_edges]i64): f32 =
  rgpu_compute_fractal_dimension node_hashes edge_sources edge_targets

entry rgpu_update_weights [n] (weights: [n]f32) (feedback: [n]f32) (learning_rate: f32): [n]f32 =
  rgpu_update_edge_weights weights feedback learning_rate

entry rgpu_adaptive_weights [n] (base_weights: [n]f32) (temporal: [n]f32) (spatial: [n]f32) (semantic: [n]f32): [n]f32 =
  rgpu_adaptive_weight base_weights temporal spatial semantic

entry rgpu_propagate_edge_weights [num_nodes][num_edges] (edge_weights: [num_edges]f32) (edge_sources: [num_edges]i64) (edge_targets: [num_edges]i64) (source_node: i64) (iterations: i64) (decay: f32): [num_edges]f32 =
  rgpu_propagate_weights edge_weights edge_sources edge_targets source_node iterations decay

entry rgpu_compute_xy_route (src_x: i64) (src_y: i64) (dst_x: i64) (dst_y: i64) (grid_width: i64): (i64, []i64) =
  rgpu_xy_route src_x src_y dst_x dst_y grid_width

entry rgpu_compute_route_cost (src_x: i64) (src_y: i64) (dst_x: i64) (dst_y: i64) (hop_cost: f32) (congestion: f32): f32 =
  rgpu_route_cost src_x src_y dst_x dst_y hop_cost congestion

entry rgpu_compute_route_cost_batch [n] (src_xs: [n]i64) (src_ys: [n]i64) (dst_xs: [n]i64) (dst_ys: [n]i64) (hop_cost: f32) (congestion: f32): [n]f32 =
  rgpu_route_cost_batch src_xs src_ys dst_xs dst_ys hop_cost congestion

entry rgpu_load_balance [num_cores] (loads: [num_cores]f32): [num_cores]f32 =
  rgpu_balance_load loads

entry rgpu_core_utilization [n] (active: [n]i64) (idle: [n]i64): [n]f32 =
  rgpu_compute_core_utilization active idle

entry rgpu_gate_core_check (utilization: f32) (threshold: f32) (current_power: f32) (budget: f32): bool =
  rgpu_should_gate_core utilization threshold current_power budget

entry rgpu_gate_core_check_batch [n] (utilizations: [n]f32) (threshold: f32) (current_power: f32) (budget: f32): [n]bool =
  rgpu_should_gate_core_batch utilizations threshold current_power budget

entry rgpu_power_check [n] (core_powers: [n]f32) (budget: f32): (bool, f32, f32) =
  rgpu_power_budget_check core_powers budget

entry rgpu_sparsity [n] (workloads: [n]f32) (threshold: f32): [n]bool =
  rgpu_sparsity_mask workloads threshold

entry rgpu_compute_energy_savings [n] (workloads: [n]f32) (threshold: f32) (idle_power: f32) (active_power: f32): f32 =
  rgpu_energy_savings workloads threshold idle_power active_power

entry rgpu_sparsity_ratio [n] (workloads: [n]f32) (threshold: f32): f32 =
  rgpu_compute_sparsity_ratio workloads threshold

entry rgpu_quantum_corr (re1: f32) (im1: f32) (re2: f32) (im2: f32): (f32, f32) =
  let result = rgpu_quantum_correlation {re=re1, im=im1} {re=re2, im=im2}
  in (result.re, result.im)

entry rgpu_quantum_corr_batch [n] (re1: [n]f32) (im1: [n]f32) (re2: [n]f32) (im2: [n]f32): ([n]f32, [n]f32) =
  let states1 = map2 (\r i -> {re=r, im=i}) re1 im1
  let states2 = map2 (\r i -> {re=r, im=i}) re2 im2
  let results = rgpu_quantum_correlation_batch states1 states2
  in (map (.re) results, map (.im) results)

entry rgpu_entangle (re1: f32) (im1: f32) (re2: f32) (im2: f32): (f32, f32) =
  let result = rgpu_entangle_states {re=re1, im=im1} {re=re2, im=im2}
  in (result.re, result.im)

entry rgpu_entangle_batch [n] (re1: [n]f32) (im1: [n]f32) (re2: [n]f32) (im2: [n]f32): ([n]f32, [n]f32) =
  let states1 = map2 (\r i -> {re=r, im=i}) re1 im1
  let states2 = map2 (\r i -> {re=r, im=i}) re2 im2
  let results = rgpu_entangle_states_batch states1 states2
  in (map (.re) results, map (.im) results)

entry rgpu_measure_prob (re: f32) (im: f32): f32 =
  rgpu_measure_probability {re=re, im=im}

entry rgpu_measure_prob_batch [n] (re: [n]f32) (im: [n]f32): [n]f32 =
  rgpu_measure_probability_batch (map2 (\r i -> {re=r, im=i}) re im)

entry rgpu_hadamard (re: f32) (im: f32): (f32, f32) =
  let result = rgpu_hadamard_transform {re=re, im=im}
  in (result.re, result.im)

entry rgpu_hadamard_batch [n] (re: [n]f32) (im: [n]f32): ([n]f32, [n]f32) =
  let results = rgpu_hadamard_transform_batch (map2 (\r i -> {re=r, im=i}) re im)
  in (map (.re) results, map (.im) results)

entry rgpu_phase (re: f32) (im: f32) (theta: f32): (f32, f32) =
  let result = rgpu_phase_shift {re=re, im=im} theta
  in (result.re, result.im)

entry rgpu_phase_batch [n] (re: [n]f32) (im: [n]f32) (thetas: [n]f32): ([n]f32, [n]f32) =
  let results = rgpu_phase_shift_batch (map2 (\r i -> {re=r, im=i}) re im) thetas
  in (map (.re) results, map (.im) results)

entry rgpu_partition [num_nodes] (num_cores: i64): [num_nodes]i64 =
  rgpu_partition_nodes num_cores

entry rgpu_partition_bounds [num_nodes] (num_cores: i64): [num_cores + 1]i64 =
  rgpu_compute_partition_boundaries num_cores

entry rgpu_edge_distribution [num_edges] (edge_sources: [num_edges]i64) (edge_targets: [num_edges]i64) (node_partitions: []i64): [num_edges]i64 =
  rgpu_distribute_edges edge_sources edge_targets node_partitions

entry rgpu_cross_partition_edges [num_edges] (edge_sources: [num_edges]i64) (edge_targets: [num_edges]i64) (node_partitions: []i64): i64 =
  rgpu_count_cross_partition_edges edge_sources edge_targets node_partitions

entry rgpu_partition_loads [num_edges] (edge_sources: [num_edges]i64) (edge_targets: [num_edges]i64) (node_partitions: []i64) (num_cores: i64): [num_cores]i64 =
  rgpu_compute_partition_load edge_sources edge_targets node_partitions num_cores

entry rgpu_core_neighbors (core_id: i64) (grid_width: i64) (grid_height: i64): [4]i64 =
  rgpu_noc_neighbors core_id grid_width grid_height

entry rgpu_core_position (core_id: i64) (grid_width: i64): (i64, i64) =
  rgpu_compute_core_position core_id grid_width

entry rgpu_core_from_pos (x: i64) (y: i64) (grid_width: i64): i64 =
  rgpu_core_id_from_position x y grid_width

entry rgpu_distance (src: i64) (dst: i64) (grid_width: i64): i64 =
  rgpu_manhattan_distance src dst grid_width

entry rgpu_latency (src: i64) (dst: i64) (grid_width: i64) (hop_lat: f32) (base_lat: f32): f32 =
  rgpu_message_latency src dst grid_width hop_lat base_lat

entry rgpu_fractal_xform (re: f32) (im: f32) (depth: i64): (f32, f32) =
  let result = rgpu_fractal_transform {re=re, im=im} depth
  in (result.re, result.im)

entry rgpu_rel_and (re1: f32) (im1: f32) (re2: f32) (im2: f32): (f32, f32) =
  let result = rgpu_relational_and {re=re1, im=im1} {re=re2, im=im2}
  in (result.re, result.im)

entry rgpu_rel_or (re1: f32) (im1: f32) (re2: f32) (im2: f32): (f32, f32) =
  let result = rgpu_relational_or {re=re1, im=im1} {re=re2, im=im2}
  in (result.re, result.im)

entry rgpu_rel_xor (re1: f32) (im1: f32) (re2: f32) (im2: f32): (f32, f32) =
  let result = rgpu_relational_xor {re=re1, im=im1} {re=re2, im=im2}
  in (result.re, result.im)