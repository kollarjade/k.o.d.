type f16 = f16

entry rsf_spectral_transform [n][d] (input: [n][d]f16) (weights_s: [d][d]f16) : *[n][d]f16 =
  map (\row ->
    map (\j ->
      f16.sum (map2 (\x w -> x f16.* w) row weights_s[j])
    ) (iota d)
  ) input

entry rsf_temporal_transform [n][d] (input: [n][d]f16) (weights_t: [d][d]f16) : *[n][d]f16 =
  map (\row ->
    map (\j ->
      f16.sum (map2 (\x w -> x f16.* w) row weights_t[j])
    ) (iota d)
  ) input

entry relu [n][d] (input: [n][d]f16) : *[n][d]f16 =
  map (map (\x -> f16.max x (f16.i32 0))) input

entry layer_norm [n][d] (input: [n][d]f16) : *[n][d]f16 =
  let epsilon = f16.f32 1e-5
  in map (\row ->
    let mean = (f16.sum row) f16./ (f16.i64 d)
    let variance = (f16.sum (map (\x -> (x f16.- mean) f16.* (x f16.- mean)) row)) f16./ (f16.i64 d)
    let std_dev = f16.sqrt (variance f16.+ epsilon)
    in map (\x -> (x f16.- mean) f16./ std_dev) row
  ) input

entry rsf_forward [n][d] (input: [n][d]f16) (weights_s: [d][d]f16) (weights_t: [d][d]f16) : *[n][d]f16 =
  let after_spectral = rsf_spectral_transform input weights_s
  let after_relu1 = relu after_spectral
  let after_temporal = rsf_temporal_transform after_relu1 weights_t
  let after_relu2 = relu after_temporal
  in layer_norm after_relu2

entry rsf_grad_spectral [n][d] (input: [n][d]f16) (grad_output: [n][d]f16) : *[d][d]f16 =
  let grad_weights = replicate d (replicate d (f16.i32 0))
  in loop grad_weights for i < n do
       loop grad_weights for j < d do
         loop grad_weights for k < d do
           let update = input[i,k] f16.* grad_output[i,j]
           let grad_weights[j,k] = grad_weights[j,k] f16.+ update
           in grad_weights

entry rsf_grad_temporal [n][d] (input: [n][d]f16) (grad_output: [n][d]f16) : *[d][d]f16 =
  let grad_weights = replicate d (replicate d (f16.i32 0))
  in loop grad_weights for i < n do
       loop grad_weights for j < d do
         loop grad_weights for k < d do
           let update = input[i,k] f16.* grad_output[i,j]
           let grad_weights[j,k] = grad_weights[j,k] f16.+ update
           in grad_weights

entry sfd_update [d] (weights: *[d][d]f16) (gradients: [d][d]f16) (learning_rate: f16) (momentum: f16) (velocity: *[d][d]f16) : (*[d][d]f16, *[d][d]f16) =
  let new_velocity = map2 (map2 (\v g -> momentum f16.* v f16.+ learning_rate f16.* g)) velocity gradients
  let new_weights = map2 (map2 (\w v -> w f16.- v)) weights new_velocity
  in (new_weights, new_velocity)

entry compute_loss [n][d] (output: [n][d]f16) (target: [n][d]f16) : f16 =
  let squared_diff = map2 (map2 (\o t -> (o f16.- t) f16.* (o f16.- t))) output target
  let total = f16.sum (flatten squared_diff)
  let count = f16.i64 (n * d)
  in total f16./ count

entry accumulate_gradients [d] (grad1: *[d][d]f16) (grad2: [d][d]f16) : *[d][d]f16 =
  map2 (map2 (f16.+)) grad1 grad2

entry batch_forward [batch_size][seq_len][d] (inputs: [batch_size][seq_len][d]f16) (weights_s: [d][d]f16) (weights_t: [d][d]f16) : *[batch_size][seq_len][d]f16 =
  map (\sample -> rsf_forward sample weights_s weights_t) inputs

entry batch_gradients [batch_size][seq_len][d] (inputs: [batch_size][seq_len][d]f16) (grad_outputs: [batch_size][seq_len][d]f16) : (*[d][d]f16, *[d][d]f16) =
  let grad_s_list = map2 rsf_grad_spectral inputs grad_outputs
  let grad_t_list = map2 rsf_grad_temporal inputs grad_outputs
  let grad_s_total = reduce (map2 (map2 (f16.+))) (replicate d (replicate d (f16.i32 0))) grad_s_list
  let grad_t_total = reduce (map2 (map2 (f16.+))) (replicate d (replicate d (f16.i32 0))) grad_t_list
  in (grad_s_total, grad_t_total)

entry xavier_fill_inplace [d] (weights: *[d][d]f16) (seed: i32) : *[d][d]f16 =
  let scale = f16.sqrt (f16.f32 2.0 f16./ f16.i64 d)
  in map (\i ->
    map (\j ->
      let hash = (seed + i32.i64 i * 73856093 + i32.i64 j * 19349663) % 1000000
      let normalized = (f16.i32 hash) f16./ (f16.i32 1000000) f16.- f16.f32 0.5
      in normalized f16.* scale
    ) (iota d)
  ) (iota d)

entry batch_compute_loss [batch_size][seq_len][d] (outputs: [batch_size][seq_len][d]f16) (targets: [batch_size][seq_len][d]f16) : f16 =
  let squared_diff = map2 (map2 (map2 (\o t -> (o f16.- t) f16.* (o f16.- t)))) outputs targets
  let total = f16.sum (flatten (flatten squared_diff))
  let count = f16.i64 (batch_size * seq_len * d)
  in total f16./ count

entry scale_weights_inplace [d] (weights: *[d][d]f16) (scale_factor: f16) : *[d][d]f16 =
  map (map (\w -> w f16./ scale_factor)) weights

entry training_step [batch_size][seq_len][d] 
  (inputs: [batch_size][seq_len][d]f16)
  (targets: [batch_size][seq_len][d]f16)
  (weights_s: *[d][d]f16)
  (weights_t: *[d][d]f16)
  (velocity_s: *[d][d]f16)
  (velocity_t: *[d][d]f16)
  (learning_rate: f16)
  (momentum: f16) : (*[d][d]f16, *[d][d]f16, *[d][d]f16, *[d][d]f16, f16) =

  let outputs = batch_forward inputs weights_s weights_t
  let loss = batch_compute_loss outputs targets
  let grad_outputs = map2 (map2 (map2 (\o t -> (f16.f32 2.0) f16.* (o f16.- t)))) outputs targets
  let (grad_s, grad_t) = batch_gradients inputs grad_outputs
  let (new_weights_s, new_velocity_s) = sfd_update weights_s grad_s learning_rate momentum velocity_s
  let (new_weights_t, new_velocity_t) = sfd_update weights_t grad_t learning_rate momentum velocity_t

  in (new_weights_s, new_weights_t, new_velocity_s, new_velocity_t, loss)