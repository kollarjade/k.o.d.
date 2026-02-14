import Std
import Mathlib.Control.Monad.State

open Std

abbrev USize := Nat
abbrev Address := Nat
abbrev F32 := Float
abbrev U64 := UInt64

inductive ZigError where
  | OutOfMemory
  | InvalidDimension
  | InvalidLayerCount
  | TooLarge
  | InvalidConfig
  | ShapeMismatch
  | DataLengthMismatch
  | NonFinite
  | NumericFailure
  | NotInitialized
  | Overflow
  | Freed
deriving Repr, Inhabited, DecidableEq

abbrev Heap := Array (Option F32)

structure Allocator where
  dummy : Unit := ()
deriving Repr, Inhabited

abbrev ZM := StateT Heap (Except ZigError)

namespace F32Ops

def isFinite (x : F32) : Bool := x.isFinite

def clamp (x lo hi : F32) : F32 :=
  if x < lo then lo else if hi < x then hi else x

def sqrt (x : F32) : F32 := Float.sqrt x
def exp (x : F32) : F32 := Float.exp x

end F32Ops

namespace HeapModel

def allocF32Array (size : Nat) : ZM Address := do
  let heap ← get
  let addr := heap.size
  set (heap ++ Array.mkArray size (some (0.0 : F32)))
  pure addr

def freeF32Array (addr size : Nat) : ZM Unit := do
  for i in [:size] do
    modify (fun heap => heap.set (addr + i.val) none)

def writeF32 (addr : Address) (val : F32) : ZM Unit := do
  let heap ← get
  match heap.get? addr with
  | some (some _) =>
      set (heap.set addr (some val))
  | some none =>
      throw ZigError.Freed
  | none =>
      throw ZigError.ShapeMismatch

def readF32 (addr : Address) : ZM F32 := do
  let heap ← get
  match heap.get? addr with
  | some (some v) => pure v
  | some none => throw ZigError.Freed
  | none => throw ZigError.ShapeMismatch

end HeapModel

open HeapModel

structure ZTensor where
  dataPtr : Address
  size : Nat
  shape : List Nat
deriving Repr

namespace ZTensor

def totalSize : List Nat → Nat
  | [] => 1
  | d :: ds => d * totalSize ds

def assert2DAndLen (t : ZTensor) : ZM (Nat × Nat) := do
  match t.shape with
  | [r, c] =>
      if t.size = r * c then
        pure (r, c)
      else
        throw ZigError.DataLengthMismatch
  | _ => throw ZigError.ShapeMismatch

def zeros (_alloc : Allocator) (shape : List Nat) : ZM ZTensor := do
  let sz := totalSize shape
  if sz = 0 then
    throw ZigError.InvalidDimension
  let addr ← HeapModel.allocF32Array sz
  pure { dataPtr := addr, size := sz, shape := shape }

def rngStep (s : U64) : U64 :=
  (6364136223846793005 : U64) * s + (1 : U64)

def rngFloat01From (s : U64) : F32 :=
  let x : U64 := s >>> 40
  let denom : F32 := Float.ofNat (Nat.pow 2 24)
  (Float.ofNat x.toNat) / denom

def uniform (s : U64) (lo hi : F32) : (F32 × U64) :=
  let s' := rngStep s
  let u := rngFloat01From s'
  (lo + u * (hi - lo), s')

def randomUniform (alloc : Allocator) (shape : List Nat) (lo hi : F32) (seed : U64) : ZM ZTensor := do
  let t ← zeros alloc shape
  let mut state := seed
  for i in [:t.size] do
    let (v, s') := uniform state lo hi
    state := s'
    HeapModel.writeF32 (t.dataPtr + i.val) v
  pure t

def deinit (t : ZTensor) : ZM Unit :=
  HeapModel.freeF32Array t.dataPtr t.size

def copy (alloc : Allocator) (t : ZTensor) : ZM ZTensor := do
  let out ← zeros alloc t.shape
  for i in [:t.size] do
    let v ← HeapModel.readF32 (t.dataPtr + i.val)
    HeapModel.writeF32 (out.dataPtr + i.val) v
  pure out

def clampFiniteInPlace (t : ZTensor) : ZM Unit := do
  for i in [:t.size] do
    let v ← HeapModel.readF32 (t.dataPtr + i.val)
    if !(F32Ops.isFinite v) then
      HeapModel.writeF32 (t.dataPtr + i.val) 0.0

def clipInPlace (t : ZTensor) (lo hi : F32) : ZM Unit := do
  for i in [:t.size] do
    let v ← HeapModel.readF32 (t.dataPtr + i.val)
    HeapModel.writeF32 (t.dataPtr + i.val) (F32Ops.clamp v lo hi)

def expInPlace (t : ZTensor) : ZM Unit := do
  for i in [:t.size] do
    let v ← HeapModel.readF32 (t.dataPtr + i.val)
    HeapModel.writeF32 (t.dataPtr + i.val) (F32Ops.exp v)

def addInPlace (a b : ZTensor) : ZM Unit := do
  if a.size != b.size then throw ZigError.ShapeMismatch
  for i in [:a.size] do
    let va ← HeapModel.readF32 (a.dataPtr + i.val)
    let vb ← HeapModel.readF32 (b.dataPtr + i.val)
    HeapModel.writeF32 (a.dataPtr + i.val) (va + vb)

def subInPlace (a b : ZTensor) : ZM Unit := do
  if a.size != b.size then throw ZigError.ShapeMismatch
  for i in [:a.size] do
    let va ← HeapModel.readF32 (a.dataPtr + i.val)
    let vb ← HeapModel.readF32 (b.dataPtr + i.val)
    HeapModel.writeF32 (a.dataPtr + i.val) (va - vb)

def mulInPlace (a b : ZTensor) : ZM Unit := do
  if a.size != b.size then throw ZigError.ShapeMismatch
  for i in [:a.size] do
    let va ← HeapModel.readF32 (a.dataPtr + i.val)
    let vb ← HeapModel.readF32 (b.dataPtr + i.val)
    HeapModel.writeF32 (a.dataPtr + i.val) (va * vb)

def divInPlace (a b : ZTensor) : ZM Unit := do
  if a.size != b.size then throw ZigError.ShapeMismatch
  for i in [:a.size] do
    let va ← HeapModel.readF32 (a.dataPtr + i.val)
    let vb ← HeapModel.readF32 (b.dataPtr + i.val)
    if vb = 0.0 || !(F32Ops.isFinite vb) then throw ZigError.NumericFailure
    HeapModel.writeF32 (a.dataPtr + i.val) (va / vb)

def addBiasInPlace (t bias : ZTensor) : ZM Unit := do
  let (batch, dim) ← assert2DAndLen t
  let (br, bc) ← assert2DAndLen bias
  if br != 1 || bc != dim then throw ZigError.ShapeMismatch
  if bias.size != dim then throw ZigError.DataLengthMismatch
  for b in [:batch] do
    for d in [:dim] do
      let idx := b.val * dim + d.val
      let x ← HeapModel.readF32 (t.dataPtr + idx)
      let bb ← HeapModel.readF32 (bias.dataPtr + d.val)
      HeapModel.writeF32 (t.dataPtr + idx) (x + bb)

def transpose (alloc : Allocator) (t : ZTensor) : ZM ZTensor := do
  let (rows, cols) ← assert2DAndLen t
  let out ← zeros alloc [cols, rows]
  for r in [:rows] do
    for c in [:cols] do
      let src := r.val * cols + c.val
      let dst := c.val * rows + r.val
      let v ← HeapModel.readF32 (t.dataPtr + src)
      HeapModel.writeF32 (out.dataPtr + dst) v
  pure out

def matmul (alloc : Allocator) (a b : ZTensor) : ZM ZTensor := do
  let (m, k1) ← assert2DAndLen a
  let (k2, n) ← assert2DAndLen b
  if k1 != k2 then throw ZigError.ShapeMismatch
  let out ← zeros alloc [m, n]
  for i in [:m] do
    for j in [:n] do
      let mut sum : F32 := 0.0
      for k in [:k1] do
        let va ← HeapModel.readF32 (a.dataPtr + i.val * k1 + k.val)
        let vb ← HeapModel.readF32 (b.dataPtr + k.val * n + j.val)
        sum := sum + va * vb
      HeapModel.writeF32 (out.dataPtr + i.val * n + j.val) sum
  pure out

def checkAllNonzeroFinite (t : ZTensor) : ZM Unit := do
  for i in [:t.size] do
    let v ← HeapModel.readF32 (t.dataPtr + i.val)
    if v = 0.0 || !(F32Ops.isFinite v) then throw ZigError.NumericFailure

def withDeinit (t : ZTensor) (k : ZTensor → ZM α) : ZM α := do
  try
    let r ← k t
    deinit t
    pure r
  catch e =>
    deinit t
    throw e

end ZTensor

structure RSFLayerConfig where
  clip_min : F32 := -5.0
  clip_max : F32 := 5.0
  seed_offset : U64 := 0
  grad_mean : Bool := true
deriving Repr

structure RSFConfig where
  clip_min : F32 := -5.0
  clip_max : F32 := 5.0
  grad_mean : Bool := true
  max_dim : Nat := Nat.shiftLeft 1 20
  max_layers : Nat := Nat.shiftLeft 1 20
deriving Repr

structure ZRSFLayer where
  dim : Nat
  s_weight : ZTensor
  t_weight : ZTensor
  s_bias : ZTensor
  t_bias : ZTensor
  s_weight_grad : ZTensor
  t_weight_grad : ZTensor
  s_bias_grad : ZTensor
  t_bias_grad : ZTensor
  clip_min : F32
  clip_max : F32
  grad_mean : Bool
deriving Repr

namespace ZRSFLayer

def checkedAddU64 (a b : U64) : Except ZigError U64 :=
  let s := a + b
  if s < a then .error ZigError.Overflow else .ok s

def initWithConfig (alloc : Allocator) (dim : Nat) (cfg : RSFLayerConfig) : ZM ZRSFLayer := do
  if dim = 0 then throw ZigError.InvalidDimension
  if !(F32Ops.isFinite cfg.clip_min) || !(F32Ops.isFinite cfg.clip_max) then throw ZigError.NonFinite
  if !(cfg.clip_min < cfg.clip_max) then throw ZigError.InvalidConfig
  let fan_in : F32 := Float.ofNat dim
  let fan_out : F32 := Float.ofNat dim
  let fan_sum : F32 := fan_in + fan_out
  if !(fan_sum > 0.0) then throw ZigError.InvalidDimension
  let xavier : F32 := F32Ops.sqrt (6.0 / fan_sum)
  let seed1 : U64 :=
    match checkedAddU64 (42 : U64) cfg.seed_offset with
    | .ok s => s
    | .error e => throw e
  let seed2 : U64 :=
    match checkedAddU64 (43 : U64) cfg.seed_offset with
    | .ok s => s
    | .error e => throw e
  let s_weight ← ZTensor.randomUniform alloc [dim, dim] (-xavier) xavier seed1
  let t_weight ← ZTensor.randomUniform alloc [dim, dim] (-xavier) xavier seed2
  let s_bias ← ZTensor.zeros alloc [1, dim]
  let t_bias ← ZTensor.zeros alloc [1, dim]
  let s_weight_grad ← ZTensor.zeros alloc [dim, dim]
  let t_weight_grad ← ZTensor.zeros alloc [dim, dim]
  let s_bias_grad ← ZTensor.zeros alloc [1, dim]
  let t_bias_grad ← ZTensor.zeros alloc [1, dim]
  pure
    { dim := dim
      s_weight := s_weight
      t_weight := t_weight
      s_bias := s_bias
      t_bias := t_bias
      s_weight_grad := s_weight_grad
      t_weight_grad := t_weight_grad
      s_bias_grad := s_bias_grad
      t_bias_grad := t_bias_grad
      clip_min := cfg.clip_min
      clip_max := cfg.clip_max
      grad_mean := cfg.grad_mean }

def deinit (self : ZRSFLayer) : ZM Unit := do
  ZTensor.deinit self.s_weight
  ZTensor.deinit self.t_weight
  ZTensor.deinit self.s_bias
  ZTensor.deinit self.t_bias
  ZTensor.deinit self.s_weight_grad
  ZTensor.deinit self.t_weight_grad
  ZTensor.deinit self.s_bias_grad
  ZTensor.deinit self.t_bias_grad

def zeroGradients (self : ZRSFLayer) : ZM Unit := do
  for i in [:self.s_weight_grad.size] do HeapModel.writeF32 (self.s_weight_grad.dataPtr + i.val) 0.0
  for i in [:self.t_weight_grad.size] do HeapModel.writeF32 (self.t_weight_grad.dataPtr + i.val) 0.0
  for i in [:self.s_bias_grad.size] do HeapModel.writeF32 (self.s_bias_grad.dataPtr + i.val) 0.0
  for i in [:self.t_bias_grad.size] do HeapModel.writeF32 (self.t_bias_grad.dataPtr + i.val) 0.0

def assertPair (self : ZRSFLayer) (a b : ZTensor) : ZM Nat := do
  let (ar, ac) ← ZTensor.assert2DAndLen a
  let (br, bc) ← ZTensor.assert2DAndLen b
  if ac != self.dim || bc != self.dim then throw ZigError.ShapeMismatch
  if ar != br then throw ZigError.ShapeMismatch
  if self.s_bias.size != self.dim || self.t_bias.size != self.dim then throw ZigError.ShapeMismatch
  if !(F32Ops.isFinite self.clip_min) || !(F32Ops.isFinite self.clip_max) then throw ZigError.NonFinite
  if !(self.clip_min < self.clip_max) then throw ZigError.InvalidConfig
  pure ar

def forward (self : ZRSFLayer) (alloc : Allocator) (x1 x2 : ZTensor) : ZM Unit := do
  let batch ← assertPair self x1 x2
  ZTensor.withDeinit (← ZTensor.transpose alloc x2) fun x2_t => do
    ZTensor.withDeinit (← ZTensor.matmul alloc self.s_weight x2_t) fun s_x2_t => do
      ZTensor.withDeinit (← ZTensor.transpose alloc s_x2_t) fun s_x2 => do
        ZTensor.addBiasInPlace s_x2 self.s_bias
        ZTensor.clampFiniteInPlace s_x2
        ZTensor.clipInPlace s_x2 self.clip_min self.clip_max
        ZTensor.clampFiniteInPlace s_x2
        ZTensor.expInPlace s_x2
        ZTensor.clampFiniteInPlace s_x2
        for b in [:batch] do
          for d in [:self.dim] do
            let idx := b.val * self.dim + d.val
            let v1 ← HeapModel.readF32 (x1.dataPtr + idx)
            let s ← HeapModel.readF32 (s_x2.dataPtr + idx)
            HeapModel.writeF32 (x1.dataPtr + idx) (v1 * s)
        ZTensor.withDeinit (← ZTensor.transpose alloc x1) fun x1_t => do
          ZTensor.withDeinit (← ZTensor.matmul alloc self.t_weight x1_t) fun t_x1_t => do
            ZTensor.withDeinit (← ZTensor.transpose alloc t_x1_t) fun t_x1 => do
              ZTensor.addBiasInPlace t_x1 self.t_bias
              ZTensor.clampFiniteInPlace t_x1
              ZTensor.addInPlace x2 t_x1
              ZTensor.clampFiniteInPlace x2

def inverse (self : ZRSFLayer) (alloc : Allocator) (y1 y2 : ZTensor) : ZM Unit := do
  let batch ← assertPair self y1 y2
  ZTensor.withDeinit (← ZTensor.transpose alloc y1) fun y1_t => do
    ZTensor.withDeinit (← ZTensor.matmul alloc self.t_weight y1_t) fun t_y1_t => do
      ZTensor.withDeinit (← ZTensor.transpose alloc t_y1_t) fun t_y1 => do
        ZTensor.addBiasInPlace t_y1 self.t_bias
        ZTensor.clampFiniteInPlace t_y1
        ZTensor.subInPlace y2 t_y1
        ZTensor.clampFiniteInPlace y2
        ZTensor.withDeinit (← ZTensor.transpose alloc y2) fun y2_t => do
          ZTensor.withDeinit (← ZTensor.matmul alloc self.s_weight y2_t) fun s_y2_t => do
            ZTensor.withDeinit (← ZTensor.transpose alloc s_y2_t) fun s_y2 => do
              ZTensor.addBiasInPlace s_y2 self.s_bias
              ZTensor.clampFiniteInPlace s_y2
              ZTensor.clipInPlace s_y2 self.clip_min self.clip_max
              ZTensor.clampFiniteInPlace s_y2
              ZTensor.expInPlace s_y2
              ZTensor.clampFiniteInPlace s_y2
              ZTensor.checkAllNonzeroFinite s_y2
              ZTensor.divInPlace y1 s_y2
              ZTensor.clampFiniteInPlace y1
  let _ := batch
  pure ()

def backward
    (self : ZRSFLayer)
    (alloc : Allocator)
    (y1 y2 dy1_in dy2_in : ZTensor)
    (x1_out x2_out dx1_out dx2_out : ZTensor)
    : ZM Unit := do
  let batch ← assertPair self y1 y2
  let _ ← assertPair self dy1_in dy2_in
  let (r1, c1) ← ZTensor.assert2DAndLen x1_out
  let (r2, c2) ← ZTensor.assert2DAndLen x2_out
  let (r3, c3) ← ZTensor.assert2DAndLen dx1_out
  let (r4, c4) ← ZTensor.assert2DAndLen dx2_out
  if r1 != batch || r2 != batch || r3 != batch || r4 != batch then throw ZigError.ShapeMismatch
  if c1 != self.dim || c2 != self.dim || c3 != self.dim || c4 != self.dim then throw ZigError.ShapeMismatch
  if x1_out.size != batch * self.dim || x2_out.size != batch * self.dim || dx1_out.size != batch * self.dim || dx2_out.size != batch * self.dim then
    throw ZigError.DataLengthMismatch
  let gradScale : F32 := if self.grad_mean then 1.0 / (Float.ofNat batch) else 1.0
  ZTensor.withDeinit (← ZTensor.copy alloc dy1_in) fun dy1 => do
    ZTensor.withDeinit (← ZTensor.copy alloc dy2_in) fun dy2 => do
      ZTensor.withDeinit (← ZTensor.transpose alloc y1) fun y1_t => do
        ZTensor.withDeinit (← ZTensor.matmul alloc self.t_weight y1_t) fun t_y1_t => do
          ZTensor.withDeinit (← ZTensor.transpose alloc t_y1_t) fun t_y1 => do
            ZTensor.addBiasInPlace t_y1 self.t_bias
            ZTensor.clampFiniteInPlace t_y1
            ZTensor.withDeinit (← ZTensor.copy alloc y2) fun x2 => do
              ZTensor.subInPlace x2 t_y1
              ZTensor.clampFiniteInPlace x2
              ZTensor.withDeinit (← ZTensor.transpose alloc dy2) fun dy2_t => do
                ZTensor.withDeinit (← ZTensor.matmul alloc dy2_t y1) fun dt => do
                  for i in [:dt.size] do
                    let g ← HeapModel.readF32 (self.t_weight_grad.dataPtr + i.val)
                    let v ← HeapModel.readF32 (dt.dataPtr + i.val)
                    HeapModel.writeF32 (self.t_weight_grad.dataPtr + i.val) (g + v * gradScale)
                  for b in [:batch] do
                    for d in [:self.dim] do
                      let idx := b.val * self.dim + d.val
                      let g ← HeapModel.readF32 (self.t_bias_grad.dataPtr + d.val)
                      let v ← HeapModel.readF32 (dy2.dataPtr + idx)
                      HeapModel.writeF32 (self.t_bias_grad.dataPtr + d.val) (g + v * gradScale)
                ZTensor.withDeinit (← ZTensor.transpose alloc self.t_weight) fun t_weight_t => do
                  ZTensor.withDeinit (← ZTensor.matmul alloc t_weight_t dy2_t) fun grad_from_t_t => do
                    ZTensor.withDeinit (← ZTensor.transpose alloc grad_from_t_t) fun grad_from_t => do
                      ZTensor.addInPlace dy1 grad_from_t
                      ZTensor.clampFiniteInPlace dy1
                      ZTensor.withDeinit (← ZTensor.transpose alloc x2) fun x2_t => do
                        ZTensor.withDeinit (← ZTensor.matmul alloc self.s_weight x2_t) fun s_pre_t => do
                          ZTensor.withDeinit (← ZTensor.transpose alloc s_pre_t) fun s_pre => do
                            ZTensor.addBiasInPlace s_pre self.s_bias
                            ZTensor.clampFiniteInPlace s_pre
                            ZTensor.withDeinit (← ZTensor.copy alloc s_pre) fun s_clipped => do
                              ZTensor.clipInPlace s_clipped self.clip_min self.clip_max
                              ZTensor.clampFiniteInPlace s_clipped
                              ZTensor.withDeinit (← ZTensor.copy alloc s_clipped) fun exp_s => do
                                ZTensor.expInPlace exp_s
                                ZTensor.clampFiniteInPlace exp_s
                                ZTensor.checkAllNonzeroFinite exp_s
                                ZTensor.withDeinit (← ZTensor.copy alloc y1) fun x1 => do
                                  ZTensor.divInPlace x1 exp_s
                                  ZTensor.clampFiniteInPlace x1
                                  for b in [:batch] do
                                    for d in [:self.dim] do
                                      let idx := b.val * self.dim + d.val
                                      let v1 ← HeapModel.readF32 (x1.dataPtr + idx)
                                      let v2 ← HeapModel.readF32 (x2.dataPtr + idx)
                                      HeapModel.writeF32 (x1_out.dataPtr + idx) v1
                                      HeapModel.writeF32 (x2_out.dataPtr + idx) v2
                                  for b in [:batch] do
                                    for d in [:self.dim] do
                                      let idx := b.val * self.dim + d.val
                                      let vdy1 ← HeapModel.readF32 (dy1.dataPtr + idx)
                                      let vs ← HeapModel.readF32 (exp_s.dataPtr + idx)
                                      HeapModel.writeF32 (dx1_out.dataPtr + idx) (vdy1 * vs)
                                  ZTensor.withDeinit (← ZTensor.copy alloc dy1) fun dscale => do
                                    ZTensor.mulInPlace dscale x1
                                    ZTensor.withDeinit (← ZTensor.copy alloc dscale) fun ds => do
                                      ZTensor.mulInPlace ds exp_s
                                      for b in [:batch] do
                                        for d in [:self.dim] do
                                          let idx := b.val * self.dim + d.val
                                          let v ← HeapModel.readF32 (s_pre.dataPtr + idx)
                                          if !(F32Ops.isFinite v) || v < self.clip_min || v > self.clip_max then
                                            HeapModel.writeF32 (ds.dataPtr + idx) 0.0
                                      ZTensor.withDeinit (← ZTensor.transpose alloc ds) fun ds_t => do
                                        ZTensor.withDeinit (← ZTensor.matmul alloc ds_t x2) fun ds_w => do
                                          for i in [:ds_w.size] do
                                            let g ← HeapModel.readF32 (self.s_weight_grad.dataPtr + i.val)
                                            let v ← HeapModel.readF32 (ds_w.dataPtr + i.val)
                                            HeapModel.writeF32 (self.s_weight_grad.dataPtr + i.val) (g + v * gradScale)
                                          for b in [:batch] do
                                            for d in [:self.dim] do
                                              let idx := b.val * self.dim + d.val
                                              let g ← HeapModel.readF32 (self.s_bias_grad.dataPtr + d.val)
                                              let v ← HeapModel.readF32 (ds.dataPtr + idx)
                                              HeapModel.writeF32 (self.s_bias_grad.dataPtr + d.val) (g + v * gradScale)
                                        ZTensor.withDeinit (← ZTensor.transpose alloc self.s_weight) fun s_weight_t => do
                                          ZTensor.withDeinit (← ZTensor.matmul alloc s_weight_t ds_t) fun grad_from_s_t => do
                                            ZTensor.withDeinit (← ZTensor.transpose alloc grad_from_s_t) fun grad_from_s => do
                                              for b in [:batch] do
                                                for d in [:self.dim] do
                                                  let idx := b.val * self.dim + d.val
                                                  let vdy2 ← HeapModel.readF32 (dy2.dataPtr + idx)
                                                  let vg ← HeapModel.readF32 (grad_from_s.dataPtr + idx)
                                                  HeapModel.writeF32 (dx2_out.dataPtr + idx) (vdy2 + vg)
                                              ZTensor.clampFiniteInPlace dx1_out
                                              ZTensor.clampFiniteInPlace dx2_out

end ZRSFLayer

structure ZControlBlock where
  freed : Bool
  allocator : Allocator
  dim : Nat
  num_layers : Nat
  cfg : RSFConfig
  layers : Array ZRSFLayer
deriving Repr

structure ZRSF where
  ctrl : ZControlBlock
deriving Repr

namespace ZRSF

def ensureNotFreed (self : ZRSF) : ZM Unit := do
  if self.ctrl.freed then throw ZigError.NotInitialized else pure ()

def initWithConfig (alloc : Allocator) (dim num_layers : Nat) (cfg : RSFConfig) : ZM ZRSF := do
  if dim = 0 then throw ZigError.InvalidDimension
  if num_layers = 0 then throw ZigError.InvalidLayerCount
  if dim > cfg.max_dim || num_layers > cfg.max_layers then throw ZigError.TooLarge
  if !(F32Ops.isFinite cfg.clip_min) || !(F32Ops.isFinite cfg.clip_max) then throw ZigError.NonFinite
  if !(cfg.clip_min < cfg.clip_max) then throw ZigError.InvalidConfig
  let mut layers : Array ZRSFLayer := #[]
  let mut i := 0
  while i < num_layers do
    let seed_offset : U64 := (UInt64.ofNat i) * (1000 : U64)
    let layer_cfg : RSFLayerConfig :=
      { clip_min := cfg.clip_min
        clip_max := cfg.clip_max
        seed_offset := seed_offset
        grad_mean := cfg.grad_mean }
    try
      let layer ← ZRSFLayer.initWithConfig alloc dim layer_cfg
      layers := layers.push layer
      i := i + 1
    catch e =>
      for l in layers do
        ZRSFLayer.deinit l
      throw e
  pure
    { ctrl :=
        { freed := false
          allocator := alloc
          dim := dim
          num_layers := num_layers
          cfg := cfg
          layers := layers } }

def deinit (self : ZRSF) : ZM ZRSF := do
  if self.ctrl.freed then
    pure self
  else
    for l in self.ctrl.layers do
      ZRSFLayer.deinit l
    pure { ctrl := { self.ctrl with freed := true } }

def split (self : ZRSF) (x : ZTensor) (x1 x2 : ZTensor) : ZM Nat := do
  ensureNotFreed self
  let (batch, cols) ← ZTensor.assert2DAndLen x
  if cols != self.ctrl.dim * 2 then throw ZigError.ShapeMismatch
  let (b1, d1) ← ZTensor.assert2DAndLen x1
  let (b2, d2) ← ZTensor.assert2DAndLen x2
  if b1 != batch || b2 != batch || d1 != self.ctrl.dim || d2 != self.ctrl.dim then throw ZigError.ShapeMismatch
  if x1.size != batch * self.ctrl.dim || x2.size != batch * self.ctrl.dim then throw ZigError.DataLengthMismatch
  for b in [:batch] do
    for d in [:self.ctrl.dim] do
      let idx1 := b.val * self.ctrl.dim + d.val
      let idx := b.val * (2 * self.ctrl.dim) + d.val
      let v1 ← HeapModel.readF32 (x.dataPtr + idx)
      HeapModel.writeF32 (x1.dataPtr + idx1) v1
      let v2 ← HeapModel.readF32 (x.dataPtr + (idx + self.ctrl.dim))
      HeapModel.writeF32 (x2.dataPtr + idx1) v2
  pure batch

def merge (self : ZRSF) (x1 x2 : ZTensor) (out : ZTensor) : ZM Unit := do
  ensureNotFreed self
  let (batch1, dim1) ← ZTensor.assert2DAndLen x1
  let (batch2, dim2) ← ZTensor.assert2DAndLen x2
  let (batch3, dim3) ← ZTensor.assert2DAndLen out
  if batch1 != batch2 || batch1 != batch3 then throw ZigError.ShapeMismatch
  if dim1 != self.ctrl.dim || dim2 != self.ctrl.dim then throw ZigError.ShapeMismatch
  if dim3 != 2 * self.ctrl.dim then throw ZigError.ShapeMismatch
  for b in [:batch1] do
    for d in [:self.ctrl.dim] do
      let idx1 := b.val * self.ctrl.dim + d.val
      let idx := b.val * (2 * self.ctrl.dim) + d.val
      let v1 ← HeapModel.readF32 (x1.dataPtr + idx1)
      let v2 ← HeapModel.readF32 (x2.dataPtr + idx1)
      HeapModel.writeF32 (out.dataPtr + idx) v1
      HeapModel.writeF32 (out.dataPtr + (idx + self.ctrl.dim)) v2

def zeroGradients (self : ZRSF) : ZM Unit := do
  ensureNotFreed self
  for l in self.ctrl.layers do
    ZRSFLayer.zeroGradients l

def forward (self : ZRSF) (x : ZTensor) : ZM Unit := do
  ensureNotFreed self
  let (batch, cols) ← ZTensor.assert2DAndLen x
  if cols != 2 * self.ctrl.dim then throw ZigError.ShapeMismatch
  ZTensor.withDeinit (← ZTensor.zeros self.ctrl.allocator [batch, self.ctrl.dim]) fun x1 => do
    ZTensor.withDeinit (← ZTensor.zeros self.ctrl.allocator [batch, self.ctrl.dim]) fun x2 => do
      let _ ← split self x x1 x2
      for l in self.ctrl.layers do
        ZRSFLayer.forward l self.ctrl.allocator x1 x2
      merge self x1 x2 x

def inverse (self : ZRSF) (y : ZTensor) : ZM Unit := do
  ensureNotFreed self
  let (batch, cols) ← ZTensor.assert2DAndLen y
  if cols != 2 * self.ctrl.dim then throw ZigError.ShapeMismatch
  ZTensor.withDeinit (← ZTensor.zeros self.ctrl.allocator [batch, self.ctrl.dim]) fun y1 => do
    ZTensor.withDeinit (← ZTensor.zeros self.ctrl.allocator [batch, self.ctrl.dim]) fun y2 => do
      let _ ← split self y y1 y2
      for l in self.ctrl.layers.reverse do
        ZRSFLayer.inverse l self.ctrl.allocator y1 y2
      merge self y1 y2 y

def backward (self : ZRSF) (grad_output input grad_input_out : ZTensor) : ZM Unit := do
  ensureNotFreed self
  let (batch, cols) ← ZTensor.assert2DAndLen input
  let (gb, gcols) ← ZTensor.assert2DAndLen grad_output
  let (ob, ocols) ← ZTensor.assert2DAndLen grad_input_out
  if cols != 2 * self.ctrl.dim then throw ZigError.ShapeMismatch
  if gb != batch || gcols != cols then throw ZigError.ShapeMismatch
  if ob != batch || ocols != cols then throw ZigError.ShapeMismatch
  ZTensor.withDeinit (← ZTensor.copy self.ctrl.allocator input) fun x => do
    forward self x
    ZTensor.withDeinit (← ZTensor.zeros self.ctrl.allocator [batch, self.ctrl.dim]) fun y1 => do
      ZTensor.withDeinit (← ZTensor.zeros self.ctrl.allocator [batch, self.ctrl.dim]) fun y2 => do
        let _ ← split self x y1 y2
        ZTensor.withDeinit (← ZTensor.copy self.ctrl.allocator grad_output) fun dy => do
          ZTensor.withDeinit (← ZTensor.zeros self.ctrl.allocator [batch, self.ctrl.dim]) fun dy1 => do
            ZTensor.withDeinit (← ZTensor.zeros self.ctrl.allocator [batch, self.ctrl.dim]) fun dy2 => do
              let _ ← split self dy dy1 dy2
              ZTensor.withDeinit (← ZTensor.copy self.ctrl.allocator y1) fun cur_y1 => do
                ZTensor.withDeinit (← ZTensor.copy self.ctrl.allocator y2) fun cur_y2 => do
                  ZTensor.withDeinit (← ZTensor.copy self.ctrl.allocator dy1) fun cur_dy1 => do
                    ZTensor.withDeinit (← ZTensor.copy self.ctrl.allocator dy2) fun cur_dy2 => do
                      ZTensor.withDeinit (← ZTensor.zeros self.ctrl.allocator [batch, self.ctrl.dim]) fun x1_prev => do
                        ZTensor.withDeinit (← ZTensor.zeros self.ctrl.allocator [batch, self.ctrl.dim]) fun x2_prev => do
                          ZTensor.withDeinit (← ZTensor.zeros self.ctrl.allocator [batch, self.ctrl.dim]) fun dx1_prev => do
                            ZTensor.withDeinit (← ZTensor.zeros self.ctrl.allocator [batch, self.ctrl.dim]) fun dx2_prev => do
                              let mut idx := self.ctrl.num_layers
                              let mut cy1 := cur_y1
                              let mut cy2 := cur_y2
                              let mut cdy1 := cur_dy1
                              let mut cdy2 := cur_dy2
                              let mut xp1 := x1_prev
                              let mut xp2 := x2_prev
                              let mut dxp1 := dx1_prev
                              let mut dxp2 := dx2_prev
                              while idx > 0 do
                                let layer := self.ctrl.layers.get! (idx - 1)
                                ZRSFLayer.backward layer self.ctrl.allocator cy1 cy2 cdy1 cdy2 xp1 xp2 dxp1 dxp2
                                let tmpy1 := cy1
                                cy1 := xp1
                                xp1 := tmpy1
                                let tmpy2 := cy2
                                cy2 := xp2
                                xp2 := tmpy2
                                let tmpdy1 := cdy1
                                cdy1 := dxp1
                                dxp1 := tmpdy1
                                let tmpdy2 := cdy2
                                cdy2 := dxp2
                                dxp2 := tmpdy2
                                idx := idx - 1
                              merge self cdy1 cdy2 grad_input_out

end ZRSF
