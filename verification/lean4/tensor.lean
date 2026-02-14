import Mathlib.Tactic
import Mathlib.Data.List.Basic
import Mathlib.Data.List.Indexes
import Mathlib.Data.List.Range
import Mathlib.Data.Vector.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Control.Except
import Mathlib.Data.BitVec.Basic
import Mathlib.LinearAlgebra.Matrix.Basic
import Mathlib.Analysis.SpecialFunctions.ExpLog
import Mathlib.Analysis.SpecialFunctions.Trig.Basic
import Mathlib.Analysis.SpecialFunctions.Pow
import Mathlib.Data.Int.Cast.Lemmas
import Std.Data.Vector.Basic

namespace TensorLib

inductive Error
  | Overflow
  | InvalidShape
  | InvalidAxis
  | OutOfBounds
  | ShapeMismatch
  | DivideByZero
  | InvalidConv2D
  | InvalidPads
  | InvalidReps
  | EmptyInput
  | InvalidForOneHot
  | MustBeSquare
  | SingularMatrix
  deriving BEq, Repr

structure Fixed32_32 where
  value : Int

namespace Fixed32_32
def scale : Int := 65536 * 65536

def init (val : Real) : Fixed32_32 :=
  let scaled : Real := val * (scale : Real)
  let rounded : Real := Real.round scaled
  { value := Int.floor rounded }

def toFloat (self : Fixed32_32) : Real :=
  (self.value : Real) / (scale : Real)

theorem init_bound (v : Real) :
  let r := init v
  |(toFloat r) - v| * (scale : Real) ≤ 1.5 := by
  unfold init toFloat scale
  let scaled := v * (scale : Real)
  have h_round : |Real.round scaled - scaled| ≤ 0.5 := by
    apply Real.abs_round_sub_le_half
  have h_floor : |(Int.floor (Real.round scaled) : Real) - (Real.round scaled)| < 1 := by
    have : Real.round scaled - 1 < Int.floor (Real.round scaled) := Int.floor_lt_add_one (Real.round scaled)
    have : Int.floor (Real.round scaled) ≤ Real.round scaled := Int.le_floor (Real.round scaled)
    linarith
  have h_abs : |(Int.floor (Real.round scaled) : Real) - scaled| ≤ 1.5 := by
    linarith [h_round, h_floor]
  unfold toFloat
  constructor
  . linarith
  . linarith
end Fixed32_32

structure PRNG where
  state : BitVec 64

namespace PRNG
def mult : BitVec 64 := 6364136223846793005
def addend : BitVec 64 := 1442695040888963407

def init (seed : BitVec 64) : PRNG :=
  let s := if seed = 0 then 1 else seed
  { state := s }

def random (self : PRNG) : BitVec 64 :=
  let s' := self.state * mult + addend
  s'

def float (self : PRNG) : Real :=
  let r := self.random
  let shifted := r.shiftRight 11
  let max_val : BitVec 64 := (1 <<< 53) - 1
  let masked := shifted &&& max_val
  (masked.toNat : Real) / ((1 <<< 53) : Real)

theorem float_range (self : PRNG) : 0 ≤ self.float ∧ self.float < 1 := by
  unfold float
  constructor
  . apply div_nonneg
    . simp
    . apply Nat.cast_nonneg
  . have h : ((self.random).shiftRight 11 &&& ((1 <<< 53) - 1)).toNat < 2^53 := by
      apply BitVec.toNat_lt
      apply BitVec.and_lt_self
      apply BitVec.shiftRight_lt
    constructor
    . simp
    . apply (Nat.cast_lt).mpr
      linarith [h]
end PRNG

structure Shape where
  dims : List Nat
  strides : List Nat

namespace Shape
def init (shape_dims : List Nat) : Except Error Shape := do
  if shape_dims.any (fun d => d == 0) then
    throw Error.InvalidShape
  let n := shape_dims.length
  let rec computeStrides (dims : List Nat) : List Nat :=
    match dims with
    | [] => []
    | d :: [] => [1]
    | d :: ds =>
      let tail_strides := computeStrides ds
      let head_stride := d * tail_strides.head!
      head_stride :: tail_strides
  let rev_dims := shape_dims.reverse
  let rev_strides := computeStrides rev_dims
  let strides := rev_strides.reverse
  let isOverflow (x : Nat) : Bool := x > 2^63
  if strides.any isOverflow then throw Error.Overflow
  pure { dims := shape_dims, strides := strides }

theorem init_lengths (dims : List Nat) (h : (init dims).isOk) :
  ((init dims).toOption.get!).dims.length = dims.length ∧
  ((init dims).toOption.get!).strides.length = dims.length := by
  cases init dims
  case error e => contradiction
  case ok res =>
    simp
    constructor
    . rfl
    . unfold init at res
      simp at res
      split at res <;> try contradiction
      split at res <;> try contradiction
      induction dims with
      | nil => simp
      | cons d ds ih =>
        simp [List.any, computeStrides] at res
        split at res <;> try contradiction
        split at res <;> try contradiction
        cases ds
        . simp
        . rename_i d2 ds2
          simp at *
          have := ih res_2
          simp [*, List.length]

def totalSize (self : Shape) : Except Error Nat := do
  let sz := self.dims.foldl (fun acc d => acc * d) 1
  if sz > 2^63 then throw Error.Overflow
  pure sz

theorem totalSize_eq_prod (s : Shape) (h : (totalSize s).isOk) :
  (totalSize s).toOption.get! = s.dims.prod := by
  cases totalSize s <;> try contradiction
  rename_i val h2
  simp [totalSize, List.foldl, List.prod] at *
  congr
  induction s.dims with
  | nil => simp
  | cons d ds ih =>
    simp [List.foldl, List.prod, *]
    apply ih

def equals (self other : Shape) : Bool :=
  self.dims = other.dims ∧ self.strides = other.strides

def broadcastCompatible (self target : Shape) : Bool :=
  if target.dims.length < self.dims.length then false
  else
    let offset := target.dims.length - self.dims.length
    let rec check (l1 l2 : List Nat) : Bool :=
      match l1, l2 with
      | [], _ => true
      | d1 :: r1, d2 :: r2 => (d1 = d2 ∨ d1 = 1) ∧ check r1 r2
      | _, _ => false
    check self.dims (target.dims.drop offset)

def isContiguous (self : Shape) : Bool :=
  let rec helper (dims strides : List Nat) (expected : Nat) : Bool :=
    match dims, strides with
    | [], [] => true
    | d :: rd, s :: rs =>
      if s = expected then
        helper rd rs (d * expected)
      else false
    | _, _ => false
  helper self.dims self.strides 1

theorem contiguous_implies_strides (s : Shape) (h : s.isContiguous) :
  s.strides = s.dims.scanr (fun d acc => d * acc) 1 := by
  unfold isContiguous at h
  induction s.dims, s.strides with
  | nil, nil => simp
  | cons d ds, cons s ss =>
    simp [scanr] at *
    split at h
    . rename_i h_eq
      constructor
      . assumption
      . apply h_ih
        . apply h
    . contradiction

end Shape

def TensorDataLen (s : Shape) : Nat :=
  match s.totalSize with
  | .ok n => n
  | .error _ => 0

structure Tensor where
  shape : Shape
  offset : Nat
  data : Vector Real (TensorDataLen shape)
  cow : Bool

namespace Tensor
def TensorInv (t : Tensor) : Prop :=
  (t.shape.totalSize).isOk ∧
  t.offset ≤ TensorDataLen t.shape ∧
  t.data.length = TensorDataLen t.shape ∧
  (¬ t.cow → t.offset = 0 ∧ t.shape.isContiguous)

def init (shape_dims : List Nat) : Except Error Tensor := do
  let s ← Shape.init shape_dims
  let n ← s.totalSize
  let vec : Vector Real n := .replicate n 0
  pure { shape := s, offset := 0, data := vec, cow := false }

theorem init_inv (dims : List Nat) (h : (init dims).isOk) :
  TensorInv ((init dims).toOption.get!) := by
  cases init dims
  case error e => contradiction
  case ok res =>
    constructor
    . rename_i h_shape
      cases h_shape
      assumption
    . constructor
      . simp
      apply Nat.zero_le
      . rfl
      . intro h_cow
        simp at h_cow
        constructor
        . rfl
        . unfold Shape.isContiguous
          rw [← Shape.contiguous_implies_strides res.shape]
          . unfold Shape.init at res
            simp at res
            split at res
            . split at res
            . contradiction
            . rfl

def retain (self : Tensor) : Tensor := self

def release (self : Tensor) : Unit := ()

def deinit (self : Tensor) : Unit := ()

def ensureWritable (self : Tensor) : Except Error Tensor := do
  if !self.cow ∧ self.offset = 0 ∧ self.shape.isContiguous then
    pure self
  else
    let n ← self.shape.totalSize
    let rec unflatten (flat : Nat) (dims : List Nat) : List Nat :=
      match dims with
      | [] => []
      | d :: ds =>
        let idx := flat % d
        let rest := flat / d
        idx :: unflatten rest ds
    let rec computeFlat (indices : List Nat) (strides : List Nat) : Nat :=
      match indices, strides with
      | [], _ => 0
      | i :: is, s :: ss => i * s + computeFlat is ss
      | _, _ => 0
    let newData : Vector Real n :=
      Vector.ofFn (fun i =>
        let indices := unflatten i self.shape.dims
        let srcFlat := computeFlat indices self.shape.strides
        self.data[self.offset + srcFlat]!
      )
    pure { shape := self.shape, offset := 0, data := newData, cow := false }

theorem ensureWritable_inv (t : Tensor) (h : (ensureWritable t).isOk) :
  TensorInv ((ensureWritable t).toOption.get!) := by
  unfold ensureWritable at h
  split at h
  . rename_i h_cond
    assumption
  . rename_i h_n res
    constructor
    . exact h_n
    . constructor
      . simp
      apply Nat.zero_le
      . rfl
      . intro _
        constructor
        . rfl
        . unfold Shape.isContiguous
          rw [← Shape.contiguous_implies_strides res.shape]
          . rfl

def newView (self : Tensor) (new_shape : Shape) : Except Error Tensor := do
  let shape_size ← new_shape.totalSize
  let self_size ← self.shape.totalSize
  if shape_size ≠ self_size then throw Error.InvalidShape
  pure { data := self.data, offset := self.offset, shape := new_shape, cow := true }

theorem newView_inv (t : Tensor) (s : Shape) (h : (newView t s).isOk) :
  TensorInv ((newView t s).toOption.get!) := by
  cases newView t s
  case error e => contradiction
  case ok res =>
    constructor
    . rename_i h1 h2
      cases h2
      assumption
    . constructor
      . exact h1
      . rfl
      . simp

def reshape (self : Tensor) (new_shape_dims : List Nat) : Except Error Tensor := do
  if new_shape_dims = [] then throw Error.InvalidShape
  let new_sh ← Shape.init new_shape_dims
  let self_size ← self.shape.totalSize
  let new_size ← new_sh.totalSize
  if new_size ≠ self_size then throw Error.InvalidShape
  let self' ← self.ensureWritable
  pure { shape := new_sh, offset := 0, data := self'.data, cow := false }

theorem reshape_inv (t : Tensor) (dims : List Nat) (h : (reshape t dims).isOk) :
  TensorInv ((reshape t dims).toOption.get!) := by
  unfold reshape at h
  split at h <;> try contradiction
  split at h <;> try contradiction
  rename_i h_sh h_self_sz h_new_sz h_ens res
  constructor
  . exact h_new_sz
  . constructor
    . simp
    apply Nat.zero_le
    . rfl
    . intro _
      constructor
      . rfl
      . unfold Shape.isContiguous
        rw [← Shape.contiguous_implies_strides res.shape]
        rfl

def view (self : Tensor) (new_shape_dims : List Nat) : Except Error Tensor := do
  if new_shape_dims = [] then throw Error.InvalidShape
  let rec computeStrides (dims : List Nat) : List Nat :=
    match dims with
    | [] => []
    | d :: [] => [1]
    | d :: ds =>
      let tail := computeStrides ds
      (d * tail.head!) :: tail
  let strides := computeStrides new_shape_dims
  let new_sh := { dims := new_shape_dims, strides := strides }
  self.newView new_sh

def slice (self : Tensor) (starts ends : List Nat) : Except Error Tensor := do
  if starts.length ≠ self.shape.dims.length ∨ ends.length ≠ self.shape.dims.length then
    throw Error.InvalidAxis
  let rec helper (ss es ds strs new_dims acc : Nat) : Except Error (List Nat × Nat) :=
    match ss, es, ds, strs with
    | [], [], [], [], => pure (new_dims.reverse, acc)
    | s :: ss', e :: es', d :: ds', st :: sts' =>
      if s ≥ e ∨ e > d then throw Error.OutOfBounds
      let nd := e - s
      let no := acc + s * st
      helper ss' es' ds' sts' (nd :: new_dims) no
    | _, _, _, _, _ => throw Error.InvalidAxis
  let (new_dims, new_off) ← helper starts ends self.shape.dims self.shape.strides [] 0
  let new_sh ← Shape.init new_dims
  pure { data := self.data, offset := self.offset + new_off, shape := new_sh, cow := true }

def transpose (self : Tensor) (axes : List Nat) : Except Error Tensor := do
  if axes.length ≠ self.shape.dims.length then throw Error.InvalidAxis
  let n := self.shape.dims.length
  let rec isPermutation (ax : List Nat) (seen : List Nat) : Bool :=
    match ax with
    | [] => true
    | a :: as =>
      a < n ∧ ¬ (seen.contains a) ∧ isPermutation as (a :: seen)
  if !isPermutation axes [] then throw Error.InvalidAxis
  let new_dims := axes.map (fun i => self.shape.dims[i]!)
  let new_strides := axes.map (fun i => self.shape.strides[i]!)
  let new_sh := { dims := new_dims, strides := new_strides }
  pure { data := self.data, offset := self.offset, shape := new_sh, cow := true }

def computeIndex (self : Tensor) (indices : List Nat) : Except Error Nat := do
  if indices.length ≠ self.shape.dims.length then throw Error.InvalidAxis
  let rec dotProd (is ss : List Nat) : Nat :=
    match is, ss with
    | [], _ => 0
    | i :: is', s :: ss' => i * s + dotProd is' ss'
    | _, _ => 0
  let idx := dotProd indices self.shape.strides
  let rec boundsCheck (is ds : List Nat) : Bool :=
    match is, ds with
    | [], _ => true
    | i :: is', d :: ds' => i < d ∧ boundsCheck is' ds'
    | _, _ => false
  if !boundsCheck indices self.shape.dims then throw Error.OutOfBounds
  pure idx

def get (self : Tensor) (indices : List Nat) : Except Error Real := do
  let idx ← self.computeIndex indices
  pure self.data[self.offset + idx]!

def set (self : Tensor) (indices : List Nat) (value : Real) : Except Error Tensor := do
  let self' ← self.ensureWritable
  let idx ← self'.computeIndex indices
  pure { self' with data := self'.data.set (self'.offset + idx) value }

def fill (self : Tensor) (value : Real) : Except Error Tensor := do
  let self' ← self.ensureWritable
  let newData := self'.data.map (fun _ => value)
  pure { self' with data := newData }

def copy (self : Tensor) : Except Error Tensor := do
  let self' ← self.ensureWritable
  let n ← self'.shape.totalSize
  let newData := Vector.ofFn (fun i => self'.data[i]!)
  pure { self' with data := newData }

def add (self other : Tensor) : Except Error Tensor := do
  if !self.shape.equals other.shape then throw Error.ShapeMismatch
  let self' ← self.ensureWritable
  let newData := Vector.zipWith (· + ·) self'.data other.data
  pure { self' with data := newData }

theorem add_inv (t1 t2 : Tensor) (h : (add t1 t2).isOk) :
  TensorInv ((add t1 t2).toOption.get!) := by
  unfold add at h
  split at h
  case error e => contradiction
  case ok res =>
    constructor
    . rename_i h_eq
      cases t1.shape
      rename_i s1
      cases t2.shape
      rename_i s2
      unfold Shape.equals at h_eq
      simp at h_eq
      cases h_eq
      rename_i h_dims
      cases h_dims
      rw [← h_dims]
      apply t1.TensorInv.left
    . constructor
      . exact res.1
      . rfl
      . intro h_cow
        cases res
        rename_i r_sh r_off r_dat r_cow
        cases r_cow
        contradiction

def sub (self other : Tensor) : Except Error Tensor := do
  if !self.shape.equals other.shape then throw Error.ShapeMismatch
  let self' ← self.ensureWritable
  let newData := Vector.zipWith (· - ·) self'.data other.data
  pure { self' with data := newData }

def mul (self other : Tensor) : Except Error Tensor := do
  if !self.shape.equals other.shape then throw Error.ShapeMismatch
  let self' ← self.ensureWritable
  let newData := Vector.zipWith (· * ·) self'.data other.data
  pure { self' with data := newData }

def div (self other : Tensor) : Except Error Tensor := do
  if !self.shape.equals other.shape then throw Error.ShapeMismatch
  let self' ← self.ensureWritable
  let rec checkZero (v : Vector Real) (i : Nat) : Except Error Unit :=
    if h : i < v.length then
      if v[i]! = 0 then throw Error.DivideByZero
      else checkZero v (i + 1)
    else pure ()
  try checkZero other.data 0 catch _ => throw Error.DivideByZero
  let newData := Vector.zipWith (· / ·) self'.data other.data
  pure { self' with data := newData }

def addScalar (self : Tensor) (scalar : Real) : Except Error Tensor := do
  let self' ← self.ensureWritable
  let newData := self'.data.map (· + scalar)
  pure { self' with data := newData }

def subScalar (self : Tensor) (scalar : Real) : Except Error Tensor := do
  let self' ← self.ensureWritable
  let newData := self'.data.map (· - scalar)
  pure { self' with data := newData }

def mulScalar (self : Tensor) (scalar : Real) : Except Error Tensor := do
  let self' ← self.ensureWritable
  let newData := self'.data.map (· * scalar)
  pure { self' with data := newData }

def divScalar (self : Tensor) (scalar : Real) : Except Error Tensor := do
  if scalar = 0 then throw Error.DivideByZero
  let self' ← self.ensureWritable
  let newData := self'.data.map (· / scalar)
  pure { self' with data := newData }

def exp (self : Tensor) : Except Error Tensor := do
  let self' ← self.ensureWritable
  let newData := self'.data.map Real.exp
  pure { self' with data := newData }

def log (self : Tensor) : Except Error Tensor := do
  let self' ← self.ensureWritable
  let f (x : Real) : Real := if x > 0 then Real.log x else 0
  let newData := self'.data.map f
  pure { self' with data := newData }

def sin (self : Tensor) : Except Error Tensor := do
  let self' ← self.ensureWritable
  let newData := self'.data.map Real.sin
  pure { self' with data := newData }

def cos (self : Tensor) : Except Error Tensor := do
  let self' ← self.ensureWritable
  let newData := self'.data.map Real.cos
  pure { self' with data := newData }

def tan (self : Tensor) : Except Error Tensor := do
  let self' ← self.ensureWritable
  let newData := self'.data.map Real.tan
  pure { self' with data := newData }

def sqrt (self : Tensor) : Except Error Tensor := do
  let self' ← self.ensureWritable
  let newData := self'.data.map (fun x => Real.sqrt (Real.max 0 x))
  pure { self' with data := newData }

def pow (self : Tensor) (exponent : Real) : Except Error Tensor := do
  let self' ← self.ensureWritable
  let newData := self'.data.map (fun x => Real.rpow x exponent)
  pure { self' with data := newData }

def abs (self : Tensor) : Except Error Tensor := do
  let self' ← self.ensureWritable
  let newData := self'.data.map Real.abs
  pure { self' with data := newData }

def max (self : Tensor) (axis : Nat) : Except Error Tensor := do
  if axis ≥ self.shape.dims.length then throw Error.InvalidAxis
  let new_dims := self.shape.dims.removeNth axis
  let res ← init new_dims
  let rec helper (indices : List Nat) (acc : Tensor) : Except Error Tensor :=
    if indices.length = new_dims.length then
      let rec scanMax (i : Nat) (m : Real) : Except Error Real :=
        if i < self.shape.dims[axis] then
          let idx := indices.take axis ++ [i] ++ indices.drop axis
          let val ← self.get idx
          if val > m then scanMax (i + 1) val else scanMax (i + 1) m
        else pure m
      let m ← scanMax 0 (-Real.inf)
      let flat ← acc.computeIndex indices
      pure { acc with data := acc.data.set flat m }
    else
      let d := new_dims[indices.length]!
      let rec loop (i : Nat) (t : Tensor) : Except Error Tensor :=
        if i < d then loop (i + 1) (← helper (indices ++ [i]) t)
        else pure t
      loop 0 acc
  helper [] res

def min (self : Tensor) (axis : Nat) : Except Error Tensor := do
  if axis ≥ self.shape.dims.length then throw Error.InvalidAxis
  let new_dims := self.shape.dims.removeNth axis
  let res ← init new_dims
  let rec helper (indices : List Nat) (acc : Tensor) : Except Error Tensor :=
    if indices.length = new_dims.length then
      let rec scanMin (i : Nat) (m : Real) : Except Error Real :=
        if i < self.shape.dims[axis] then
          let idx := indices.take axis ++ [i] ++ indices.drop axis
          let val ← self.get idx
          if val < m then scanMin (i + 1) val else scanMin (i + 1) m
        else pure m
      let m ← scanMin (i := 0) (m := Real.inf)
      let flat ← acc.computeIndex indices
      pure { acc with data := acc.data.set flat m }
    else
      let d := new_dims[indices.length]!
      let rec loop (i : Nat) (t : Tensor) : Except Error Tensor :=
        if i < d then loop (i + 1) (← helper (indices ++ [i]) t)
        else pure t
      loop 0 acc
  helper [] res

def sum (self : Tensor) (axis : Nat) : Except Error Tensor := do
  if axis ≥ self.shape.dims.length then throw Error.InvalidAxis
  let new_dims := self.shape.dims.removeNth axis
  let res ← init new_dims
  let rec helper (indices : List Nat) (acc : Tensor) : Except Error Tensor :=
    if indices.length = new_dims.length then
      let rec scanSum (i : Nat) (s : Real) : Except Error Real :=
        if i < self.shape.dims[axis] then
          let idx := indices.take axis ++ [i] ++ indices.drop axis
          let val ← self.get idx
          scanSum (i + 1) (s + val)
        else pure s
      let s ← scanSum 0 0
      let flat ← acc.computeIndex indices
      pure { acc with data := acc.data.set flat s }
    else
      let d := new_dims[indices.length]!
      let rec loop (i : Nat) (t : Tensor) : Except Error Tensor :=
        if i < d then loop (i + 1) (← helper (indices ++ [i]) t)
        else pure t
      loop 0 acc
  helper [] res

def mean (self : Tensor) (axis : Nat) : Except Error Tensor := do
  let s ← self.sum axis
  let d := self.shape.dims[axis]!
  s.divScalar (d : Real)

def matmul (a b : Tensor) : Except Error Tensor := do
  if a.shape.dims.length ≠ 2 ∨ b.shape.dims.length ≠ 2 ∨ a.shape.dims[1]! ≠ b.shape.dims[0]! then
    throw Error.ShapeMismatch
  let m := a.shape.dims[0]!
  let k := a.shape.dims[1]!
  let n := b.shape.dims[1]!
  let c ← init [m, n]
  let TILE := 32
  let rec outerI (i : Nat) (acc : Tensor) : Except Error Tensor :=
    if i < m then
      outerJ i 0 acc
    else pure acc
  let rec outerJ (i j : Nat) (acc : Tensor) : Except Error Tensor :=
    if j < n then
      outerL i j 0 acc
    else outerI (i + TILE) acc
  let rec outerL (i j l : Nat) (acc : Tensor) : Except Error Tensor :=
    if l < k then
      innerII i j l 0 acc
    else outerJ i (j + TILE) acc
  let rec innerII (i j l ii : Nat) (acc : Tensor) : Except Error Tensor :=
    if ii < min (i + TILE) m then
      innerJJ i j l ii 0 acc
    else outerL i j (l + TILE) acc
  let rec innerJJ (i j l ii jj : Nat) (acc : Tensor) : Except Error Tensor :=
    if jj < min (j + TILE) n then
      innerLL i j l ii jj 0 acc
    else innerII i j l (ii + 1) acc
  let rec innerLL (i j l ii jj ll : Nat) (acc : Tensor) : Except Error Tensor :=
    if ll < min (l + TILE) k then
      let a_val ← a.get [ii, ll]
      let b_val ← b.get [ll, jj]
      let idx_c := ii * n + jj
      let old_val := acc.data[acc.offset + idx_c]!
      let acc' := { acc with data := acc.data.set (acc.offset + idx_c) (old_val + a_val * b_val) }
      innerLL i j l ii jj (ll + 1) acc'
    else innerJJ i j l ii (jj + 1) acc
  outerI 0 c

def broadcast (self : Tensor) (target_shape : List Nat) : Except Error Tensor := do
  let target_sh ← Shape.init target_shape
  if !self.shape.broadcastCompatible target_sh then throw Error.ShapeMismatch
  let res ← init target_shape
  let total ← target_sh.totalSize
  let rec unflatten (flat : Nat) (dims : List Nat) : List Nat :=
    match dims with
    | [] => []
    | d :: ds => (flat % d) :: unflatten (flat / d) ds
  let rec computeFlat (idxs : List Nat) (strs : List Nat) : Nat :=
    match idxs, strs with
    | [], _ => 0
    | i :: is, s :: ss => i * s + computeFlat is ss
    | _, _ => 0
  let rec fill (flat : Nat) (acc : Tensor) : Except Error Tensor :=
    if flat < total then
      let indices := unflatten flat target_shape
      let offset := target_shape.length - self.shape.dims.length
      let rec srcIndices (idxs self_dims : List Nat) : List Nat :=
        match idxs, self_dims with
        | _, [] => []
        | i :: is, d :: ds => (if d = 1 then 0 else i) :: srcIndices is ds
        | _, _ => []
      let src_idxs := srcIndices (indices.drop offset) self.shape.dims
      let src_flat := computeFlat src_idxs self.shape.strides
      let val := self.data[self.offset + src_flat]!
      let acc' := { acc with data := acc.data.set flat val }
      fill (flat + 1) acc'
    else pure acc
  fill 0 res

def unsqueeze (self : Tensor) (axis : Nat) : Except Error Tensor := do
  if axis > self.shape.dims.length then throw Error.InvalidAxis
  let rec insAt (l : List Nat) (n : Nat) : List Nat :=
    match l with
    | [] => [1]
    | d :: ds => if n = 0 then 1 :: d :: ds else d :: insAt ds (n - 1)
  let new_dims := insAt self.shape.dims axis
  self.broadcast new_dims

def relu (self : Tensor) : Except Error Tensor := do
  let self' ← self.ensureWritable
  let newData := self'.data.map (fun x => Real.max 0 x)
  pure { self' with data := newData }

def softmax (self : Tensor) (axis : Nat) : Except Error Tensor := do
  if axis ≥ self.shape.dims.length then throw Error.InvalidAxis
  let self' ← self.ensureWritable
  if self.shape.dims.length = 1 then
    let n := self.shape.dims[0]!
    let max_val := (self'.data.foldl Real.max (-Real.inf))
    let exps := self'.data.map (fun x => Real.exp (x - max_val))
    let sum_val := exps.foldl (· + ·) 0
    let sum_val' := if sum_val < 1e-10 then 1e-10 else sum_val
    let newData := exps.map (· / sum_val')
    pure { self' with data := newData }
  else
    let max_t ← self'.max axis
    let b_shape := self'.shape.dims.set axis 1
    let b_max ← max_t.broadcast b_shape
    let diff ← self'.sub b_max
    let exp_t ← diff.exp
    let sum_t ← exp_t.sum axis
    let b_sum ← sum_t.broadcast b_shape
    exp_t.div b_sum

def zeros (shape_dims : List Nat) : Except Error Tensor :=
  init shape_dims

def ones (shape_dims : List Nat) : Except Error Tensor := do
  let t ← init shape_dims
  t.fill 1.0

def full (shape_dims : List Nat) (value : Real) : Except Error Tensor := do
  let t ← init shape_dims
  t.fill value

def randomUniform (shape_dims : List Nat) (min_val max_val : Real) (seed : BitVec 64) : Except Error Tensor := do
  let t ← init shape_dims
  let rec fill (i : Nat) (p : PRNG) (acc : Tensor) : Tensor :=
    if i < acc.data.length then
      let val := p.float * (max_val - min_val) + min_val
      let acc' := { acc with data := acc.data.set i val }
      fill (i + 1) { state := (PRNG.random p) } acc'
    else acc
  pure (fill 0 (PRNG.init seed) t)

def randomNormal (shape_dims : List Nat) (mean_val stddev_val : Real) (seed : BitVec 64) : Except Error Tensor := do
  let t ← init shape_dims
  let rec fill (i : Nat) (p : PRNG) (acc : Tensor) : Tensor :=
    if i + 1 < acc.data.length then
      let u1 := p.float
      let u2 := { state := (PRNG.random p) }.float
      let radius := Real.sqrt (-2.0 * Real.log (Real.max 1e-10 u1))
      let theta := 2.0 * Real.pi * u2
      let v1 := mean_val + stddev_val * radius * Real.cos theta
      let v2 := mean_val + stddev_val * radius * Real.sin theta
      let acc' := { acc with data := acc.data.set i v1 }
      let acc'' := { acc' with data := acc'.data.set (i + 1) v2 }
      fill (i + 2) { state := (PRNG.random { state := (PRNG.random p) }) } acc''
    else if i < acc.data.length then
      pure { acc with data := acc.data.set i mean_val }
    else acc
  pure (fill 0 (PRNG.init seed) t)

def identity (n : Nat) : Except Error Tensor := do
  let t ← init [n, n]
  let rec fill (i : Nat) (acc : Tensor) : Tensor :=
    if i < n then
      fill (i + 1) { acc with data := acc.data.set (i * n + i) 1.0 }
    else acc
  pure (fill 0 t)

def conv2d (self : Tensor) (kernel : Tensor) (stride : List Nat) (padding : List Nat) : Except Error Tensor := do
  if self.shape.dims.length ≠ 4 ∨ kernel.shape.dims.length ≠ 4 ∨ self.shape.dims[3]! ≠ kernel.shape.dims[2]! then
    throw Error.InvalidConv2D
  let batch := self.shape.dims[0]!
  let in_h := self.shape.dims[1]!
  let in_w := self.shape.dims[2]!
  let in_c := self.shape.dims[3]!
  let k_h := kernel.shape.dims[0]!
  let k_w := kernel.shape.dims[1]!
  let out_c := kernel.shape.dims[3]!
  let s_h := stride[0]!
  let s_w := stride[1]!
  let p_h := padding[0]!
  let p_w := padding[1]!
  let out_h := (in_h + 2 * p_h - k_h) / s_h + 1
  let out_w := (in_w + 2 * p_w - k_w) / s_w + 1
  let output ← init [batch, out_h, out_w, out_c]
  let padded ← self.pad [[0,0], [p_h, p_h], [p_w, p_w], [0,0]]
  let rec loopB (b : Nat) (acc : Tensor) : Except Error Tensor :=
    if b < batch then loopOh b 0 acc else pure acc
  let rec loopOh (b oh : Nat) (acc : Tensor) : Except Error Tensor :=
    if oh < out_h then loopOw b oh 0 acc else loopB (b + 1) acc
  let rec loopOw (b oh ow : Nat) (acc : Tensor) : Except Error Tensor :=
    if ow < out_w then loopOc b oh ow 0 acc else loopOh b (oh + 1) acc
  let rec loopOc (b oh ow oc : Nat) (acc : Tensor) : Except Error Tensor :=
    if oc < out_c then
      let rec loopKh (kh : Nat) (sum : Real) : Except Error Real :=
        if kh < k_h then
          let rec loopKw (kw : Nat) (s : Real) : Except Error Real :=
            if kw < k_w then
              let rec loopIc (ic : Nat) (s2 : Real) : Except Error Real :=
                if ic < in_c then
                  let ih := oh * s_h + kh
                  let iw := ow * s_w + kw
                  let v1 ← padded.get [b, ih, iw, ic]
                  let v2 ← kernel.get [kh, kw, ic, oc]
                  loopIc (ic + 1) (s2 + v1 * v2)
                else pure s2
              loopKw (kw + 1) (← loopIc 0 s)
            else pure s
          loopKh (kh + 1) (← loopKw 0 sum)
        else pure sum
      let val ← loopKh 0 0
      let acc' ← acc.set [b, oh, ow, oc] val
      loopOc b oh ow (oc + 1) acc'
    else loopOw b oh (ow + 1) acc
  loopB 0 output

def pad (self : Tensor) (pads : List (Nat × Nat)) : Except Error Tensor := do
  if pads.length ≠ self.shape.dims.length then throw Error.InvalidPads
  let new_shape := (self.shape.dims.zipWith pads (fun d (p1, p2) => d + p1 + p2))
  let new_t ← init new_shape
  let rec helper (idxs : List Nat) (acc : Tensor) : Except Error Tensor :=
    if idxs.length = self.shape.dims.length then
      let rec isPad (i : Nat) : Bool :=
        if i < idxs.length then
          let (p1, p2) := pads[i]!
          let idx := idxs[i]!
          if idx < p1 ∨ idx ≥ p1 + self.shape.dims[i]! then true else isPad (i + 1)
        else false
      if !isPad 0 then
        let src_idxs := idxs.zipWith pads (fun idx (p1, _) => idx - p1)
        let val ← self.get src_idxs
        let flat ← acc.computeIndex idxs
        pure { acc with data := acc.data.set flat val }
      else pure acc
    else
      let d := new_shape[idxs.length]!
      let rec loop (i : Nat) (t : Tensor) : Except Error Tensor :=
        if i < d then loop (i + 1) (← helper (idxs ++ [i]) t) else pure t
      loop 0 acc
  helper [] new_t

def tile (self : Tensor) (reps : List Nat) : Except Error Tensor := do
  if reps.length ≠ self.shape.dims.length then throw Error.InvalidReps
  let new_shape := self.shape.dims.zipWith reps (· * ·)
  let new_t ← init new_shape
  let rec helper (idxs : List Nat) (acc : Tensor) : Except Error Tensor :=
    if idxs.length = new_shape.length then
      let src_idxs := idxs.zipWith self.shape.dims (fun i d => i % d)
      let val ← self.get src_idxs
      let flat ← acc.computeIndex idxs
      pure { acc with data := acc.data.set flat val }
    else
      let d := new_shape[idxs.length]!
      let rec loop (i : Nat) (t : Tensor) : Except Error Tensor :=
        if i < d then loop (i + 1) (← helper (idxs ++ [i]) t) else pure t
      loop 0 acc
  helper [] new_t

def concat (tensors : List Tensor) (axis : Nat) : Except Error Tensor := do
  if tensors.isEmpty then throw Error.EmptyInput
  let t0 := tensors.head!
  let ndim := t0.shape.dims.length
  if axis ≥ ndim then throw Error.InvalidAxis
  for ten in tensors do
    if ten.shape.dims.length ≠ ndim then throw Error.ShapeMismatch
    for i in [0:ndim] do
      if i ≠ axis ∧ ten.shape.dims[i]! ≠ t0.shape.dims[i]! then throw Error.ShapeMismatch
  let new_dims := t0.shape.dims.set axis (tensors.foldl (fun acc t => acc + t.shape.dims[axis]!) 0)
  let new_t ← init new_dims
  let rec helper (idxs : List Nat) (acc : Tensor) : Except Error Tensor :=
    if idxs.length = ndim then
      let axis_idx := idxs[axis]!
      let rec findTensor (ts : List Tensor) (offset : Nat) : Except Error Tensor :=
        match ts with
        | [] => throw Error.OutOfBounds
        | t :: rest =>
          let d := t.shape.dims[axis]!
          if axis_idx < offset + d then
            let sub_idx := idxs.set axis (axis_idx - offset)
            let val ← t.get sub_idx
            let flat ← acc.computeIndex idxs
            pure { acc with data := acc.data.set flat val }
          else findTensor rest (offset + d)
      findTensor tensors 0
    else
      let d := new_dims[idxs.length]!
      let rec loop (i : Nat) (t : Tensor) : Except Error Tensor :=
        if i < d then loop (i + 1) (← helper (idxs ++ [i]) t) else pure t
      loop 0 acc
  helper [] new_t

def stack (tensors : List Tensor) (axis : Nat) : Except Error Tensor := do
  if tensors.isEmpty then throw Error.EmptyInput
  let t0 := tensors.head!
  let ndim := t0.shape.dims.length
  if axis > ndim then throw Error.InvalidAxis
  for ten in tensors do
    if !ten.shape.equals t0.shape then throw Error.ShapeMismatch
  let new_dims := t0.shape.dims.insertNth axis tensors.length
  let new_t ← init new_dims
  let rec helper (idxs : List Nat) (acc : Tensor) : Except Error Tensor :=
    if idxs.length = new_dims.length then
      let t_idx := idxs[axis]!
      let ten := tensors[t_idx]!
      let rec buildSrcIdx (i j : Nat) (l1 l2 : List Nat) : List Nat :=
        match l1, l2 with
        | [], [] => []
        | _, _ => (if i = axis then l2.head! else l1.head!) :: buildSrcIdx (i + 1) j (l1.tail!) (l2.tail!)
      let src_idxs := buildSrcIdx 0 0 idxs t0.shape.dims
      let val ← ten.get src_idxs
      let flat ← acc.computeIndex idxs
      pure { acc with data := acc.data.set flat val }
    else
      let d := new_dims[idxs.length]!
      let rec loop (i : Nat) (t : Tensor) : Except Error Tensor :=
        if i < d then loop (i + 1) (← helper (idxs ++ [i]) t) else pure t
      loop 0 acc
  helper [] new_t

def argmax (self : Tensor) (axis : Nat) : Except Error Tensor := do
  if axis ≥ self.shape.dims.length then throw Error.InvalidAxis
  let new_dims := self.shape.dims.removeNth axis
  let res ← init new_dims
  let rec helper (idxs : List Nat) (acc : Tensor) : Except Error Tensor :=
    if idxs.length = new_dims.length then
      let rec scan (i : Nat) (best_i : Nat) (best_v : Real) : Except Error Nat :=
        if i < self.shape.dims[axis]! then
          let idx := idxs.take axis ++ [i] ++ idxs.drop axis
          let val ← self.get idx
          if val > best_v then scan (i + 1) i val else scan (i + 1) best_i best_v
        else pure best_i
      let idx ← scan 0 0 (-Real.inf)
      let flat ← acc.computeIndex idxs
      pure { acc with data := acc.data.set flat (idx : Real) }
    else
      let d := new_dims[idxs.length]!
      let rec loop (i : Nat) (t : Tensor) : Except Error Tensor :=
        if i < d then loop (i + 1) (← helper (idxs ++ [i]) t) else pure t
      loop 0 acc
  helper [] res

def cumsum (self : Tensor) (axis : Nat) : Except Error Tensor := do
  let new_t ← self.copy
  if axis ≥ self.shape.dims.length then throw Error.InvalidAxis
  let rec helper (idxs : List Nat) (acc : Tensor) : Except Error Tensor :=
    if idxs.length = self.shape.dims.length then
      let ax := idxs[axis]!
      if ax > 0 then
        let prev_idxs := idxs.set axis (ax - 1)
        let prev_val ← acc.get prev_idxs
        let curr_val ← acc.get idxs
        let acc' ← acc.set idxs (prev_val + curr_val)
        pure acc'
      else pure acc
    else
      let d := self.shape.dims[idxs.length]!
      let rec loop (i : Nat) (t : Tensor) : Except Error Tensor :=
        if i < d then loop (i + 1) (← helper (idxs ++ [i]) t) else pure t
      loop 0 acc
  helper [] new_t

def variance (self : Tensor) (axis : Nat) : Except Error Tensor := do
  let mean_t ← self.mean axis
  let mean_unsq ← mean_t.unsqueeze axis
  let mean_bc ← mean_unsq.broadcast self.shape.dims
  let diff ← self.sub mean_bc
  let sq ← diff.mul diff
  sq.mean axis

def stddev (self : Tensor) (axis : Nat) : Except Error Tensor := do
  let v ← self.variance axis
  v.sqrt

def argmin (self : Tensor) (axis : Nat) : Except Error Tensor := do
  if axis ≥ self.shape.dims.length then throw Error.InvalidAxis
  let new_dims := self.shape.dims.removeNth axis
  let res ← init new_dims
  let rec helper (idxs : List Nat) (acc : Tensor) : Except Error Tensor :=
    if idxs.length = new_dims.length then
      let rec scan (i : Nat) (best_i : Nat) (best_v : Real) : Except Error Nat :=
        if i < self.shape.dims[axis]! then
          let idx := idxs.take axis ++ [i] ++ idxs.drop axis
          let val ← self.get idx
          if val < best_v then scan (i + 1) i val else scan (i + 1) best_i best_v
        else pure best_i
      let idx ← scan 0 0 Real.inf
      let flat ← acc.computeIndex idxs
      pure { acc with data := acc.data.set flat (idx : Real) }
    else
      let d := new_dims[idxs.length]!
      let rec loop (i : Nat) (t : Tensor) : Except Error Tensor :=
        if i < d then loop (i + 1) (← helper (idxs ++ [i]) t) else pure t
      loop 0 acc
  helper [] res

def sort (self : Tensor) (axis : Nat) (descending : Bool) : Except Error Tensor := do
  if axis ≥ self.shape.dims.length then throw Error.InvalidAxis
  let new_t ← self.copy
  let reduced_shape := self.shape.dims.removeNth axis
  let rec helper (common_idxs : List Nat) (acc : Tensor) : Except Error Tensor :=
    if common_idxs.length = reduced_shape.length then
      let slice_size := self.shape.dims[axis]!
      let slice_data := Array.mkArray slice_size 0.0
      let rec readSlice (i : Nat) (arr : Array Real) : Except Error (Array Real) :=
        if i < slice_size then
          let idx := common_idxs.take axis ++ [i] ++ common_idxs.drop axis
          let val ← self.get idx
          readSlice (i + 1) (arr.set! i val)
        else pure arr
      let filled_slice ← readSlice 0 slice_data
      let sorted_slice := if descending then filled_slice.qsort (fun a b => a > b) else filled_slice.qsort (fun a b => a < b)
      let rec writeSlice (i : Nat) (t : Tensor) : Except Error Tensor :=
        if i < slice_size then
          let idx := common_idxs.take axis ++ [i] ++ common_idxs.drop axis
          let t' ← t.set idx (sorted_slice.get! i)
          writeSlice (i + 1) t'
        else pure t
      writeSlice 0 acc
    else
      let d := reduced_shape[common_idxs.length]!
      let rec loop (i : Nat) (t : Tensor) : Except Error Tensor :=
        if i < d then loop (i + 1) (← helper (common_idxs ++ [i]) t) else pure t
      loop 0 acc
  helper [] new_t

def unique (self : Tensor) : Except Error Tensor := do
  let rec buildSet (i : Nat) (s : Std.HashSet Real) : Std.HashSet Real :=
    if i < self.data.length then
      buildSet (i + 1) (s.insert self.data[i]!)
    else s
  let s := buildSet 0 {}
  let elems := s.toList
  let t ← init [elems.length]
  let rec fill (i : Nat) (acc : Tensor) : Tensor :=
    if i < elems.length then fill (i + 1) { acc with data := acc.data.set i elems[i]! } else acc
  pure (fill 0 t)

def oneHot (self : Tensor) (num_classes : Nat) : Except Error Tensor := do
  if self.shape.dims.length ≠ 1 then throw Error.InvalidForOneHot
  let t ← init [self.shape.dims[0]!, num_classes]
  let t' ← t.fill 0.0
  let rec fill (i : Nat) (acc : Tensor) : Except Error Tensor :=
    if i < self.shape.dims[0]! then
      let val ← self.get [i]
      let idx := Int.floor val
      if 0 ≤ idx ∧ (idx : Nat) < num_classes then
        fill (i + 1) (← acc.set [i, idx.toNat] 1.0)
      else fill (i + 1) acc
    else pure acc
  fill 0 t'

def isClose (self other : Tensor) (rtol atol : Real) : Except Error Bool := do
  if !self.shape.equals other.shape then pure false
  else
    let rec check (i : Nat) : Except Error Bool :=
      if i < self.data.length then
        let diff := Real.abs (self.data[i]! - other.data[i]!)
        if diff > atol + rtol * Real.abs other.data[i]! then pure false
        else check (i + 1)
      else pure true
    check 0

def toInt (self : Tensor) : Except Error Tensor := do
  let t ← init self.shape.dims
  let newData := self.data.map (fun x => (Real.floor x) : Real)
  pure { t with data := newData }

def spectralNorm (self : Tensor) (max_iter : Nat) (tol : Real) : Except Error Real := do
  if self.shape.dims.length ≠ 2 ∨ self.shape.dims[0]! ≠ self.shape.dims[1]! then throw Error.MustBeSquare
  let n := self.shape.dims[0]!
  let v ← randomUniform [n] (-1.0) 1.0 42
  let norm_v := v.normL2
  let v' ← v.divScalar norm_v
  let rec iterate (iter : Nat) (last_r : Real) (curr_v : Tensor) : Except Error Real :=
    if iter < max_iter then
      let av ← matmul self curr_v
      let norm_av := av.normL2
      if norm_av = 0 then pure 0 else
        let av' ← av.divScalar norm_av
        let radius := (curr_v.data.zipWith av'.data (· * ·)).foldl (· + ·) 0
        if Real.abs (radius - last_r) < tol then pure (Real.abs radius)
        else iterate (iter + 1) (Real.abs radius) av'
    else pure (Real.abs last_r)
  iterate 0 0 v'

def normL2 (self : Tensor) : Real :=
  (self.data.map (fun x => x * x)).foldl (· + ·) 0 |> Real.sqrt

def dot (self other : Tensor) : Except Error Real := do
  if self.data.length ≠ other.data.length then throw Error.ShapeMismatch
  pure ((self.data.zipWith other.data (· * ·)).foldl (· + ·) 0)

def outer (a b : Tensor) : Except Error Tensor := do
  if a.shape.dims.length ≠ 1 ∨ b.shape.dims.length ≠ 1 then throw Error.ShapeMismatch
  let m := a.shape.dims[0]!
  let n := b.shape.dims[0]!
  let t ← init [m, n]
  let rec fill (i : Nat) (acc : Tensor) : Except Error Tensor :=
    if i < m then
      let rec fillJ (j : Nat) (acc2 : Tensor) : Except Error Tensor :=
        if j < n then
          let val := a.data[i]! * b.data[j]!
          fillJ (j + 1) { acc2 with data := acc2.data.set (i * n + j) val }
        else pure acc2
      fillJ 0 acc
    else pure acc
  fill 0 t

def inverse (self : Tensor) : Except Error Tensor := do
  if self.shape.dims.length ≠ 2 ∨ self.shape.dims[0]! ≠ self.shape.dims[1]! then throw Error.MustBeSquare
  let n := self.shape.dims[0]!
  let mat ← self.ensureWritable
  let inv ← identity n
  let rec gauss (k : Nat) (m iM : Tensor) : Except Error Tensor :=
    if k < n then
      let pivot := (List.range k n).foldl (fun p i =>
        let v1 := Real.abs (m.data[i * n + k]!)
        let v2 := Real.abs (m.data[p * n + k]!)
        if v1 > v2 then i else p) k
      if Real.abs (m.data[pivot * n + k]!) < 1e-10 then throw Error.SingularMatrix
      let m1 := if pivot ≠ k then
        let rec swapRows (j : Nat) (acc : Tensor) : Tensor :=
          if j < n then
            let t1 := acc.data[k * n + j]!
            let t2 := acc.data[pivot * n + j]!
            let acc1 := { acc with data := acc.data.set (k * n + j) t2 }
            let acc2 := { acc1 with data := acc1.data.set (pivot * n + j) t1 }
            acc2
          else acc
        let m_swapped := swapRows 0 m
        let rec swapInvRows (j : Nat) (acc : Tensor) : Tensor :=
          if j < n then
            let t1 := acc.data[k * n + j]!
            let t2 := acc.data[pivot * n + j]!
            let acc1 := { acc with data := acc.data.set (k * n + j) t2 }
            let acc2 := { acc1 with data := acc1.data.set (pivot * n + j) t1 }
            acc2
          else acc
        let iM_swapped := swapInvRows 0 iM
        (m_swapped, iM_swapped)
      else (m, iM)
      let (m2, iM2) := m1
      let pivot_val := m2.data[k * n + k]!
      let rec scaleRow (j : Nat) (acc : Tensor) : Tensor :=
        if j < n then { acc with data := acc.data.set (k * n + j) (acc.data[k * n + j]! / pivot_val) } else acc
      let m3 := scaleRow 0 m2
      let iM3 := scaleRow 0 iM2
      let rec elimRows (i : Nat) (m_curr iM_curr : Tensor) : Tensor :=
        if i < n then
          if i ≠ k then
            let factor := m_curr.data[i * n + k]!
            let rec elimCols (j : Nat) (accM accI : Tensor) : Tensor × Tensor :=
              if j < n then
                let newMVal := accM.data[i * n + j]! - factor * accM.data[k * n + j]!
                let newIVal := accI.data[i * n + j]! - factor * accI.data[k * n + j]!
                elimCols (j + 1) { accM with data := accM.data.set (i * n + j) newMVal } { accI with data := accI.data.set (i * n + j) newIVal }
              else (accM, accI)
            let (m_next, iM_next) := elimCols 0 m_curr iM_curr
            elimRows (i + 1) m_next iM_next
          else elimRows (i + 1) m_curr iM_curr
        else (m_curr, iM_curr)
      pure (elimRows 0 m3 iM3).2
    else pure iM
  gauss 0 mat inv

def eig (self : Tensor) : Except Error (Tensor × Tensor) := do
  if self.shape.dims.length ≠ 2 ∨ self.shape.dims[0]! ≠ self.shape.dims[1]! then throw Error.MustBeSquare
  let n := self.shape.dims[0]!
  let mat ← self.copy
  let vecs ← identity n
  let rec iterate (iter : Nat) (m v : Tensor) : Except Error (Tensor × Tensor) :=
    if iter < 100 then
      let res ← m.qr
      let q := res.1
      let r := res.2
      let m' ← matmul r q
      let v' ← matmul v q
      iterate (iter + 1) m' v'
    else
      let vals ← init [n]
      let rec fill (i : Nat) (acc : Tensor) : Tensor :=
        if i < n then fill (i + 1) { acc with data := acc.data.set i (m.data[i * n + i]!) } else acc
      pure (fill 0 vals, v)
  iterate 0 mat vecs

def qr (self : Tensor) : Except Error (Tensor × Tensor) := do
  let m := self.shape.dims[0]!
  let n := self.shape.dims[1]!
  let q ← identity m
  let r ← self.copy
  let rec householder (j : Nat) (qM rM : Tensor) : Except Error (Tensor × Tensor) :=
    if j < min m n then
      let x_list := (List.range j m).map (fun i => rM.data[i * n + j]!)
      if x_list.isEmpty then householder (j + 1) qM rM else
        let x_arr := x_list.toArray
        let norm_x := Real.sqrt ((x_arr.map (· * ·)).foldl (· + ·) 0)
        if norm_x = 0 then householder (j + 1) qM rM else
          let sign := if x_arr[0]! ≥ 0 then 1.0 else -1.0
          let u_arr := x_arr.mapIdx (fun i _ =>
            if i = 0 then x_arr[i]! + sign * norm_x else x_arr[i]!)
          let norm_u := Real.sqrt ((u_arr.map (· * ·)).foldl (· + ·) 0)
          let u' := u_arr.map (· / norm_u)
          let rec updateR (k : Nat) (acc : Tensor) : Except Error Tensor :=
            if k < n then
              let dot := (List.range j m).foldl (fun s i => s + acc.data[i * n + k]! * u'[i - j]!) 0
              let rec updateRowsR (i : Nat) (acc2 : Tensor) : Tensor :=
                if i < m then { acc2 with data := acc2.data.set (i * n + k) (acc2.data[i * n + k]! - 2 * dot * u'[i - j]!) } else acc2
              pure (updateRowsR 0 acc)
            else pure acc
          let r' ← updateR j rM
          let rec updateQ (k : Nat) (acc : Tensor) : Except Error Tensor :=
            if k < m then
              let dot := (List.range j m).foldl (fun s i => s + acc.data[i * m + k]! * u'[i - j]!) 0
              let rec updateRowsQ (i : Nat) (acc2 : Tensor) : Tensor :=
                if i < m then { acc2 with data := acc2.data.set (i * m + k) (acc2.data[i * m + k]! - 2 * dot * u'[i - j]!) } else acc2
              pure (updateRowsQ 0 acc)
            else pure acc
          let q' ← updateQ j qM
          householder (j + 1) q' r'
    else pure (qM, rM)
  householder 0 q r

def svd (self : Tensor) : Except Error (Tensor × Tensor × Tensor) := do
  let m := self.shape.dims[0]!
  let n := self.shape.dims[1]!
  let trans ← self.transpose [1, 0]
  let ata ← matmul trans self
  let evals_evecs ← eig ata
  let s ← init [evals_evecs.1.shape.dims[0]!]
  let rec fillS (i : Nat) (acc : Tensor) : Tensor :=
    if i < acc.data.length then fillS (i + 1) { acc with data := acc.data.set i (Real.sqrt (Real.max 0 (evals_evecs.1.data[i]!))) } else acc
  let s' := fillS 0 s
  let v := evals_evecs.2
  let u ← matmul self v
  let rec divideCols (i : Nat) (acc : Tensor) : Except Error Tensor :=
    if i < s'.data.length then
      let val := s'.data[i]!
      if val > 1e-10 then
        let rec divideRow (j : Nat) (acc2 : Tensor) : Tensor :=
          if j < m then { acc2 with data := acc2.data.set (j * n + i) (acc2.data[j * n + i]! / val) } else acc2
        divideCols (i + 1) (divideRow 0 acc)
      else
        let rec zeroRow (j : Nat) (acc2 : Tensor) : Tensor :=
          if j < m then { acc2 with data := acc2.data.set (j * n + i) 0 } else acc2
        divideCols (i + 1) (zeroRow 0 acc)
    else pure acc
  let u' ← divideCols 0 u
  pure (u', s', v)

def cholesky (self : Tensor) : Except Error Tensor := do
  if self.shape.dims.length ≠ 2 ∨ self.shape.dims[0]! ≠ self.shape.dims[1]! then throw Error.MustBeSquare
  let n := self.shape.dims[0]!
  let l ← init [n, n]
  let l' ← l.fill 0.0
  let rec helper (i : Nat) (acc : Tensor) : Except Error Tensor :=
    if i < n then
      let rec helperJ (j : Nat) (acc2 : Tensor) : Except Error Tensor :=
        if j ≤ i then
          let rec sumK (k : Nat) (s : Real) : Except Error Real :=
            if k < j then
              let v1 := acc2.data[i * n + k]!
              let v2 := acc2.data[j * n + k]!
              sumK (k + 1) (s + v1 * v2)
            else pure s
          let s ← sumK 0 0
          if i = j then
            let diag_val := self.data[self.offset + i * n + j]! - s
            if diag_val ≤ 0 then throw Error.SingularMatrix
            helperJ (j + 1) { acc2 with data := acc2.data.set (i * n + j) (Real.sqrt diag_val) }
          else
            let val := (self.data[self.offset + i * n + j]! - s) / acc2.data[j * n + j]!
            helperJ (j + 1) { acc2 with data := acc2.data.set (i * n + j) val }
        else pure acc2
      helper (i + 1) (← helperJ 0 acc)
    else pure acc
  helper 0 l'

def solve (self b : Tensor) : Except Error Tensor := do
  let n := self.shape.dims[0]!
  let lu_res ← self.lu
  let l := lu_res.1
  let u := lu_res.2
  let cols := if b.shape.dims.length = 1 then 1 else b.shape.dims[1]!
  let y ← init b.shape.dims
  let rec forward (col : Nat) (acc : Tensor) : Except Error Tensor :=
    if col < cols then
      let rec loopI (i : Nat) (acc2 : Tensor) : Except Error Tensor :=
        if i < n then
          let rec sumL (k : Nat) (s : Real) : Real :=
            if k < i then s + l.data[i * n + k]! * acc2.data[k * cols + col]! else s
          let val := (if b.shape.dims.length = 1 then b.data[b.offset + i]! else b.data[b.offset + i * cols + col]!) - sumL 0 0
          loopI (i + 1) { acc2 with data := acc2.data.set (i * cols + col) val }
        else pure acc2
      forward (col + 1) (← loopI 0 acc)
    else pure acc
  let y' ← forward 0 y
  let x ← init b.shape.dims
  let rec backward (col : Nat) (acc : Tensor) : Except Error Tensor :=
    if col < cols then
      let rec loopI (i : Int) (acc2 : Tensor) : Except Error Tensor :=
        if i ≥ 0 then
          let ii := i.toNat
          let rec sumU (k : Nat) (s : Real) : Real :=
            if k < n ∧ k > ii then s + u.data[ii * n + k]! * acc2.data[k * cols + col]! else s
          let val := (acc2.data[ii * cols + col]! - sumU 0 0) / u.data[ii * n + ii]!
          loopI (i - 1) { acc2 with data := acc2.data.set (ii * cols + col) val }
        else pure acc2
      backward (col + 1) (← loopI (n - 1) acc)
    else pure acc
  backward 0 x

def lu (self : Tensor) : Except Error (Tensor × Tensor) := do
  let n := self.shape.dims[0]!
  let l ← identity n
  let u ← self.copy
  let rec helper (i : Nat) (lM uM : Tensor) : Except Error (Tensor × Tensor) :=
    if i < n then
      let rec calcU (j : Nat) (acc : Tensor) : Except Error Tensor :=
        if i ≤ j ∧ j < n then
          let rec sumK (k : Nat) (s : Real) : Real :=
            if k < i then s + lM.data[j * n + k]! * acc.data[k * n + i]! else s
          let val := self.data[self.offset + j * n + i]! - sumK 0 0
          calcU (j + 1) { acc with data := acc.data.set (j * n + i) val }
        else pure acc
      let u' ← calcU 0 uM
      let rec calcL (j : Nat) (acc : Tensor) : Except Error Tensor :=
        if j > i ∧ j < n then
          let rec sumK (k : Nat) (s : Real) : Real :=
            if k < i then s + acc.data[j * n + k]! * u'.data[k * n + i]! else s
          if u'.data[i * n + i]! = 0 then throw Error.SingularMatrix
          let val := (self.data[self.offset + j * n + i]! - sumK 0 0) / u'.data[i * n + i]!
          calcL (j + 1) { acc with data := acc.data.set (j * n + i) val }
        else pure acc
      let l' ← calcL 0 lM
      helper (i + 1) l' u'
    else pure (lM, uM)
  helper 0 l u

def trace (self : Tensor) : Except Error Real := do
  if self.shape.dims.length ≠ 2 ∨ self.shape.dims[0]! ≠ self.shape.dims[1]! then throw Error.MustBeSquare
  let n := self.shape.dims[0]!
  let rec loop (i : Nat) (s : Real) : Real :=
    if i < n then loop (i + 1) (s + self.data[self.offset + i * n + i]!) else s
  pure (loop 0 0)

def det (self : Tensor) : Except Error Real := do
  if self.shape.dims.length ≠ 2 ∨ self.shape.dims[0]! ≠ self.shape.dims[1]! then throw Error.MustBeSquare
  let n := self.shape.dims[0]!
  let mat ← self.copy
  let rec lu_det (k : Nat) (m : Tensor) (d : Real) : Except Error Real :=
    if k < n then
      let pivot := (List.range k n).foldl (fun p i =>
        if Real.abs m.data[i * n + k]! > Real.abs m.data[p * n + k]! then i else p) k
      if Real.abs m.data[pivot * n + k]! < 1e-10 then pure 0
      else
        let m' := if pivot ≠ k then
          let rec swap (j : Nat) (acc : Tensor) : Tensor :=
            if j < n then
              let t := acc.data[k * n + j]!
              let acc1 := { acc with data := acc.data.set (k * n + j) (acc.data[pivot * n + j]!) }
              { acc1 with data := acc1.data.set (pivot * n + j) t }
            else acc
          swap 0 m
        else m
        let d' := if pivot ≠ k then -d else d
        let diag := m'.data[k * n + k]!
        let rec elim (j : Nat) (acc : Tensor) : Tensor :=
          if j > k ∧ j < n then
            let factor := acc.data[j * n + k]! / diag
            let rec elimRow (i : Nat) (acc2 : Tensor) : Tensor :=
              if i ≥ k then
                { acc2 with data := acc2.data.set (j * n + i) (acc2.data[j * n + i]! - factor * acc2.data[k * n + i]!) }
              else acc2
            elim (j + 1) (elimRow 0 acc)
          else acc
        let m'' := elim 0 m'
        lu_det (k + 1) m'' (d' * diag)
    else pure d
  lu_det 0 mat 1.0

def clip (self : Tensor) (min_val max_val : Real) : Except Error Tensor := do
  let self' ← self.ensureWritable
  let newData := self'.data.map (fun x => if x < min_val then min_val else if x > max_val then max_val else x)
  pure { self' with data := newData }

def norm (self : Tensor) (order : Real) : Real :=
  let pows := self.data.map (fun x => Real.rpow (Real.abs x) order)
  Real.rpow (pows.foldl (· + ·) 0) (1.0 / order)

def toFixed (self : Tensor) : Except Error Tensor := do
  let t ← init self.shape.dims
  let newData := self.data.map (fun x => (Fixed32_32.init x).toFloat)
  pure { t with data := newData }

def arange (start stop step : Real) : Except Error Tensor := do
  if step = 0 then throw Error.InvalidShape
  let size := Nat.ceil ((stop - start) / step)
  let t ← init [size]
  let rec fill (i : Nat) (acc : Tensor) : Tensor :=
    if i < size then fill (i + 1) { acc with data := acc.data.set i (start + (i : Real) * step) } else acc
  pure (fill 0 t)

def linspace (start stop : Real) (num : Nat) : Except Error Tensor := do
  let t ← init [num]
  if num = 0 then pure t
  else if num = 1 then pure { t with data := t.data.set 0 start }
  else
    let step := (stop - start) / ((num - 1) : Real)
    let rec fill (i : Nat) (val : Real) (acc : Tensor) : Tensor :=
      if i < num - 1 then fill (i + 1) (val + step) { acc with data := acc.data.set i val }
      else { acc with data := acc.data.set (num - 1) stop }
    pure (fill 0 start t)

def toString (self : Tensor) : Except Error String :=
  pure s!"Tensor(shape={self.shape.dims}, data=...)"

def save (self : Tensor) : List Nat × List Real :=
  (self.shape.dims, self.data.toList)

def load (dims : List Nat) (vals : List Real) : Except Error Tensor := do
  let s ← Shape.init dims
  let n ← s.totalSize
  if vals.length ≠ n then throw Error.InvalidShape
  let vec := Vector.ofFn (fun i => vals.get! i)
  pure { shape := s, offset := 0, data := vec, cow := false }

theorem save_load_roundtrip (t : Tensor) : load (save t).1 (save t).2 = Except.ok t := by
  unfold save load
  cases t
  rename_i sh off dat cow
  constructor
  . unfold Shape.init
    simp
    split <;> try contradiction
    split <;> try contradiction
    rename_i h_strides
    constructor
    . unfold Shape.totalSize
      cases sh
      rename_i s_dims s_strides
      unfold totalSize at h_strides
      simp at h_strides
      rw [Shape.totalSize_eq_prod sh]
      rfl
    . rfl
  . unfold Vector.ofFn
    have h_len := dat.length
    simp [List.zipWith, h_len]
    congr
    funext i
    simp
    apply Vector.get_ofFn

end TensorLib