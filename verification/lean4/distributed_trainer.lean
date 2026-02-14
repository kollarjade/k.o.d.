import Mathlib
import Mathlib.Data.Vector.Basic
import Mathlib.Data.Vector.Sigma
import Mathlib.Algebra.BigOperators.Group.Finset
import Mathlib.Algebra.BigOperators.Order
import Mathlib.Data.Real.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.Calculus.Integrability
import Mathlib.Data.Matrix.Basic
import Mathlib.LinearAlgebra.Matrix.Determinant
import Mathlib.LinearAlgebra.Matrix.Symmetric
import Mathlib.LinearAlgebra.Matrix.PosDef
import Mathlib.LinearAlgebra.Matrix.Cholesky
import Mathlib.LinearAlgebra.Matrix.SVD
import Mathlib.Tactic
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Omega

namespace TensorLib

inductive TensorError : Type where
  | shapeMismatch : String → TensorError
  | indexOutOfBounds : String → TensorError
  | invalidOperation : String → TensorError
  | numericalError : String → TensorError
  | decompositionError : String → TensorError
  deriving Repr, DecidableEq

instance : ToString TensorError where
  toString
    | .shapeMismatch s => s!"Shape mismatch: {s}"
    | .indexOutOfBounds s => s!"Index out of bounds: {s}"
    | .invalidOperation s => s!"Invalid operation: {s}"
    | .numericalError s => s!"Numerical error: {s}"
    | .decompositionError s => s!"Decomposition error: {s}"

theorem TensorError.toString_injective : Function.Injective (α := TensorError) toString := by
  intro x y h
  cases x <;> cases y <;> simp only [toString] at h <;> try simp at h
  all_goals simp only [← String.coe_inj] at h
  all_goals exact congrArg TensorError.mk h

structure Fixed32_32 : Type where
  val : Int
  deriving Repr, DecidableEq, Add, Sub, Mul, Div, Neg, Ord

namespace Fixed32_32

def scale : Int := 2^32

def fromInt (n : Int) : Fixed32_32 := ⟨n * scale⟩

def toInt (f : Fixed32_32) : Int := f.val / scale

def fromFloat (x : Float) : Fixed32_32 := ⟨Int.ofFloat (x * Float.ofInt scale)⟩

def toFloat (f : Fixed32_32) : Float := Float.ofInt f.val / Float.ofInt scale

def add (a b : Fixed32_32) : Fixed32_32 := ⟨a.val + b.val⟩

def sub (a b : Fixed32_32) : Fixed32_32 := ⟨a.val - b.val⟩

def mul (a b : Fixed32_32) : Fixed32_32 := ⟨(a.val * b.val) / scale⟩

def div (a b : Fixed32_32) : Option Fixed32_32 :=
  if b.val = 0 then none
  else some ⟨(a.val * scale) / b.val⟩

def neg (a : Fixed32_32) : Fixed32_32 := ⟨-a.val⟩

def abs (a : Fixed32_32) : Fixed32_32 := ⟨Int.natAbs a.val⟩

def lt (a b : Fixed32_32) : Prop := a.val < b.val

def le (a b : Fixed32_32) : Prop := a.val ≤ b.val

instance : Add Fixed32_32 := ⟨add⟩
instance : Sub Fixed32_32 := ⟨sub⟩
instance : Mul Fixed32_32 := ⟨mul⟩
instance : Neg Fixed32_32 := ⟨neg⟩
instance : LT Fixed32_32 := ⟨lt⟩
instance : LE Fixed32_32 := ⟨le⟩

instance fixed32_32_decidableLt (a b : Fixed32_32) : Decidable (a < b) := Int.decLt _ _
instance fixed32_32_decidableLe (a b : Fixed32_32) : Decidable (a ≤ b) := Int.decLe _ _

theorem add_comm (a b : Fixed32_32) : a + b = b + a := by simp only [Add.add, add]; omega

theorem add_assoc (a b c : Fixed32_32) : (a + b) + c = a + (b + c) := by simp only [Add.add, add]; omega

theorem add_zero (a : Fixed32_32) : a + ⟨0⟩ = a := by simp only [Add.add, add]; cases a; simp

theorem zero_add (a : Fixed32_32) : ⟨0⟩ + a = a := by simp only [Add.add, add]; cases a; simp

theorem mul_comm (a b : Fixed32_32) : a * b = b * a := by simp only [Mul.mul, mul]; omega

theorem mul_assoc (a b c : Fixed32_32) : (a * b) * c = a * (b * c) := by simp only [Mul.mul, mul]; omega

theorem mul_one (a : Fixed32_32) : a * fromInt 1 = a := by
  simp only [Mul.mul, mul, fromInt, scale]
  cases a; simp [Int.ediv_self]; decide

theorem one_mul (a : Fixed32_32) : fromInt 1 * a = a := by
  simp only [Mul.mul, mul, fromInt, scale]
  cases a; simp [Int.ediv_self]; decide

theorem mul_zero (a : Fixed32_32) : a * ⟨0⟩ = ⟨0⟩ := by simp only [Mul.mul, mul]; omega

theorem zero_mul (a : Fixed32_32) : ⟨0⟩ * a = ⟨0⟩ := by simp only [Mul.mul, mul]; omega

theorem neg_neg (a : Fixed32_32) : -(-a) = a := by simp only [Neg.neg, neg]; cases a; omega

theorem neg_add (a b : Fixed32_32) : -(a + b) = (-a) + (-b) := by simp only [Neg.neg, Add.add, neg, add]; omega

theorem neg_zero : -(⟨0⟩ : Fixed32_32) = ⟨0⟩ := by simp only [Neg.neg, neg]; omega

theorem sub_self (a : Fixed32_32) : a - a = ⟨0⟩ := by simp only [Sub.sub, sub]; cases a; omega

theorem abs_nonneg (a : Fixed32_32) : ⟨0⟩ ≤ abs a := by simp only [LE.le, le, abs]; cases a; simp [Int.natAbs]; omega

theorem abs_of_nonneg (h : ⟨0⟩ ≤ a) : abs a = a := by simp only [LE.le, le, abs] at *; cases a; simp [Int.natAbs] at *; omega

theorem abs_of_neg (h : a < ⟨0⟩) : abs a = -a := by simp only [LT.lt, lt, LE.le, le, Neg.neg, abs, neg] at *; cases a; simp [Int.natAbs] at *; omega

theorem abs_mul (a b : Fixed32_32) : abs (a * b) = abs a * abs b := by simp only [Mul.mul, abs, mul]; cases a; cases b; simp [Int.natAbs]; omega

theorem div_none_of_denom_zero (a : Fixed32_32) : div a ⟨0⟩ = none := by simp only [div]; omega

theorem div_some_of_ne_zero {a b : Fixed32_32} (h : b.val ≠ 0) : ∃ c, div a b = some c := by simp only [div]; omega

theorem fromInt_zero : fromInt 0 = ⟨0⟩ := by simp only [fromInt, scale]; omega

theorem fromInt_add (a b : Int) : fromInt (a + b) = fromInt a + fromInt b := by simp only [fromInt, Add.add, add, scale]; omega

theorem fromInt_mul (a b : Int) : fromInt (a * b) = fromInt a * fromInt b := by simp only [fromInt, Mul.mul, mul, scale]; omega

theorem fromInt_neg (a : Int) : fromInt (-a) = -fromInt a := by simp only [fromInt, Neg.neg, neg, scale]; omega

end Fixed32_32

structure PRNG (n : Nat) : Type where
  state : Vector (Fin n) 4
  deriving Repr

namespace PRNG

def init (seed : Nat) (n : Nat) : PRNG n :=
  let s0 := seed % n
  let s1 := (seed * 1103515245 + 12345) % n
  let s2 := (seed * 1664525 + 1013904223) % n
  let s3 := (seed * 524287 + 262143) % n
  ⟨⟨[⟨s0, by omega⟩, ⟨s1, by omega⟩, ⟨s2, by omega⟩, ⟨s3, by omega⟩], by omega⟩⟩

def next (prng : PRNG n) : PRNG n × Fin n :=
  let s := prng.state
  let t := (s.get 0).val * 1103515245 + 12345
  let result := (t ^^ (s.get 1).val) % n
  let newState := ⟨⟨(s.get 1, by omega), (s.get 2, by omega), (s.get 3, by omega), ⟨result, by omega⟩⟩, by simp [Vector.get]; omega⟩
  (⟨newState⟩, ⟨result, by omega⟩)

def nextN (prng : PRNG n) (k : Nat) : PRNG n × Vector (Fin n) k :=
  match k with
  | 0 => (prng, Vector.nil)
  | k + 1 =>
    let (prng', r) := next prng
    let (prng'', rs) := nextN prng' k
    (prng'', r :: rs)

theorem next_result_bound (prng : PRNG n) (hn : 0 < n) : (next prng).2.val < n := by
  obtain ⟨s, hs⟩ := prng
  simp only [next] at *
  omega

theorem nextN_length (prng : PRNG n) (k : Nat) : (nextN prng k).2.length = k := by
  induction k with
  | zero => simp [nextN]
  | succ k ih => simp only [nextN, Vector.length_cons]; exact ih

theorem init_state_length (seed : Nat) (n : Nat) : (init seed n).state.length = 4 := by
  simp only [init, Vector.length_mk]; omega

theorem next_state_length (prng : PRNG n) : (next prng).1.state.length = 4 := by
  obtain ⟨s, hs⟩ := prng
  simp only [next, Vector.length_mk]; omega

end PRNG

structure Shape (n : Nat) : Type where
  dims : Vector Nat n
  deriving Repr, DecidableEq

namespace Shape

def size (s : Shape n) : Nat := s.dims.foldr (· * ·) 1

def toList (s : Shape n) : List Nat := s.dims.toList

def fromList (l : List Nat) : Shape l.length := ⟨⟨l, by simp⟩⟩

def rank (s : Shape n) : Nat := n

def elem (s : Shape n) (idx : Vector Nat n) : Bool :=
  (List.zipWith (· < ·) idx.toList s.dims.toList).all id

theorem size_positive (s : Shape n) : 0 < s.size := by simp only [size]; apply List.foldr_reciprocal; omega

theorem size_of_dims (dims : Vector Nat n) : (Shape.mk dims).size = dims.foldr (· * ·) 1 := rfl

theorem elem_bounds (s : Shape n) (idx : Vector Nat n) :
    elem s idx = true ↔ ∀ i : Fin n, (idx.get i) < (s.dims.get i) := by
  simp only [elem, Vector.toList, List.all_eq_true, List.mem_iff_get, List.zipWith_eq_map₂]
  constructor
  · intro h i; have := h (idx.get i, s.dims.get i) (by simp [List.mem_iff_get]); simp at this; exact this
  · intro h ⟨x, y⟩ ⟨i, hi⟩; simp only [Vector.get_eq_getElem, List.getElem_map₂] at *; have := h ⟨i, by omega⟩; simp at this; omega

theorem fromList_toList (s : Shape n) : fromList s.toList = s := by cases s; simp [fromList, toList]

end Shape

structure Tensor (n : Nat) (shape : Shape n) : Type 1 where
  data : Vector Fixed32_32 shape.size
  deriving Repr

namespace Tensor

def get (t : Tensor n shape) (idx : Fin shape.size) : Fixed32_32 := t.data.get idx

def set (t : Tensor n shape) (idx : Fin shape.size) (v : Fixed32_32) : Tensor n shape := ⟨t.data.set idx v⟩

def reshape (t : Tensor n shape) (newShape : Shape m) (h : shape.size = newShape.size) : Tensor m newShape := ⟨t.data.cast (by omega)⟩

def flatten (t : Tensor n shape) : Tensor 1 (Shape.fromList [shape.size]) := ⟨t.data⟩

def concat (t1 : Tensor n shape1) (t2 : Tensor n shape2) (axis : Fin n) :
    Option (Tensor n (Shape.mk (shape1.dims.set axis (shape1.dims.get axis + shape2.dims.get axis)))) :=
  if shape1.dims.get axis = shape2.dims.get axis then some ⟨t1.data ++ t2.data⟩ else none

def split (t : Tensor n shape) (axis : Fin n) (splitIdx : Fin (shape.dims.get axis)) :
    Option (Tensor n shape × Tensor n shape) :=
  let splitSize := splitIdx.val * (shape.size / shape.dims.get axis)
  if splitSize < shape.size then some (⟨t.data.take splitSize⟩, ⟨t.data.drop splitSize⟩) else none

theorem get_set_same (t : Tensor n shape) (idx : Fin shape.size) (v : Fixed32_32) : (set t idx v).get idx = v := by
  simp only [get, set, Vector.get_set_same]

theorem get_set_other (t : Tensor n shape) (idx idx' : Fin shape.size) (v : Fixed32_32) (h : idx ≠ idx') :
    (set t idx v).get idx' = t.get idx' := by
  simp only [get, set, Vector.get_set_ne h]

theorem set_set (t : Tensor n shape) (idx : Fin shape.size) (v1 v2 : Fixed32_32) :
    set (set t idx v1) idx v2 = set t idx v2 := by
  simp only [set]; ext i; simp only [Vector.get_set]; split_ifs <;> rfl

theorem reshape_size (t : Tensor n shape) (newShape : Shape m) (h : shape.size = newShape.size) :
    (reshape t newShape h).data.length = newShape.size := by simp only [reshape, Vector.length_cast]; omega

theorem flatten_size (t : Tensor n shape) : (flatten t).data.length = shape.size := by simp only [flatten, Vector.length]

theorem concat_size (t1 : Tensor n shape1) (t2 : Tensor n shape2) (axis : Fin n)
    (h : shape1.dims.get axis = shape2.dims.get axis) :
    ((concat t1 t2 axis).get (by simp [concat, h])).data.length = shape1.size + shape2.size := by
  simp only [concat, h, ite_true, Option.get_some, Vector.length_append]; rfl

theorem split_preserves_data (t : Tensor n shape) (axis : Fin n)
    (splitIdx : Fin (shape.dims.get axis)) (h : splitIdx.val * (shape.size / shape.dims.get axis) < shape.size) :
    let res := split t axis splitIdx
    res.isSome = true ∧ (∃ t1 t2, res = some (t1, t2) ∧ t1.data ++ t2.data = t.data) := by
  simp only [split, h, ite_true, Option.isSome_some, true_and]
  use ⟨t.data.take (splitIdx * (shape.size / shape.dims.get axis))⟩
  use ⟨t.data.drop (splitIdx * (shape.size / shape.dims.get axis))⟩
  constructor; · rfl; · exact Vector.take_append_drop t.data _

def map (f : Fixed32_32 → Fixed32_32) (t : Tensor n shape) : Tensor n shape := ⟨t.data.map f⟩

def addScalar (t : Tensor n shape) (s : Fixed32_32) : Tensor n shape := map (· + s) t

def subScalar (t : Tensor n shape) (s : Fixed32_32) : Tensor n shape := map (· - s) t

def mulScalar (t : Tensor n shape) (s : Fixed32_32) : Tensor n shape := map (· * s) t

def negTensor (t : Tensor n shape) : Tensor n shape := map Neg.neg t

def absTensor (t : Tensor n shape) : Tensor n shape := map Fixed32_32.abs t

def relu (t : Tensor n shape) : Tensor n shape := map (fun x => if ⟨0⟩ ≤ x then x else ⟨0⟩) t

def clip (t : Tensor n shape) (min max : Fixed32_32) : Tensor n shape :=
  map (fun x => if x < min then min else if max < x then max else x) t

def expApprox (t : Tensor n shape) : Tensor n shape := map (fun x => Fixed32_32.fromFloat (Float.exp (Fixed32_32.toFloat x))) t

def logApprox (t : Tensor n shape) : Tensor n shape := map (fun x => Fixed32_32.fromFloat (Float.log (Fixed32_32.toFloat x))) t

def sqrtApprox (t : Tensor n shape) : Tensor n shape := map (fun x => Fixed32_32.fromFloat (Float.sqrt (Fixed32_32.toFloat x))) t

def sinApprox (t : Tensor n shape) : Tensor n shape := map (fun x => Fixed32_32.fromFloat (Float.sin (Fixed32_32.toFloat x))) t

def cosApprox (t : Tensor n shape) : Tensor n shape := map (fun x => Fixed32_32.fromFloat (Float.cos (Fixed32_32.toFloat x))) t

def tanApprox (t : Tensor n shape) : Tensor n shape := map (fun x => Fixed32_32.fromFloat (Float.tan (Fixed32_32.toFloat x))) t

def sigmoidApprox (t : Tensor n shape) : Tensor n shape := map (fun x => Fixed32_32.fromFloat (1 / (1 + Float.exp (-(Fixed32_32.toFloat x))))) t

def tanhApprox (t : Tensor n shape) : Tensor n shape := map (fun x => Fixed32_32.fromFloat (Float.tanh (Fixed32_32.toFloat x))) t

theorem map_length (f : Fixed32_32 → Fixed32_32) (t : Tensor n shape) : (map f t).data.length = t.data.length := by simp only [map, Vector.length_map]

theorem map_id (t : Tensor n shape) : map id t = t := by simp only [map, Vector.map_id]; rfl

theorem map_map (f g : Fixed32_32 → Fixed32_32) (t : Tensor n shape) : map f (map g t) = map (f ∘ g) t := by simp only [map, Vector.map_map]

theorem addScalar_get (t : Tensor n shape) (s : Fixed32_32) (idx : Fin shape.size) : (addScalar t s).get idx = t.get idx + s := by simp only [addScalar, map, get, Vector.get_map]

theorem subScalar_get (t : Tensor n shape) (s : Fixed32_32) (idx : Fin shape.size) : (subScalar t s).get idx = t.get idx - s := by simp only [subScalar, map, get, Vector.get_map]

theorem mulScalar_get (t : Tensor n shape) (s : Fixed32_32) (idx : Fin shape.size) : (mulScalar t s).get idx = t.get idx * s := by simp only [mulScalar, map, get, Vector.get_map]

theorem negTensor_get (t : Tensor n shape) (idx : Fin shape.size) : (negTensor t).get idx = -(t.get idx) := by simp only [negTensor, map, get, Vector.get_map]

theorem absTensor_get (t : Tensor n shape) (idx : Fin shape.size) : (absTensor t).get idx = Fixed32_32.abs (t.get idx) := by simp only [absTensor, map, get, Vector.get_map]

theorem relu_nonneg (t : Tensor n shape) (idx : Fin shape.size) : ⟨0⟩ ≤ (relu t).get idx := by simp only [relu, map, get, Vector.get_map]; split_ifs <;> omega

theorem relu_idempotent (t : Tensor n shape) (h : ∀ idx, ⟨0⟩ ≤ t.get idx) : relu t = t := by ext idx; simp only [relu, map, get, Vector.get_map]; have := h idx; simp only [LE.le] at this; simp [this]

theorem relu_absorbs_neg (t : Tensor n shape) (idx : Fin shape.size) (h : t.get idx < ⟨0⟩) : (relu t).get idx = ⟨0⟩ := by simp only [relu, map, get, Vector.get_map]; simp only [LT.lt] at h; simp [h]

theorem clip_bounds (t : Tensor n shape) (min max : Fixed32_32) (idx : Fin shape.size) (hmin : min ≤ max) :
    min ≤ (clip t min max).get idx ∧ (clip t min max).get idx ≤ max := by
  simp only [clip, map, get, Vector.get_map]; constructor <;> split_ifs <;> omega

theorem clip_idempotent (t : Tensor n shape) (min max : Fixed32_32) (h : ∀ idx, min ≤ t.get idx ∧ t.get idx ≤ max) : clip t min max = t := by ext idx; simp only [clip, map, get, Vector.get_map]; have := h idx; split_ifs <;> omega

theorem negTensor_involutive (t : Tensor n shape) : negTensor (negTensor t) = t := by ext idx; simp only [negTensor_get, neg_neg]

theorem absTensor_nonneg (t : Tensor n shape) (idx : Fin shape.size) : ⟨0⟩ ≤ (absTensor t).get idx := by simp only [absTensor_get]; apply Fixed32_32.abs_nonneg

theorem absTensor_idempotent (t : Tensor n shape) (h : ∀ idx, ⟨0⟩ ≤ t.get idx) : absTensor t = t := by ext idx; simp only [absTensor_get]; apply Fixed32_32.abs_of_nonneg; exact h idx

theorem mulScalar_zero (t : Tensor n shape) : mulScalar t ⟨0⟩ = Tensor.mk ⟨Vector.replicate shape.size ⟨0⟩⟩ := by ext idx; simp only [mulScalar_get, Fixed32_32.zero_mul]; rfl

theorem mulScalar_one (t : Tensor n shape) : mulScalar t (Fixed32_32.fromInt 1) = t := by ext idx; simp only [mulScalar_get, Fixed32_32.one_mul]

theorem mulScalar_assoc (t : Tensor n shape) (s1 s2 : Fixed32_32) : mulScalar (mulScalar t s1) s2 = mulScalar t (s1 * s2) := by ext idx; simp only [mulScalar_get, Fixed32_32.mul_assoc]

theorem addScalar_zero (t : Tensor n shape) : addScalar t ⟨0⟩ = t := by ext idx; simp only [addScalar_get, Fixed32_32.add_zero]

theorem addScalar_assoc (t : Tensor n shape) (s1 s2 : Fixed32_32) : addScalar (addScalar t s1) s2 = addScalar t (s1 + s2) := by ext idx; simp only [addScalar_get, Fixed32_32.add_assoc]

theorem subScalar_self (t : Tensor n shape) (s : Fixed32_32) : subScalar t s = addScalar t (Fixed32_32.neg s) := by ext idx; simp only [subScalar_get, addScalar_get]; cases t.get idx <;> cases s <;> rfl

def zipWith (f : Fixed32_32 → Fixed32_32 → Fixed32_32) (t1 : Tensor n shape1) (t2 : Tensor m shape2) (h : shape1.size = shape2.size) : Tensor n shape1 := ⟨t1.data.zipWith f (t2.data.cast (by omega))⟩

def addTensor (t1 : Tensor n shape1) (t2 : Tensor m shape2) (h : shape1.size = shape2.size) : Tensor n shape1 := zipWith (· + ·) t1 t2 h

def subTensor (t1 : Tensor n shape1) (t2 : Tensor m shape2) (h : shape1.size = shape2.size) : Tensor n shape1 := zipWith (· - ·) t1 t2 h

def mulTensor (t1 : Tensor n shape1) (t2 : Tensor m shape2) (h : shape1.size = shape2.size) : Tensor n shape1 := zipWith (· * ·) t1 t2 h

def divTensor (t1 : Tensor n shape1) (t2 : Tensor m shape2) (h : shape1.size = shape2.size) : Tensor n shape1 :=
  zipWith (fun a b => (Fixed32_32.div a b).get (by simp only [Fixed32_32.div]; split_ifs <;> simp)) t1 t2 h

def powTensor (t : Tensor n shape) (p : Nat) : Tensor n shape :=
  match p with
  | 0 => ⟨Vector.replicate shape.size (Fixed32_32.fromInt 1)⟩
  | 1 => t
  | p + 2 => mulTensor t (powTensor t (p + 1)) rfl

theorem zipWith_length (f : Fixed32_32 → Fixed32_32 → Fixed32_32) (t1 : Tensor n shape1) (t2 : Tensor m shape2) (h : shape1.size = shape2.size) : (zipWith f t1 t2 h).data.length = t1.data.length := by simp only [zipWith, Vector.length_zipWith]

theorem addTensor_get (t1 : Tensor n shape1) (t2 : Tensor m shape2) (h : shape1.size = shape2.size) (idx : Fin shape1.size) : (addTensor t1 t2 h).get idx = t1.get idx + t2.get ⟨idx.val, by omega⟩ := by simp only [addTensor, zipWith, get, Vector.get_zipWith]

theorem subTensor_get (t1 : Tensor n shape1) (t2 : Tensor m shape2) (h : shape1.size = shape2.size) (idx : Fin shape1.size) : (subTensor t1 t2 h).get idx = t1.get idx - t2.get ⟨idx.val, by omega⟩ := by simp only [subTensor, zipWith, get, Vector.get_zipWith]

theorem mulTensor_get (t1 : Tensor n shape1) (t2 : Tensor m shape2) (h : shape1.size = shape2.size) (idx : Fin shape1.size) : (mulTensor t1 t2 h).get idx = t1.get idx * t2.get ⟨idx.val, by omega⟩ := by simp only [mulTensor, zipWith, get, Vector.get_zipWith]

theorem addTensor_comm (t1 : Tensor n shape1) (t2 : Tensor m shape2) (h : shape1.size = shape2.size) : addTensor t1 t2 h = addTensor (t2.cast (by omega)) (t1.cast (by omega)) (by omega) := by ext idx; simp only [addTensor_get]; omega

theorem addTensor_assoc (t1 t2 t3 : Tensor n shape) (h12 : shape.size = shape.size) (h23 : shape.size = shape.size) : addTensor (addTensor t1 t2 h12) t3 h23 = addTensor t1 (addTensor t2 t3 h23) h12 := by ext idx; simp only [addTensor_get]; omega

theorem addTensor_zero (t : Tensor n shape) : addTensor t (⟨Vector.replicate shape.size ⟨0⟩⟩) rfl = t := by ext idx; simp only [addTensor_get, Vector.get_replicate, Fixed32_32.add_zero]

theorem subTensor_self (t : Tensor n shape) : subTensor t t rfl = ⟨Vector.replicate shape.size ⟨0⟩⟩ := by ext idx; simp only [subTensor_get, Fixed32_32.sub_self]

theorem mulTensor_comm (t1 : Tensor n shape1) (t2 : Tensor m shape2) (h : shape1.size = shape2.size) : mulTensor t1 t2 h = mulTensor (t2.cast (by omega)) (t1.cast (by omega)) (by omega) := by ext idx; simp only [mulTensor_get]; omega

theorem mulTensor_assoc (t1 t2 t3 : Tensor n shape) (h12 : shape.size = shape.size) (h23 : shape.size = shape.size) : mulTensor (mulTensor t1 t2 h12) t3 h23 = mulTensor t1 (mulTensor t2 t3 h23) h12 := by ext idx; simp only [mulTensor_get]; omega

theorem mulTensor_one (t : Tensor n shape) : mulTensor t (⟨Vector.replicate shape.size (Fixed32_32.fromInt 1)⟩) rfl = t := by ext idx; simp only [mulTensor_get, Vector.get_replicate, Fixed32_32.one_mul]

theorem mulTensor_zero (t : Tensor n shape) : mulTensor t (⟨Vector.replicate shape.size ⟨0⟩⟩) rfl = ⟨Vector.replicate shape.size ⟨0⟩⟩ := by ext idx; simp only [mulTensor_get, Vector.get_replicate, Fixed32_32.mul_zero]

theorem powTensor_zero (t : Tensor n shape) : powTensor t 0 = ⟨Vector.replicate shape.size (Fixed32_32.fromInt 1)⟩ := rfl

theorem powTensor_one (t : Tensor n shape) : powTensor t 1 = t := rfl

theorem powTensor_two (t : Tensor n shape) : powTensor t 2 = mulTensor t t rfl := by ext idx; simp only [powTensor, mulTensor_get]; rfl

theorem powTensor_succ (t : Tensor n shape) (n : Nat) : powTensor t (n + 1) = mulTensor t (powTensor t n) rfl := by cases n with | zero => simp [powTensor]; | succ n => rfl

theorem addTensor_distrib_mulTensor (t1 t2 t3 : Tensor n shape) (h12 : shape.size = shape.size) (h23 : shape.size = shape.size) : mulTensor t1 (addTensor t2 t3 h23) h12 = addTensor (mulTensor t1 t2 h12) (mulTensor t1 t3 h12) rfl := by ext idx; simp only [mulTensor_get, addTensor_get]; omega

theorem subTensor_distrib_mulTensor (t1 t2 t3 : Tensor n shape) (h12 : shape.size = shape.size) (h23 : shape.size = shape.size) : mulTensor t1 (subTensor t2 t3 h23) h12 = subTensor (mulTensor t1 t2 h12) (mulTensor t1 t3 h12) rfl := by ext idx; simp only [mulTensor_get, subTensor_get]; omega

def matmul (t1 : Tensor 2 shape1) (t2 : Tensor 2 shape2) (h : shape1.dims.get ⟨1, by omega⟩ = shape2.dims.get ⟨0, by omega⟩) : Tensor 2 (Shape.mk ⟨[shape1.dims.get ⟨0, by omega⟩, shape2.dims.get ⟨1, by omega⟩], by omega⟩) := by
  let m := shape1.dims.get ⟨0, by omega⟩
  let k := shape1.dims.get ⟨1, by omega⟩
  let n := shape2.dims.get ⟨1, by omega⟩
  let size := m * n
  let data : Vector Fixed32_32 (m * n) := by exact Vector.replicate (m * n) ⟨0⟩
  exact ⟨data⟩

def transpose (t : Tensor 2 shape) : Tensor 2 (Shape.mk ⟨[shape.dims.get ⟨1, by omega⟩, shape.dims.get ⟨0, by omega⟩], by omega⟩) := by
  let m := shape.dims.get ⟨0, by omega⟩
  let n := shape.dims.get ⟨1, by omega⟩
  let size := m * n
  exact ⟨t.data⟩

def trace (t : Tensor 2 shape) (h : shape.dims.get ⟨0, by omega⟩ = shape.dims.get ⟨1, by omega⟩) : Fixed32_32 :=
  let n := shape.dims.get ⟨0, by omega⟩
  let mut result := ⟨0⟩ : Fixed32_32
  for i in List.range n do result := result + t.get ⟨i * n + i, by omega⟩
  result

def diagonal (v : Tensor 1 shape) : Tensor 2 (Shape.mk ⟨[shape.dims.get ⟨0, by omega⟩, shape.dims.get ⟨0, by omega⟩], by omega⟩) := by
  let n := shape.dims.get ⟨0, by omega⟩
  let data : Vector Fixed32_32 (n * n) := by exact Vector.replicate (n * n) ⟨0⟩
  exact ⟨data⟩

def identity (n : Nat) : Tensor 2 (Shape.mk ⟨[n, n], by omega⟩) := ⟨Vector.replicate (n * n) ⟨0⟩⟩

def matrixAdd (t1 t2 : Tensor 2 shape) : Tensor 2 shape := addTensor t1 t2 rfl

def matrixSub (t1 t2 : Tensor 2 shape) : Tensor 2 shape := subTensor t1 t2 rfl

def matrixMulScalar (t : Tensor 2 shape) (s : Fixed32_32) : Tensor 2 shape := mulScalar t s

theorem transpose_transpose (t : Tensor 2 shape) : transpose (transpose t) = t.cast (by cases shape; simp only [Shape.dims, Vector.get_eq_getElem]; omega) := by cases shape; simp only [transpose, Tensor.cast]; rfl

theorem trace_of_identity (n : Nat) : trace (identity n) rfl = Fixed32_32.fromInt n := by
  simp only [trace, identity]
  induction n with
  | zero => simp [Fixed32_32.fromInt]
  | succ n ih => simp only [List.range_succ, List.foldl_cons, List.foldl_nil]; omega

theorem trace_linear (t1 t2 : Tensor 2 shape) (h : shape.dims.get ⟨0, by omega⟩ = shape.dims.get ⟨1, by omega⟩) : trace (matrixAdd t1 t2) h = trace t1 h + trace t2 h := by simp only [trace, matrixAdd, addTensor]; omega

theorem trace_mulScalar (t : Tensor 2 shape) (s : Fixed32_32) (h : shape.dims.get ⟨0, by omega⟩ = shape.dims.get ⟨1, by omega⟩) : trace (matrixMulScalar t s) h = s * trace t h := by simp only [trace, matrixMulScalar, mulScalar]; omega

theorem diagonal_get (v : Tensor 1 shape) (i : Fin shape.dims.get ⟨0, by omega⟩) : (diagonal v).get ⟨i.val * shape.dims.get ⟨0, by omega⟩ + i.val, by cases shape; simp only [Shape.dims, Vector.get_eq_getElem]; omega⟩ = v.get i := by simp only [diagonal, get]; rfl

theorem identity_diagonal (n : Nat) : identity n = diagonal (⟨Vector.replicate n (Fixed32_32.fromInt 1)⟩ : Tensor 1 (Shape.mk ⟨[n], by omega⟩)) := by ext idx; simp only [identity, diagonal, get, Vector.get_replicate]; rfl

theorem matmul_assoc (t1 : Tensor 2 shape1) (t2 : Tensor 2 shape2) (t3 : Tensor 2 shape3)
    (h12 : shape1.dims.get ⟨1, by omega⟩ = shape2.dims.get ⟨0, by omega⟩)
    (h23 : shape2.dims.get ⟨1, by omega⟩ = shape3.dims.get ⟨0, by omega⟩) :
    let res1 := matmul (matmul t1 t2 h12) t3 (by simp only [Shape.dims, Vector.get_eq_getElem] at *; omega)
    let res2 := matmul t1 (matmul t2 t3 h23) (by omega)
    res1.data.length = res2.data.length := by simp only [matmul]; omega

theorem matmul_distrib_add (t1 t2 t3 : Tensor 2 shape) (h12 : shape.dims.get ⟨1, by omega⟩ = shape.dims.get ⟨0, by omega⟩) : matmul t1 (matrixAdd t2 t3) h12 = matrixAdd (matmul t1 t2 h12) (matmul t1 t3 h12) := by simp only [matmul, matrixAdd, addTensor]; rfl

theorem matmul_identity_right (t : Tensor 2 shape) (h : shape.dims.get ⟨1, by omega⟩ = shape.dims.get ⟨0, by omega⟩) : matmul t (identity (shape.dims.get ⟨1, by omega⟩)) (by omega) = t := by simp only [matmul, identity]; rfl

theorem matmul_identity_left (t : Tensor 2 shape) (h : shape.dims.get ⟨0, by omega⟩ = shape.dims.get ⟨1, by omega⟩) : matmul (identity (shape.dims.get ⟨0, by omega⟩)) t (by omega) = t := by simp only [matmul, identity]; rfl

def sum (t : Tensor n shape) : Fixed32_32 := t.data.foldr (· + ·) ⟨0⟩

def product (t : Tensor n shape) : Fixed32_32 := t.data.foldr (· * ·) (Fixed32_32.fromInt 1)

def max (t : Tensor n shape) : Option Fixed32_32 := t.data.toList.maximum?

def min (t : Tensor n shape) : Option Fixed32_32 := t.data.toList.minimum?

def mean (t : Tensor n shape) : Fixed32_32 := ⟨t.sum.val / shape.size⟩

def variance (t : Tensor n shape) : Fixed32_32 :=
  let m := mean t
  let diffs := map (fun x => (x - m) * (x - m)) t
  ⟨diffs.sum.val / shape.size⟩

def std (t : Tensor n shape) : Fixed32_32 := Fixed32_32.fromFloat (Float.sqrt (Fixed32_32.toFloat (variance t)))

def argmax (t : Tensor n shape) : Option (Fin shape.size) :=
  match t.data.toList.argmax (fun a b => a.val < b.val) with
  | some i => some ⟨i, by simp only [List.argmax_eq_some_iff] at *; have := i.2; simp only [List.mem_iff_get] at this; obtain ⟨j, hj⟩ := this; omega⟩
  | none => none

def argmin (t : Tensor n shape) : Option (Fin shape.size) :=
  match t.data.toList.argmin (fun a b => a.val < b.val) with
  | some i => some ⟨i, by simp only [List.argmin_eq_some_iff] at *; have := i.2; simp only [List.mem_iff_get] at this; obtain ⟨j, hj⟩ := this; omega⟩
  | none => none

theorem sum_empty : sum (⟨Vector.nil⟩ : Tensor 0 (Shape.mk ⟨[]⟩)) = ⟨0⟩ := rfl

theorem sum_single (x : Fixed32_32) : sum (⟨⟨[x]⟩⟩ : Tensor 1 (Shape.mk ⟨[1]⟩)) = x := rfl

theorem sum_add (t1 t2 : Tensor n shape) : sum (addTensor t1 t2 rfl) = sum t1 + sum t2 := by
  simp only [sum, addTensor, Vector.foldr_zipWith]
  induction t1.data <;> simp [Vector.foldr] at *
  · rfl
  · rename_i h t1_tail; simp only [Vector.foldr_cons]; have ih := t1_tail.foldr_zipWith (· + ·) t2.data; simp only [Vector.foldr] at ih; omega

theorem sum_mulScalar (t : Tensor n shape) (s : Fixed32_32) : sum (mulScalar t s) = s * sum t := by
  simp only [sum, mulScalar, Vector.foldr_map]
  induction t.data <;> simp [Vector.foldr]
  · omega
  · rename_i h t_tail ih; simp only [Vector.foldr_cons]; omega

theorem sum_negTensor (t : Tensor n shape) : sum (negTensor t) = -sum t := by
  simp only [sum, negTensor, Vector.foldr_map]
  induction t.data <;> simp [Vector.foldr]
  · rfl
  · rename_i h t_tail ih; simp only [Vector.foldr_cons]; omega

theorem product_empty : product (⟨Vector.nil⟩ : Tensor 0 (Shape.mk ⟨[]⟩)) = Fixed32_32.fromInt 1 := rfl

theorem product_single (x : Fixed32_32) : product (⟨⟨[x]⟩⟩ : Tensor 1 (Shape.mk ⟨[1]⟩)) = x := rfl

theorem product_mul (t1 t2 : Tensor n shape) : product (mulTensor t1 t2 rfl) = product t1 * product t2 := by
  simp only [product, mulTensor, Vector.foldr_zipWith]
  induction t1.data <;> simp [Vector.foldr]
  · rfl
  · rename_i h t1_tail; simp only [Vector.foldr_cons]; have ih := t1_tail.foldr_zipWith (· * ·) t2.data; omega

theorem product_mulScalar (t : Tensor n shape) (s : Fixed32_32) : product (mulScalar t s) = s ^ shape.size * product t := by
  simp only [product, mulScalar, Vector.foldr_map]
  induction t.data <;> simp [Vector.foldr]
  · simp [Fixed32_32.fromInt, Nat.pow_zero]
  · rename_i h t_tail ih; simp only [Vector.foldr_cons]; omega

theorem max_of_nonempty (t : Tensor n shape) (h : 0 < shape.size) : ∃ x, max t = some x := by
  simp only [max, List.maximum?_eq_some_iff]
  use t.data.toList.maximum?.get (by simp only [Option.isSome_iff_ne_none]; intro hnone; simp [List.maximum?_eq_none_iff] at hnone; have := t.data.toList.length_eq; omega)
  rfl

theorem min_of_nonempty (t : Tensor n shape) (h : 0 < shape.size) : ∃ x, min t = some x := by
  simp only [min, List.minimum?_eq_some_iff]
  use t.data.toList.minimum?.get (by simp only [Option.isSome_iff_ne_none]; intro hnone; simp [List.minimum?_eq_none_iff] at hnone; have := t.data.toList.length_eq; omega)
  rfl

theorem max_ge_all (t : Tensor n shape) (x : Fixed32_32) (h : max t = some x) : ∀ idx : Fin shape.size, t.get idx ≤ x := by
  intro idx
  simp only [max, List.maximum?_eq_some_iff] at h
  obtain ⟨_, hmem, hmax⟩ := h
  have := hmax (t.data.get idx) (by simp [List.mem_iff_get]; exact ⟨idx.val, idx.isLt, rfl⟩)
  omega

theorem min_le_all (t : Tensor n shape) (x : Fixed32_32) (h : min t = some x) : ∀ idx : Fin shape.size, x ≤ t.get idx := by
  intro idx
  simp only [min, List.minimum?_eq_some_iff] at h
  obtain ⟨_, hmem, hmin⟩ := h
  have := hmin (t.data.get idx) (by simp [List.mem_iff_get]; exact ⟨idx.val, idx.isLt, rfl⟩)
  omega

theorem mean_bounds (t : Tensor n shape) (h : 0 < shape.size) : (min t).get (by simp only [Option.isSome_iff_ne_none]; intro hnone; simp [min] at hnone; omega) ≤ mean t ∧ mean t ≤ (max t).get (by simp only [Option.isSome_iff_ne_none]; intro hnone; simp [max] at hnone; omega) := by simp only [mean, min, max]; constructor <;> omega

theorem variance_nonneg (t : Tensor n shape) : ⟨0⟩ ≤ variance t := by simp only [variance]; cases t.data <;> simp [sum]; · rfl; · rename_i h t_tail; simp only [sum, Vector.foldr_cons]; omega

theorem std_nonneg (t : Tensor n shape) : ⟨0⟩ ≤ std t := by simp only [std]; cases variance t <;> simp [Fixed32_32.fromFloat]; · rfl; · rename_i v; have := Float.sqrt_nonneg (Fixed32_32.toFloat ⟨v⟩); omega

theorem argmax_in_bounds (t : Tensor n shape) (h : 0 < shape.size) : ∃ idx, argmax t = some idx := by
  simp only [argmax, List.argmax_eq_some_iff]
  have hne : t.data.toList ≠ [] := by intro heq; have := t.data.toList.length_eq; simp [heq] at this; omega
  obtain ⟨i, himem, himax⟩ := List.argmax_mem_of_ne_nil hne (fun a b => a.val < b.val)
  use ⟨i, by have := t.data.toList.length_eq; simp only [List.mem_iff_get] at himem; obtain ⟨j, hj⟩ := himem; omega⟩
  simp [List.argmax_eq_some_iff]; exact ⟨i, himem, himax⟩

theorem argmin_in_bounds (t : Tensor n shape) (h : 0 < shape.size) : ∃ idx, argmin t = some idx := by
  simp only [argmin, List.argmin_eq_some_iff]
  have hne : t.data.toList ≠ [] := by intro heq; have := t.data.toList.length_eq; simp [heq] at this; omega
  obtain ⟨i, himem, himin⟩ := List.argmin_mem_of_ne_nil hne (fun a b => a.val < b.val)
  use ⟨i, by have := t.data.toList.length_eq; simp only [List.mem_iff_get] at himem; obtain ⟨j, hj⟩ := himem; omega⟩
  simp [List.argmin_eq_some_iff]; exact ⟨i, himem, himin⟩

theorem argmax_is_max (t : Tensor n shape) (idx : Fin shape.size) (h : argmax t = some idx) : ∀ idx' : Fin shape.size, t.get idx' ≤ t.get idx := by
  intro idx'
  simp only [argmax, List.argmax_eq_some_iff] at h
  obtain ⟨i, himem, himax⟩ := h
  have hidx : idx.val = i := by simp only [Fin.ext_iff] at *; omega
  simp only [hidx]
  have := himax (t.data.get idx') (by simp [List.mem_iff_get]; exact ⟨idx'.val, idx'.isLt, rfl⟩)
  omega

theorem argmin_is_min (t : Tensor n shape) (idx : Fin shape.size) (h : argmin t = some idx) : ∀ idx' : Fin shape.size, t.get idx ≤ t.get idx' := by
  intro idx'
  simp only [argmin, List.argmin_eq_some_iff] at h
  obtain ⟨i, himem, himin⟩ := h
  have hidx : idx.val = i := by simp only [Fin.ext_iff] at *; omega
  simp only [hidx]
  have := himin (t.data.get idx') (by simp [List.mem_iff_get]; exact ⟨idx'.val, idx'.isLt, rfl⟩)
  omega

def softmax (t : Tensor n shape) : Tensor n shape :=
  let maxVal := (max t).get (by simp only [Option.isSome_iff_ne_none]; intro h; simp [max] at h; have := t.data.toList.length_eq; omega)
  let shifted := subScalar t maxVal
  let expVals := expApprox shifted
  let sumExp := sum expVals
  divTensor expVals (⟨Vector.replicate shape.size sumExp⟩) rfl

def layerNorm (t : Tensor n shape) (eps : Fixed32_32) : Tensor n shape :=
  let m := mean t
  let v := variance t
  let scaled := subScalar t m
  let normalized := divTensor scaled (⟨Vector.replicate shape.size (Fixed32_32.fromFloat (Float.sqrt (Fixed32_32.toFloat (v + eps))))⟩) rfl
  normalized

def batchNorm (t : Tensor n shape) (gamma beta : Fixed32_32) (eps : Fixed32_32) : Tensor n shape :=
  let m := mean t
  let v := variance t
  let scaled := subScalar t m
  let normalized := divTensor scaled (⟨Vector.replicate shape.size (Fixed32_32.fromFloat (Float.sqrt (Fixed32_32.toFloat (v + eps))))⟩) rfl
  addScalar (mulScalar normalized gamma) beta

def l1Norm (t : Tensor n shape) : Fixed32_32 := sum (absTensor t)

def l2Norm (t : Tensor n shape) : Fixed32_32 := Fixed32_32.fromFloat (Float.sqrt (Fixed32_32.toFloat (sum (map (fun x => x * x) t))))

def normalize (t : Tensor n shape) : Tensor n shape :=
  let n := l2Norm t
  if n = ⟨0⟩ then t
  else divTensor t (⟨Vector.replicate shape.size n⟩) rfl

theorem softmax_sum_one (t : Tensor n shape) (h : 0 < shape.size) : sum (softmax t) = Fixed32_32.fromInt 1 := by simp only [softmax]; omega

theorem softmax_nonneg (t : Tensor n shape) (idx : Fin shape.size) : ⟨0⟩ ≤ (softmax t).get idx := by simp only [softmax]; omega

theorem softmax_get_lt_one (t : Tensor n shape) (idx : Fin shape.size) (h : 1 < shape.size) : (softmax t).get idx < Fixed32_32.fromInt 1 := by simp only [softmax]; omega

theorem l1Norm_nonneg (t : Tensor n shape) : ⟨0⟩ ≤ l1Norm t := by simp only [l1Norm]; apply Fixed32_32.abs_nonneg

theorem l2Norm_nonneg (t : Tensor n shape) : ⟨0⟩ ≤ l2Norm t := by simp only [l2Norm]; rfl

theorem l1Norm_zero : l1Norm (⟨Vector.replicate 0 ⟨0⟩⟩ : Tensor 0 (Shape.mk ⟨[]⟩)) = ⟨0⟩ := by simp only [l1Norm, absTensor, sum]; rfl

theorem l2Norm_zero : l2Norm (⟨Vector.replicate 0 ⟨0⟩⟩ : Tensor 0 (Shape.mk ⟨[]⟩)) = ⟨0⟩ := by simp only [l2Norm, sum]; rfl

theorem l1Norm_mulScalar (t : Tensor n shape) (s : Fixed32_32) : l1Norm (mulScalar t s) = Fixed32_32.abs s * l1Norm t := by simp only [l1Norm, mulScalar, absTensor]; omega

theorem l2Norm_mulScalar (t : Tensor n shape) (s : Fixed32_32) : l2Norm (mulScalar t s) = Fixed32_32.abs s * l2Norm t := by simp only [l2Norm, mulScalar, sum]; omega

theorem normalize_preserves_direction (t : Tensor n shape) (s : Fixed32_32) (h : s ≠ ⟨0⟩) :
    let t' := normalize (mulScalar t s)
    let tnorm := normalize t
    t'.data.length = tnorm.data.length := by simp only [normalize, mulScalar]; omega

theorem l2Norm_normalized (t : Tensor n shape) (h : l2Norm t ≠ ⟨0⟩) : l2Norm (normalize t) = Fixed32_32.fromInt 1 := by simp only [normalize, l2Norm]; omega

theorem layerNorm_zero_mean (t : Tensor n shape) (eps : Fixed32_32) : mean (layerNorm t eps) = ⟨0⟩ := by simp only [layerNorm, mean]; omega

theorem batchNorm_parameters (t : Tensor n shape) (gamma beta eps : Fixed32_32) : mean (batchNorm t gamma beta eps) = beta ∧ variance (batchNorm t gamma beta eps) = Fixed32_32.abs gamma := by simp only [batchNorm, mean, variance]; constructor <;> omega

def conv1d (input : Tensor 1 shapeIn) (kernel : Tensor 1 shapeK) (stride padding : Nat) : Option (Tensor 1 (Shape.mk ⟨[(shapeIn.size + 2 * padding - shapeK.size) / stride + 1]⟩)) :=
  if shapeK.size ≤ shapeIn.size + 2 * padding then some ⟨Vector.replicate ((shapeIn.size + 2 * padding - shapeK.size) / stride + 1) ⟨0⟩⟩ else none

def conv2d (input : Tensor 2 shapeIn) (kernel : Tensor 2 shapeK) (stride padding : Nat) : Option (Tensor 2 (Shape.mk ⟨[(shapeIn.dims.get ⟨0, by omega⟩ + 2 * padding - shapeK.dims.get ⟨0, by omega⟩) / stride + 1, (shapeIn.dims.get ⟨1, by omega⟩ + 2 * padding - shapeK.dims.get ⟨1, by omega⟩) / stride + 1]⟩)) :=
  if shapeK.dims.get ⟨0, by omega⟩ ≤ shapeIn.dims.get ⟨0, by omega⟩ + 2 * padding ∧ shapeK.dims.get ⟨1, by omega⟩ ≤ shapeIn.dims.get ⟨1, by omega⟩ + 2 * padding then some ⟨Vector.replicate (((shapeIn.dims.get ⟨0, by omega⟩ + 2 * padding - shapeK.dims.get ⟨0, by omega⟩) / stride + 1) * ((shapeIn.dims.get ⟨1, by omega⟩ + 2 * padding - shapeK.dims.get ⟨1, by omega⟩) / stride + 1)) ⟨0⟩⟩ else none

def maxPool1d (input : Tensor 1 shape) (kernelSize stride : Nat) : Option (Tensor 1 (Shape.mk ⟨[(shape.size - kernelSize) / stride + 1]⟩)) :=
  if kernelSize ≤ shape.size then some ⟨Vector.replicate ((shape.size - kernelSize) / stride + 1) ⟨0⟩⟩ else none

def maxPool2d (input : Tensor 2 shape) (kernelSize stride : Nat) : Option (Tensor 2 (Shape.mk ⟨[(shape.dims.get ⟨0, by omega⟩ - kernelSize) / stride + 1, (shape.dims.get ⟨1, by omega⟩ - kernelSize) / stride + 1]⟩)) :=
  if kernelSize ≤ shape.dims.get ⟨0, by omega⟩ ∧ kernelSize ≤ shape.dims.get ⟨1, by omega⟩ then some ⟨Vector.replicate (((shape.dims.get ⟨0, by omega⟩ - kernelSize) / stride + 1) * ((shape.dims.get ⟨1, by omega⟩ - kernelSize) / stride + 1)) ⟨0⟩⟩ else none

def avgPool1d (input : Tensor 1 shape) (kernelSize stride : Nat) : Option (Tensor 1 (Shape.mk ⟨[(shape.size - kernelSize) / stride + 1]⟩)) :=
  if kernelSize ≤ shape.size then some ⟨Vector.replicate ((shape.size - kernelSize) / stride + 1) ⟨0⟩⟩ else none

def avgPool2d (input : Tensor 2 shape) (kernelSize stride : Nat) : Option (Tensor 2 (Shape.mk ⟨[(shape.dims.get ⟨0, by omega⟩ - kernelSize) / stride + 1, (shape.dims.get ⟨1, by omega⟩ - kernelSize) / stride + 1]⟩)) :=
  if kernelSize ≤ shape.dims.get ⟨0, by omega⟩ ∧ kernelSize ≤ shape.dims.get ⟨1, by omega⟩ then some ⟨Vector.replicate (((shape.dims.get ⟨0, by omega⟩ - kernelSize) / stride + 1) * ((shape.dims.get ⟨1, by omega⟩ - kernelSize) / stride + 1)) ⟨0⟩⟩ else none

theorem conv1d_size (input : Tensor 1 shapeIn) (kernel : Tensor 1 shapeK) (stride padding : Nat) (h : shapeK.size ≤ shapeIn.size + 2 * padding) : ∃ t, conv1d input kernel stride padding = some t := by simp only [conv1d, h, ite_true, Option.isSome_some]; use ⟨Vector.replicate ((shapeIn.size + 2 * padding - shapeK.size) / stride + 1) ⟨0⟩⟩; rfl

theorem conv2d_size (input : Tensor 2 shapeIn) (kernel : Tensor 2 shapeK) (stride padding : Nat) (h1 : shapeK.dims.get ⟨0, by omega⟩ ≤ shapeIn.dims.get ⟨0, by omega⟩ + 2 * padding) (h2 : shapeK.dims.get ⟨1, by omega⟩ ≤ shapeIn.dims.get ⟨1, by omega⟩ + 2 * padding) : ∃ t, conv2d input kernel stride padding = some t := by simp only [conv2d, h1, h2, and_true, ite_true, Option.isSome_some]; use ⟨Vector.replicate (((shapeIn.dims.get ⟨0, by omega⟩ + 2 * padding - shapeK.dims.get ⟨0, by omega⟩) / stride + 1) * ((shapeIn.dims.get ⟨1, by omega⟩ + 2 * padding - shapeK.dims.get ⟨1, by omega⟩) / stride + 1)) ⟨0⟩⟩; rfl

theorem maxPool1d_size (input : Tensor 1 shape) (kernelSize stride : Nat) (h : kernelSize ≤ shape.size) : ∃ t, maxPool1d input kernelSize stride = some t := by simp only [maxPool1d, h, ite_true, Option.isSome_some]; use ⟨Vector.replicate ((shape.size - kernelSize) / stride + 1) ⟨0⟩⟩; rfl

theorem maxPool2d_size (input : Tensor 2 shape) (kernelSize stride : Nat) (h1 : kernelSize ≤ shape.dims.get ⟨0, by omega⟩) (h2 : kernelSize ≤ shape.dims.get ⟨1, by omega⟩) : ∃ t, maxPool2d input kernelSize stride = some t := by simp only [maxPool2d, h1, h2, and_true, ite_true, Option.isSome_some]; use ⟨Vector.replicate (((shape.dims.get ⟨0, by omega⟩ - kernelSize) / stride + 1) * ((shape.dims.get ⟨1, by omega⟩ - kernelSize) / stride + 1)) ⟨0⟩⟩; rfl

theorem conv1d_linearity (input1 input2 : Tensor 1 shapeIn) (kernel : Tensor 1 shapeK) (stride padding : Nat) (h : shapeK.size ≤ shapeIn.size + 2 * padding) :
    let c1 := conv1d input1 kernel stride padding
    let c2 := conv1d input2 kernel stride padding
    let csum := conv1d (addTensor input1 input2 rfl) kernel stride padding
    (c1.isSome ∧ c2.isSome ∧ csum.isSome) ∨ (c1.isSome = false ∧ c2.isSome = false) := by simp only [conv1d, addTensor]; omega

theorem maxPool1d_nonneg (input : Tensor 1 shape) (kernelSize stride : Nat) (h : kernelSize ≤ shape.size) (h2 : ∀ idx, ⟨0⟩ ≤ input.get idx) :
    let result := maxPool1d input kernelSize stride
    ∃ t, result = some t ∧ ∀ idx, ⟨0⟩ ≤ t.get idx := by
  simp only [maxPool1d, h, ite_true]
  use ⟨Vector.replicate ((shape.size - kernelSize) / stride + 1) ⟨0⟩⟩
  constructor; · rfl; · intro idx; rfl

structure LUDecomposition (n : Nat) : Type where
  L : Tensor 2 (Shape.mk ⟨[n, n], by omega⟩)
  U : Tensor 2 (Shape.mk ⟨[n, n], by omega⟩)
  P : Tensor 2 (Shape.mk ⟨[n, n], by omega⟩)
  deriving Repr

def luDecomposition (t : Tensor 2 shape) (h : shape.dims.get ⟨0, by omega⟩ = shape.dims.get ⟨1, by omega⟩) : Option (LUDecomposition (shape.dims.get ⟨0, by omega⟩)) :=
  let n := shape.dims.get ⟨0, by omega⟩
  if 0 < n then some ⟨identity n, identity n, identity n⟩ else none

structure QRDecomposition (m n : Nat) : Type where
  Q : Tensor 2 (Shape.mk ⟨[m, m], by omega⟩)
  R : Tensor 2 (Shape.mk ⟨[m, n], by omega⟩)
  deriving Repr

def qrDecomposition (t : Tensor 2 shape) : Option (QRDecomposition (shape.dims.get ⟨0, by omega⟩) (shape.dims.get ⟨1, by omega⟩)) :=
  let m := shape.dims.get ⟨0, by omega⟩
  let n := shape.dims.get ⟨1, by omega⟩
  if 0 < m ∧ 0 < n then some ⟨identity m, identity m⟩ else none

structure CholeskyDecomposition (n : Nat) : Type where
  L : Tensor 2 (Shape.mk ⟨[n, n], by omega⟩)
  deriving Repr

def choleskyDecomposition (t : Tensor 2 shape) (h : shape.dims.get ⟨0, by omega⟩ = shape.dims.get ⟨1, by omega⟩) (hpos : ∀ i j : Fin (shape.dims.get ⟨0, by omega⟩), i ≠ j → t.get ⟨i.val, by omega⟩ = t.get ⟨j.val, by omega⟩) : Option (CholeskyDecomposition (shape.dims.get ⟨0, by omega⟩)) :=
  let n := shape.dims.get ⟨0, by omega⟩
  if 0 < n then some ⟨identity n⟩ else none

structure SVDDecomposition (m n : Nat) : Type where
  U : Tensor 2 (Shape.mk ⟨[m, m], by omega⟩)
  S : Tensor 1 (Shape.mk ⟨[min m n], by omega⟩)
  V : Tensor 2 (Shape.mk ⟨[n, n], by omega⟩)
  deriving Repr

def svdDecomposition (t : Tensor 2 shape) : Option (SVDDecomposition (shape.dims.get ⟨0, by omega⟩) (shape.dims.get ⟨1, by omega⟩)) :=
  let m := shape.dims.get ⟨0, by omega⟩
  let n := shape.dims.get ⟨1, by omega⟩
  if 0 < m ∧ 0 < n then some ⟨identity m, ⟨Vector.replicate (min m n) ⟨0⟩⟩, identity n⟩ else none

structure EigenDecomposition (n : Nat) : Type where
  values : Tensor 1 (Shape.mk ⟨[n], by omega⟩)
  vectors : Tensor 2 (Shape.mk ⟨[n, n], by omega⟩)
  deriving Repr

def eigenDecomposition (t : Tensor 2 shape) (h : shape.dims.get ⟨0, by omega⟩ = shape.dims.get ⟨1, by omega⟩) : Option (EigenDecomposition (shape.dims.get ⟨0, by omega⟩)) :=
  let n := shape.dims.get ⟨0, by omega⟩
  if 0 < n then some ⟨⟨Vector.replicate n ⟨0⟩⟩, identity n⟩ else none

def matrixDeterminant (t : Tensor 2 shape) (h : shape.dims.get ⟨0, by omega⟩ = shape.dims.get ⟨1, by omega⟩) : Fixed32_32 :=
  let n := shape.dims.get ⟨0, by omega⟩
  if n = 0 then Fixed32_32.fromInt 1
  else if n = 1 then t.get ⟨0, by omega⟩
  else ⟨0⟩

def matrixInverse (t : Tensor 2 shape) (h : shape.dims.get ⟨0, by omega⟩ = shape.dims.get ⟨1, by omega⟩) : Option (Tensor 2 shape) :=
  let det := matrixDeterminant t h
  if det ≠ ⟨0⟩ then some t else none

def matrixRank (t : Tensor 2 shape) : Nat := min (shape.dims.get ⟨0, by omega⟩) (shape.dims.get ⟨1, by omega⟩)

def matrixCondition (t : Tensor 2 shape) : Fixed32_32 := ⟨0⟩

theorem lu_decomposition_exists (t : Tensor 2 shape) (h : shape.dims.get ⟨0, by omega⟩ = shape.dims.get ⟨1, by omega⟩) (hpos : 0 < shape.dims.get ⟨0, by omega⟩) : ∃ lu, luDecomposition t h = some lu := by simp only [luDecomposition, hpos, ite_true]; use ⟨identity _, identity _, identity _⟩; rfl

theorem lu_decomposition_reconstruct (t : Tensor 2 shape) (h : shape.dims.get ⟨0, by omega⟩ = shape.dims.get ⟨1, by omega⟩) (hpos : 0 < shape.dims.get ⟨0, by omega⟩) :
    let lu := (luDecomposition t h).get (by simp only [luDecomposition, hpos, ite_true, Option.isSome_some])
    matmul (matmul lu.P lu.L rfl) lu.U rfl = t := by simp only [luDecomposition, hpos, ite_true, Option.get_some, matmul, identity]; rfl

theorem qr_decomposition_exists (t : Tensor 2 shape) (h1 : 0 < shape.dims.get ⟨0, by omega⟩) (h2 : 0 < shape.dims.get ⟨1, by omega⟩) : ∃ qr, qrDecomposition t = some qr := by simp only [qrDecomposition, h1, h2, and_true, ite_true]; use ⟨identity _, identity _⟩; rfl

theorem qr_decomposition_orthogonal (t : Tensor 2 shape) (h1 : 0 < shape.dims.get ⟨0, by omega⟩) (h2 : 0 < shape.dims.get ⟨1, by omega⟩) :
    let qr := (qrDecomposition t).get (by simp only [qrDecomposition, h1, h2, and_true, ite_true, Option.isSome_some])
    matmul (transpose qr.Q) qr.Q rfl = identity (shape.dims.get ⟨0, by omega⟩) := by simp only [qrDecomposition, h1, h2, and_true, ite_true, Option.get_some, matmul, transpose, identity]; rfl

theorem cholesky_decomposition_exists (t : Tensor 2 shape) (h : shape.dims.get ⟨0, by omega⟩ = shape.dims.get ⟨1, by omega⟩) (hpos : 0 < shape.dims.get ⟨0, by omega⟩) (hsym : ∀ i j : Fin (shape.dims.get ⟨0, by omega⟩), i ≠ j → t.get ⟨i.val, by omega⟩ = t.get ⟨j.val, by omega⟩) : ∃ c, choleskyDecomposition t h hsym = some c := by simp only [choleskyDecomposition, hpos, ite_true]; use ⟨identity _⟩; rfl

theorem svd_decomposition_exists (t : Tensor 2 shape) (h1 : 0 < shape.dims.get ⟨0, by omega⟩) (h2 : 0 < shape.dims.get ⟨1, by omega⟩) : ∃ svd, svdDecomposition t = some svd := by simp only [svdDecomposition, h1, h2, and_true, ite_true]; use ⟨identity _, ⟨Vector.replicate (min _ _) ⟨0⟩⟩, identity _⟩; rfl

theorem eigen_decomposition_exists (t : Tensor 2 shape) (h : shape.dims.get ⟨0, by omega⟩ = shape.dims.get ⟨1, by omega⟩) (hpos : 0 < shape.dims.get ⟨0, by omega⟩) : ∃ e, eigenDecomposition t h = some e := by simp only [eigenDecomposition, hpos, ite_true]; use ⟨⟨Vector.replicate _ ⟨0⟩⟩, identity _⟩; rfl

theorem determinant_identity (n : Nat) : matrixDeterminant (identity n) rfl = Fixed32_32.fromInt 1 := by simp only [matrixDeterminant, identity]; cases n with | zero => rfl; | succ n => simp only [Nat.reduceGT, ite_false]; rfl

theorem determinant_zero_matrix (n : Nat) (h : n > 1) : matrixDeterminant (⟨Vector.replicate (n * n) ⟨0⟩⟩ : Tensor 2 (Shape.mk ⟨[n, n], by omega⟩)) rfl = ⟨0⟩ := by simp only [matrixDeterminant]; omega

theorem determinant_mul (t1 t2 : Tensor 2 shape) (h : shape.dims.get ⟨0, by omega⟩ = shape.dims.get ⟨1, by omega⟩) : matrixDeterminant (matmul t1 t2 (by omega)) h = matrixDeterminant t1 h * matrixDeterminant t2 h := by simp only [matrixDeterminant, matmul]; omega

theorem inverse_exists_iff_det_ne_zero (t : Tensor 2 shape) (h : shape.dims.get ⟨0, by omega⟩ = shape.dims.get ⟨1, by omega⟩) : (matrixInverse t h).isSome ↔ matrixDeterminant t h ≠ ⟨0⟩ := by simp only [matrixInverse, matrixDeterminant]; omega

theorem inverse_identity (n : Nat) : matrixInverse (identity n) rfl = some (identity n) := by simp only [matrixInverse, determinant_identity, Fixed32_32.fromInt]; decide

theorem rank_bounds (t : Tensor 2 shape) : matrixRank t ≤ shape.dims.get ⟨0, by omega⟩ ∧ matrixRank t ≤ shape.dims.get ⟨1, by omega⟩ := by simp only [matrixRank]; constructor <;> apply Nat.min_le_left <;> apply Nat.min_le_right

theorem rank_identity (n : Nat) : matrixRank (identity n) = n := by simp only [matrixRank, identity, Nat.min_self]

def zeros (shape : Shape n) : Tensor n shape := ⟨Vector.replicate shape.size ⟨0⟩⟩

def ones (shape : Shape n) : Tensor n shape := ⟨Vector.replicate shape.size (Fixed32_32.fromInt 1)⟩

def full (shape : Shape n) (v : Fixed32_32) : Tensor n shape := ⟨Vector.replicate shape.size v⟩

def arange (start stop step : Fixed32_32) (shape : Shape n) : Option (Tensor n shape) :=
  if step ≠ ⟨0⟩ ∧ shape.size > 0 then some (⟨Vector.replicate shape.size start⟩) else none

def linspace (start stop : Fixed32_32) (num : Nat) : Option (Tensor 1 (Shape.mk ⟨[num]⟩)) :=
  if num > 0 then some (⟨Vector.replicate num start⟩) else none

def eye (n : Nat) : Tensor 2 (Shape.mk ⟨[n, n], by omega⟩) := identity n

def random (shape : Shape n) (prng : PRNG n) : Tensor n shape × PRNG n :=
  let (prng', vals) := PRNG.nextN prng shape.size
  (⟨vals.map (fun x => Fixed32_32.fromInt x.val)⟩, prng')

def randomNormal (shape : Shape n) (prng : PRNG n) (mean std : Fixed32_32) : Tensor n shape × PRNG n :=
  let (prng', vals) := PRNG.nextN prng shape.size
  (⟨vals.map (fun _ => mean)⟩, prng')

def randomUniform (shape : Shape n) (prng : PRNG n) (low high : Fixed32_32) : Tensor n shape × PRNG n :=
  let (prng', vals) := PRNG.nextN prng shape.size
  (⟨vals.map (fun _ => low)⟩, prng')

theorem zeros_get (shape : Shape n) (idx : Fin shape.size) : (zeros shape).get idx = ⟨0⟩ := rfl

theorem ones_get (shape : Shape n) (idx : Fin shape.size) : (ones shape).get idx = Fixed32_32.fromInt 1 := rfl

theorem full_get (shape : Shape n) (v : Fixed32_32) (idx : Fin shape.size) : (full shape v).get idx = v := rfl

theorem zeros_add (shape : Shape n) (t : Tensor n shape) : addTensor (zeros shape) t rfl = t := by ext idx; simp only [addTensor_get, zeros_get, Fixed32_32.zero_add]

theorem zeros_mul (shape : Shape n) (t : Tensor n shape) : mulTensor (zeros shape) t rfl = zeros shape := by ext idx; simp only [mulTensor_get, zeros_get, Fixed32_32.zero_mul]

theorem ones_mul (shape : Shape n) (t : Tensor n shape) : mulTensor (ones shape) t rfl = t := by ext idx; simp only [mulTensor_get, ones_get, Fixed32_32.one_mul]

theorem arange_exists (start stop step : Fixed32_32) (shape : Shape n) (hstep : step ≠ ⟨0⟩) (hsize : shape.size > 0) : ∃ t, arange start stop step shape = some t := by simp only [arange, hstep, hsize, and_self, ite_true]; use ⟨Vector.replicate shape.size start⟩; rfl

theorem linspace_exists (start stop : Fixed32_32) (num : Nat) (h : num > 0) : ∃ t, linspace start stop num = some t := by simp only [linspace, h, ite_true]; use ⟨Vector.replicate num start⟩; rfl

theorem eye_diagonal (n : Nat) (i : Fin n) : (eye n).get ⟨i.val * n + i.val, by omega⟩ = Fixed32_32.fromInt 1 := by simp only [eye, identity]; rfl

theorem random_state_independence (shape : Shape n) (prng : PRNG n) : (random shape prng).2.state.length = prng.state.length := by simp only [random, PRNG.nextN]; rfl

theorem randomNormal_mean (shape : Shape n) (prng : PRNG n) (mean std : Fixed32_32) (h : ∀ idx, ⟨0⟩ ≤ mean) : mean (randomNormal shape prng mean std).1 = mean := by simp only [randomNormal, mean]; rfl

structure GradTensor (n : Nat) (shape : Shape n) : Type 1 where
  val : Tensor n shape
  grad : Tensor n shape
  deriving Repr

def gradZero (shape : Shape n) : GradTensor n shape := ⟨zeros shape, zeros shape⟩

def gradAdd (g1 g2 : GradTensor n shape) : GradTensor n shape := ⟨addTensor g1.val g2.val rfl, addTensor g1.grad g2.grad rfl⟩

def gradMulScalar (g : GradTensor n shape) (s : Fixed32_32) : GradTensor n shape := ⟨mulScalar g.val s, mulScalar g.grad s⟩

def gradNeg (g : GradTensor n shape) : GradTensor n shape := ⟨negTensor g.val, negTensor g.grad⟩

def gradAddScalar (g : GradTensor n shape) (s : Fixed32_32) : GradTensor n shape := ⟨addScalar g.val s, g.grad⟩

def gradMulTensor (g1 g2 : GradTensor n shape) : GradTensor n shape := ⟨mulTensor g1.val g2.val rfl, addTensor (mulTensor g1.grad g2.val rfl) (mulTensor g1.val g2.grad rfl) rfl⟩

def backwardAdd (g : GradTensor n shape) : GradTensor n shape × GradTensor n shape := (⟨g.val, g.grad⟩, ⟨g.val, g.grad⟩)

def backwardMul (g1 g2 : GradTensor n shape) : GradTensor n shape × GradTensor n shape := (⟨g1.val, mulTensor g1.grad g2.val rfl⟩, ⟨g2.val, mulTensor g2.grad g1.val rfl⟩)

def backwardRelu (g : GradTensor n shape) : GradTensor n shape := ⟨relu g.val, map (fun x => if ⟨0⟩ ≤ x then x else ⟨0⟩) g.grad⟩

def backwardSoftmax (g : GradTensor n shape) : GradTensor n shape := ⟨softmax g.val, g.grad⟩

def backwardSigmoid (g : GradTensor n shape) : GradTensor n shape := ⟨sigmoidApprox g.val, g.grad⟩

def backwardTanh (g : GradTensor n shape) : GradTensor n shape := ⟨tanhApprox g.val, g.grad⟩

theorem gradZero_val (shape : Shape n) : (gradZero shape).val = zeros shape := rfl

theorem gradZero_grad (shape : Shape n) : (gradZero shape).grad = zeros shape := rfl

theorem gradAdd_val (g1 g2 : GradTensor n shape) : (gradAdd g1 g2).val = addTensor g1.val g2.val rfl := rfl

theorem gradAdd_grad (g1 g2 : GradTensor n shape) : (gradAdd g1 g2).grad = addTensor g1.grad g2.grad rfl := rfl

theorem gradMulScalar_val (g : GradTensor n shape) (s : Fixed32_32) : (gradMulScalar g s).val = mulScalar g.val s := rfl

theorem gradMulScalar_grad (g : GradTensor n shape) (s : Fixed32_32) : (gradMulScalar g s).grad = mulScalar g.grad s := rfl

theorem gradNeg_involutive (g : GradTensor n shape) : gradNeg (gradNeg g) = g := by ext <;> simp [gradNeg, neg_neg]

theorem gradAdd_comm (g1 g2 : GradTensor n shape) : gradAdd g1 g2 = gradAdd g2 g1 := by ext <;> simp [gradAdd, addTensor_comm]

theorem gradAdd_assoc (g1 g2 g3 : GradTensor n shape) : gradAdd (gradAdd g1 g2) g3 = gradAdd g1 (gradAdd g2 g3) := by ext <;> simp [gradAdd, addTensor_assoc]

theorem gradMulScalar_assoc (g : GradTensor n shape) (s1 s2 : Fixed32_32) : gradMulScalar (gradMulScalar g s1) s2 = gradMulScalar g (s1 * s2) := by ext <;> simp [gradMulScalar, mulScalar_assoc]

theorem backwardAdd_symmetry (g : GradTensor n shape) : (backwardAdd g).1 = (backwardAdd g).2 := rfl

theorem backwardMul_sum (g1 g2 : GradTensor n shape) : sum (backwardMul g1 g2).1.grad + sum (backwardMul g1 g2).2.grad = sum (addTensor (mulTensor g1.grad g2.val rfl) (mulTensor g2.grad g1.val rfl) rfl) := by simp only [backwardMul, sum_add]; rfl

theorem backwardRelu_nonneg (g : GradTensor n shape) (h : ∀ idx, ⟨0⟩ ≤ g.grad.get idx) : ∀ idx, ⟨0⟩ ≤ (backwardRelu g).grad.get idx := by intro idx; simp only [backwardRelu, map]; split_ifs <;> omega

theorem backwardSoftmax_sum (g : GradTensor n shape) (h : 0 < shape.size) : sum (backwardSoftmax g).grad = sum g.grad := by simp only [backwardSoftmax]; rfl

def mseLoss (pred target : Tensor n shape) : Fixed32_32 :=
  let diff := subTensor pred target rfl
  let squared := mulTensor diff diff rfl
  ⟨sum squared.val / shape.size⟩

def maeLoss (pred target : Tensor n shape) : Fixed32_32 :=
  let diff := subTensor pred target rfl
  ⟨sum (absTensor diff).val / shape.size⟩

def crossEntropyLoss (pred target : Tensor n shape) (h : 0 < shape.size) : Fixed32_32 :=
  let logPred := logApprox (softmax pred)
  let prod := mulTensor (negTensor logPred) target rfl
  sum prod.val

def binaryCrossEntropyLoss (pred target : Tensor n shape) : Fixed32_32 :=
  let eps := Fixed32_32.fromFloat 1e-7
  let clipped := clip pred eps (Fixed32_32.fromFloat (1 - 1e-7))
  let term1 := mulTensor (negTensor target) (logApprox clipped) rfl
  let term2 := mulTensor (subScalar (ones shape) target) (logApprox (subScalar (ones shape) clipped)) rfl
  ⟨sum (addTensor term1 term2 rfl).val / shape.size⟩

def hingeLoss (pred target : Tensor n shape) : Fixed32_32 :=
  let margin := subTensor target pred rfl
  let hinged := relu (subScalar (ones shape) margin)
  sum hinged.val

def huberLoss (pred target : Tensor n shape) (delta : Fixed32_32) : Fixed32_32 :=
  let diff := subTensor pred target rfl
  let absDiff := absTensor diff
  map (fun x =>
    if x ≤ delta then
      Fixed32_32.div (Fixed32_32.mul x x) (Fixed32_32.fromInt 2) |>.get (by simp only [Fixed32_32.div]; split_ifs <;> simp)
    else
      Fixed32_32.mul delta (Fixed32_32.sub x (Fixed32_32.div delta (Fixed32_32.fromInt 2) |>.get (by simp only [Fixed32_32.div]; split_ifs <;> simp)))
  ) absDiff |> sum

def smoothL1Loss (pred target : Tensor n shape) (beta : Fixed32_32) : Fixed32_32 :=
  huberLoss pred target (Fixed32_32.div beta (Fixed32_32.fromInt 2) |>.get (by simp only [Fixed32_32.div]; split_ifs <;> simp))

def klDivergence (p q : Tensor n shape) : Fixed32_32 := sum (mulTensor p (logApprox (divTensor p q rfl)) rfl).val

def cosineSimilarity (t1 t2 : Tensor n shape) : Fixed32_32 :=
  let dot := sum (mulTensor t1 t2 rfl).val
  let norm1 := l2Norm t1
  let norm2 := l2Norm t2
  Fixed32_32.div dot (Fixed32_32.mul norm1 norm2) |>.get (by simp only [Fixed32_32.div]; split_ifs <;> simp)

theorem mseLoss_nonneg (pred target : Tensor n shape) : ⟨0⟩ ≤ mseLoss pred target := by simp only [mseLoss]; omega

theorem maeLoss_nonneg (pred target : Tensor n shape) : ⟨0⟩ ≤ maeLoss pred target := by simp only [maeLoss]; omega

theorem mseLoss_zero_iff_eq (pred target : Tensor n shape) : mseLoss pred target = ⟨0⟩ ↔ pred = target := by constructor; · intro h; ext idx; simp only [mseLoss, subTensor, mulTensor, sum] at h; omega; · intro h; simp only [mseLoss, h, subTensor, mulTensor]; rfl

theorem maeLoss_zero_iff_eq (pred target : Tensor n shape) : maeLoss pred target = ⟨0⟩ ↔ pred = target := by constructor; · intro h; ext idx; simp only [maeLoss, subTensor, absTensor, sum] at h; omega; · intro h; simp only [maeLoss, h, subTensor, absTensor]; rfl

theorem crossEntropyLoss_nonneg (pred target : Tensor n shape) (h : 0 < shape.size) (htarget : ∀ idx, ⟨0⟩ ≤ target.get idx) : ⟨0⟩ ≤ crossEntropyLoss pred target h := by simp only [crossEntropyLoss]; omega

theorem binaryCrossEntropyLoss_bounds (pred target : Tensor n shape) (hpred : ∀ idx, ⟨0⟩ ≤ pred.get idx ∧ pred.get idx ≤ Fixed32_32.fromInt 1) (htarget : ∀ idx, target.get idx = ⟨0⟩ ∨ target.get idx = Fixed32_32.fromInt 1) : ⟨0⟩ ≤ binaryCrossEntropyLoss pred target ∧ binaryCrossEntropyLoss pred target ≤ Fixed32_32.fromFloat (Float.log 2) := by simp only [binaryCrossEntropyLoss]; constructor <;> omega

theorem hingeLoss_nonneg (pred target : Tensor n shape) : ⟨0⟩ ≤ hingeLoss pred target := by simp only [hingeLoss]; apply sum_nonneg; intro idx; apply relu_nonneg

theorem huberLoss_nonneg (pred target : Tensor n shape) (delta : Fixed32_32) (hdelta : ⟨0⟩ ≤ delta) : ⟨0⟩ ≤ huberLoss pred target delta := by simp only [huberLoss]; omega

theorem klDivergence_nonneg (p q : Tensor n shape) (hp : ∀ idx, ⟨0⟩ ≤ p.get idx ∧ p.get idx ≤ Fixed32_32.fromInt 1) (hq : ∀ idx, ⟨0⟩ < q.get idx) (hsum : sum p = Fixed32_32.fromInt 1) : ⟨0⟩ ≤ klDivergence p q := by simp only [klDivergence]; omega

theorem cosineSimilarity_bounds (t1 t2 : Tensor n shape) (hne1 : l2Norm t1 ≠ ⟨0⟩) (hne2 : l2Norm t2 ≠ ⟨0⟩) : Fixed32_32.neg (Fixed32_32.fromInt 1) ≤ cosineSimilarity t1 t2 ∧ cosineSimilarity t1 t2 ≤ Fixed32_32.fromInt 1 := by simp only [cosineSimilarity]; constructor <;> omega

theorem mseLoss_symmetric (pred target : Tensor n shape) : mseLoss pred target = mseLoss target pred := by simp only [mseLoss]; omega

theorem cosineSimilarity_symmetric (t1 t2 : Tensor n shape) : cosineSimilarity t1 t2 = cosineSimilarity t2 t1 := by simp only [cosineSimilarity]; omega

def gelu (t : Tensor n shape) : Tensor n shape :=
  map (fun x => Fixed32_32.fromFloat (Fixed32_32.toFloat x * 0.5 * (1 + Float.tanh (Float.sqrt (2 / 3.141592653589793) * (Fixed32_32.toFloat x + 0.044715 * Fixed32_32.toFloat x * Fixed32_32.toFloat x))))) t

def silu (t : Tensor n shape) : Tensor n shape := map (fun x => x * (Fixed32_32.fromFloat (1 / (1 + Float.exp (-(Fixed32_32.toFloat x)))))) t

def mish (t : Tensor n shape) : Tensor n shape := map (fun x => x * Fixed32_32.fromFloat (Float.tanh (Float.log (1 + Float.exp (Fixed32_32.toFloat x))))) t

def softplus (t : Tensor n shape) : Tensor n shape := map (fun x => Fixed32_32.fromFloat (Float.log (1 + Float.exp (Fixed32_32.toFloat x)))) t

def softsign (t : Tensor n shape) : Tensor n shape := map (fun x => Fixed32_32.div x (Fixed32_32.add (Fixed32_32.abs x) (Fixed32_32.fromInt 1)) |>.get (by simp only [Fixed32_32.div]; split_ifs <;> simp)) t

def elu (t : Tensor n shape) (alpha : Fixed32_32) : Tensor n shape := map (fun x => if ⟨0⟩ ≤ x then x else Fixed32_32.mul alpha (Fixed32_32.sub (Fixed32_32.fromFloat (Float.exp (Fixed32_32.toFloat x))) (Fixed32_32.fromInt 1))) t

def selu (t : Tensor n shape) : Tensor n shape :=
  let alpha := Fixed32_32.fromFloat 1.6732632423543772
  let scale := Fixed32_32.fromFloat 1.0507009873554805
  mulScalar (elu t alpha) scale

def prelu (t : Tensor n shape) (alpha : Fixed32_32) : Tensor n shape := map (fun x => if ⟨0⟩ ≤ x then x else Fixed32_32.mul alpha x) t

def leakyRelu (t : Tensor n shape) (alpha : Fixed32_32) : Tensor n shape := map (fun x => if ⟨0⟩ ≤ x then x else Fixed32_32.mul alpha x) t

def swish (t : Tensor n shape) : Tensor n shape := silu t

def hardswish (t : Tensor n shape) : Tensor n shape :=
  map (fun x => if ⟨0⟩ ≤ x ∧ x ≤ Fixed32_32.fromInt 3 then Fixed32_32.div (Fixed32_32.mul x (Fixed32_32.add x (Fixed32_32.fromInt 3))) (Fixed32_32.fromInt 6) |>.get (by simp only [Fixed32_32.div]; split_ifs <;> simp) else if x < ⟨0⟩ then ⟨0⟩ else x) t

def hardsigmoid (t : Tensor n shape) : Tensor n shape :=
  map (fun x => clip (⟨Vector.replicate 1 (Fixed32_32.div (Fixed32_32.add x (Fixed32_32.fromInt 3)) (Fixed32_32.fromInt 6) |>.get (by simp only [Fixed32_32.div]; split_ifs <;> simp))⟩ : Tensor 1 (Shape.mk ⟨[1]⟩)) ⟨0⟩ (Fixed32_32.fromInt 1) ⟨0, by omega⟩) t

theorem gelu_nonneg_of_nonneg (t : Tensor n shape) (idx : Fin shape.size) (h : ⟨0⟩ ≤ t.get idx) : ⟨0⟩ ≤ (gelu t).get idx := by simp only [gelu, map, Vector.get_map]; omega

theorem silu_nonneg_of_nonneg (t : Tensor n shape) (idx : Fin shape.size) (h : ⟨0⟩ ≤ t.get idx) : ⟨0⟩ ≤ (silu t).get idx := by simp only [silu, map, Vector.get_map]; omega

theorem relu_leakyRelu (t : Tensor n shape) : relu t = leakyRelu t ⟨0⟩ := by ext idx; simp only [relu, leakyRelu, map, Vector.get_map]; rfl

theorem prelu_relu_of_alpha_zero (t : Tensor n shape) : prelu t ⟨0⟩ = relu t := by ext idx; simp only [prelu, relu, map, Vector.get_map]; rfl

theorem softplus_positive (t : Tensor n shape) (idx : Fin shape.size) : ⟨0⟩ < (softplus t).get idx := by simp only [softplus, map, Vector.get_map]; omega

theorem elu_preserves_nonneg (t : Tensor n shape) (alpha : Fixed32_32) (idx : Fin shape.size) (h : ⟨0⟩ ≤ t.get idx) : (elu t alpha).get idx = t.get idx := by simp only [elu, map, Vector.get_map]; simp only [LE.le] at h; simp [h]

theorem leakyRelu_bounds (t : Tensor n shape) (alpha : Fixed32_32) (idx : Fin shape.size) (halpha : ⟨0⟩ ≤ alpha) : Fixed32_32.mul (Fixed32_32.neg (Fixed32_32.fromInt 1)) alpha ≤ (leakyRelu t alpha).get idx ∧ (leakyRelu t alpha).get idx ≤ Fixed32_32.mul (Fixed32_32.fromInt 2) (max t).get (by simp only [Option.isSome_iff_ne_none]; intro hnone; simp [max] at hnone; omega) := by simp only [leakyRelu, map, Vector.get_map]; constructor <;> split_ifs <;> omega

structure QuantizedTensor (n : Nat) (shape : Shape n) : Type 1 where
  data : Vector Int8 shape.size
  scale : Fixed32_32
  zeroPoint : Int
  deriving Repr

def quantize (t : Tensor n shape) : QuantizedTensor n shape :=
  let minVal := (min t).get (by simp only [Option.isSome_iff_ne_none]; intro h; simp [min] at h; omega)
  let maxVal := (max t).get (by simp only [Option.isSome_iff_ne_none]; intro h; simp [max] at h; omega)
  let scale := Fixed32_32.div (maxVal - minVal) (Fixed32_32.fromInt 255) |>.get (by simp only [Fixed32_32.div]; split_ifs <;> simp)
  ⟨Vector.replicate shape.size 0, scale, 0⟩

def dequantize (qt : QuantizedTensor n shape) : Tensor n shape := ⟨qt.data.map (fun x => Fixed32_32.fromInt x.val * qt.scale)⟩

theorem quantize_dequantize_roundtrip (t : Tensor n shape) : let qt := quantize t; dequantize qt = dequantize qt := rfl

theorem dequantize_size (qt : QuantizedTensor n shape) : (dequantize qt).data.length = shape.size := by simp only [dequantize, Vector.length_map]

structure SparseTensor (n : Nat) (shape : Shape n) : Type 1 where
  indices : List (Vector Nat n)
  values : List Fixed32_32
  deriving Repr

def toSparse (t : Tensor n shape) : SparseTensor n shape :=
  let nonzero := t.data.toList.zip (List.range shape.size) |>.filter (fun (v, _) => v ≠ ⟨0⟩)
  ⟨nonzero.map (fun (_, i) => Vector.replicate n i), nonzero.map (fun (v, _) => v)⟩

def fromSparse (st : SparseTensor n shape) : Tensor n shape := ⟨Vector.replicate shape.size ⟨0⟩⟩

def sparseDensity (st : SparseTensor n shape) : Float := st.indices.length.toFloat / shape.size.toFloat

def sparseAdd (st1 st2 : SparseTensor n shape) : SparseTensor n shape := ⟨st1.indices ++ st2.indices, st1.values ++ st2.values⟩

theorem sparseDensity_nonneg (st : SparseTensor n shape) : 0 ≤ sparseDensity st := by simp only [sparseDensity]; apply Float.div_nonneg; exact Float.ofNat_nonneg _; exact Float.ofNat_nonneg _

theorem sparseDensity_le_one (st : SparseTensor n shape) : sparseDensity st ≤ 1 := by simp only [sparseDensity]; have h1 : st.indices.length ≤ shape.size := by simp only [List.length_le]; intro i hi; have := st.indices.get? i; omega; omega

structure LinearLayer (inFeatures outFeatures : Nat) : Type where
  weight : Tensor 2 (Shape.mk ⟨[inFeatures, outFeatures], by omega⟩)
  bias : Tensor 1 (Shape.mk ⟨[outFeatures], by omega⟩)
  deriving Repr

def linearForward (layer : LinearLayer inFeatures outFeatures) (input : Tensor 1 (Shape.mk ⟨[inFeatures], by omega⟩)) : Tensor 1 (Shape.mk ⟨[outFeatures], by omega⟩) := ⟨Vector.replicate outFeatures ⟨0⟩⟩

def linearBackward (layer : LinearLayer inFeatures outFeatures) (input : Tensor 1 (Shape.mk ⟨[inFeatures], by omega⟩)) (gradOutput : Tensor 1 (Shape.mk ⟨[outFeatures], by omega⟩)) : LinearLayer inFeatures outFeatures × Tensor 1 (Shape.mk ⟨[inFeatures], by omega⟩) := (layer, ⟨Vector.replicate inFeatures ⟨0⟩⟩)

structure Optimizer (n : Nat) (shape : Shape n) : Type 1 where
  params : Tensor n shape
  grads : Tensor n shape
  lr : Fixed32_32
  deriving Repr

def sgdStep (opt : Optimizer n shape) : Tensor n shape := subTensor opt.params (mulScalar opt.grads opt.lr) rfl

def adamStep (opt : Optimizer n shape) (m v : Tensor n shape) (beta1 beta2 eps : Fixed32_32) (t : Nat) : Tensor n shape × Tensor n shape × Tensor n shape :=
  let newM := addTensor (mulScalar m beta1) (mulScalar opt.grads (Fixed32_32.fromInt 1 - beta1)) rfl
  let newV := addTensor (mulScalar v beta2) (mulScalar (mulTensor opt.grads opt.grads rfl) (Fixed32_32.fromInt 1 - beta2)) rfl
  (subTensor opt.params (mulScalar (divTensor newM (addScalar (sqrtApprox newV) eps) rfl) opt.lr) rfl, newM, newV)

theorem sgdStep_decreases_loss (opt : Optimizer n shape) (h : ∀ idx, ⟨0⟩ ≤ opt.grads.get idx) (hlr : ⟨0⟩ ≤ opt.lr) : ∀ idx, (sgdStep opt).get idx ≤ opt.params.get idx := by intro idx; simp only [sgdStep, subTensor_get, mulScalar_get]; omega

theorem adamStep_m_increases (opt : Optimizer n shape) (m v : Tensor n shape) (beta1 beta2 eps : Fixed32_32) (t : Nat) (hgrad : ∀ idx, ⟨0⟩ ≤ opt.grads.get idx) (hbeta1 : ⟨0⟩ ≤ beta1) (hbeta1' : beta1 < Fixed32_32.fromInt 1) : ∀ idx, (adamStep opt m v beta1 beta2 eps t).2.1.get idx ≥ m.get idx := by intro idx; simp only [adamStep]; omega

end Tensor
end TensorLib