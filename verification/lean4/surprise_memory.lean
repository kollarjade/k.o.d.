import Mathlib.Data.Vector.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Ring.Basic
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Order.Field
import Mathlib.Algebra.Order.Ring
import Mathlib.Algebra.Order.Group
import Mathlib.Algebra.Log
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.List.Sort
import Mathlib.Data.List.Basic
import Mathlib.Tactic.LibrarySearch
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.FieldSimp
import Mathlib.MeasureTheory.Measure.Lebesgue.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Std.Data.HashMap
import Std.Data.HashMap.Default

def RETENTION_AGE_WEIGHT : Real := 0.3
def RETENTION_FREQUENCY_WEIGHT : Real := 0.2
def RETENTION_BASE_WEIGHT : Real := 0.5
def NANOSECONDS_TO_MILLISECONDS : Real := 1000000.0
def HASH_SIZE : Nat := 16
def HASH_BITS : Nat := HASH_SIZE * 8
def MAX_INPUT_SIZE : Nat := 100 * 1024 * 1024
def JACCARD_SAMPLE_SIZE : Nat := 1000
def MAX_ENTANGLEMENT_PAIRS : Nat := 100
def DEFAULT_SURPRISE_THRESHOLD : Real := 0.3

instance [Hashable α] : Hashable (Vector α n) where
  hash v := mixHash 0 (v.toList.hash)

structure SurpriseMetrics where
  jaccard_dissimilarity : Real
  content_hash_distance : Real
  temporal_novelty : Real
  combined_surprise : Real

def SurpriseMetrics.init (jaccard : Real) (hash_dist : Real) (temporal : Real) : SurpriseMetrics :=
  let clamped_jaccard := max 0.0 (min 1.0 jaccard)
  let clamped_hash := max 0.0 (min 1.0 hash_dist)
  let clamped_temporal := max 0.0 (min 1.0 temporal)
  let combined := (clamped_jaccard + clamped_hash + clamped_temporal) / 3.0
  { jaccard_dissimilarity := clamped_jaccard, content_hash_distance := clamped_hash, temporal_novelty := clamped_temporal, combined_surprise := combined }

def SurpriseMetrics.exceedsThreshold (self : SurpriseMetrics) (threshold : Real) : Bool :=
  self.combined_surprise > threshold

structure SurpriseRecord where
  block_id : Vector (Fin 256) HASH_SIZE
  surprise_score : Real
  creation_time : Int
  last_access_time : Int
  retention_priority : Real
  access_frequency : Nat

parameter (nanoTimestamp : Unit → Int)

def SurpriseRecord.init (block_id : Vector (Fin 256) HASH_SIZE) (score : Real) : SurpriseRecord :=
  let now := nanoTimestamp ()
  { block_id := block_id, surprise_score := score, creation_time := now, last_access_time := now, retention_priority := score, access_frequency := 1 }

def SurpriseRecord.updateRetention (self : SurpriseRecord) : SurpriseRecord :=
  let now := nanoTimestamp ()
  let age_ns := now - self.creation_time
  let abs_age_ns := if age_ns < 0 then -age_ns else age_ns
  let clamped_age := min abs_age_ns (2^63 - 1)
  let age_ms := (clamped_age : Real) / NANOSECONDS_TO_MILLISECONDS
  let age_factor := 1.0 / (1.0 + age_ms)
  let freq_val := (self.access_frequency + 1 : Real)
  let frequency_factor := Real.log freq_val
  let new_priority := self.surprise_score * (RETENTION_BASE_WEIGHT + RETENTION_AGE_WEIGHT * age_factor + RETENTION_FREQUENCY_WEIGHT * frequency_factor)
  { self with retention_priority := new_priority, last_access_time := now }

def SurpriseRecord.recordAccess (self : SurpriseRecord) : SurpriseRecord :=
  { self with access_frequency := self.access_frequency + 1 }.updateRetention

def SurpriseRecord.getRetentionPriority (self : SurpriseRecord) : Real :=
  self.retention_priority

def SurpriseRecord.getAccessFrequency (self : SurpriseRecord) : Nat :=
  self.access_frequency

structure SurpriseMemoryStatistics where
  total_blocks : Nat
  high_surprise_blocks : Nat
  low_surprise_blocks : Nat
  average_surprise : Real
  surprise_threshold : Real
  evictions_due_to_low_surprise : Nat
  novel_block_allocations : Nat
  total_surprise_sum : Real

def SurpriseMemoryStatistics.init (threshold : Real) : SurpriseMemoryStatistics :=
  { total_blocks := 0, high_surprise_blocks := 0, low_surprise_blocks := 0, average_surprise := 0.0, surprise_threshold := threshold, evictions_due_to_low_surprise := 0, novel_block_allocations := 0, total_surprise_sum := 0.0 }

def SurpriseMemoryStatistics.addBlock (self : SurpriseMemoryStatistics) (surprise_score : Real) (threshold : Real) : SurpriseMemoryStatistics :=
  let new_total := self.total_blocks + 1
  let new_sum := self.total_surprise_sum + surprise_score
  let new_high := if surprise_score > threshold then self.high_surprise_blocks + 1 else self.high_surprise_blocks
  let new_low := if surprise_score ≤ threshold then self.low_surprise_blocks + 1 else self.low_surprise_blocks
  let new_avg := if new_total > 0 then new_sum / new_total.toReal else 0.0
  { self with total_blocks := new_total, high_surprise_blocks := new_high, low_surprise_blocks := new_low, average_surprise := new_avg, total_surprise_sum := new_sum }

def SurpriseMemoryStatistics.removeBlock (self : SurpriseMemoryStatistics) (surprise_score : Real) (threshold : Real) : SurpriseMemoryStatistics :=
  if self.total_blocks = 0 then self else
  let new_total := self.total_blocks - 1
  let new_sum := max 0.0 (self.total_surprise_sum - surprise_score)
  let new_high := if surprise_score > threshold then if self.high_surprise_blocks > 0 then self.high_surprise_blocks - 1 else 0 else self.high_surprise_blocks
  let new_low := if surprise_score ≤ threshold then if self.low_surprise_blocks > 0 then self.low_surprise_blocks - 1 else 0 else self.low_surprise_blocks
  let new_avg := if new_total > 0 then new_sum / new_total.toReal else 0.0
  { self with total_blocks := new_total, high_surprise_blocks := new_high, low_surprise_blocks := new_low, average_surprise := new_avg, total_surprise_sum := new_sum }

structure MemoryBlock where
  data : List (Fin 256)
  content_hash : Vector (Fin 256) HASH_SIZE
  state : Nat

structure ContentAddressableStorage where
  storage : Std.HashMap (Vector (Fin 256) HASH_SIZE) MemoryBlock
  store : ContentAddressableStorage → List (Fin 256) → Option Nat → ContentAddressableStorage × Vector (Fin 256) HASH_SIZE
  retrieveByContent : ContentAddressableStorage → List (Fin 256) → Option (Vector (Fin 256) HASH_SIZE)
  containsBlock : ContentAddressableStorage → Vector (Fin 256) HASH_SIZE → Bool
  removeBlock : ContentAddressableStorage → Vector (Fin 256) HASH_SIZE → ContentAddressableStorage
  entangleBlocks : ContentAddressableStorage → Vector (Fin 256) HASH_SIZE → Vector (Fin 256) HASH_SIZE → Option ContentAddressableStorage

def ContentAddressableStorage.create : ContentAddressableStorage where
  storage := Std.HashMap.empty
  store := fun s data _ =>
    let h := computeContentHash data
    let b := { data := data, content_hash := h, state := 0 }
    let s' := { s with storage := s.storage.insert h b }
    (s', h)
  retrieveByContent := fun s data =>
    let h := computeContentHash data
    if s.storage.contains h then some h else none
  containsBlock := fun s id => s.storage.contains id
  removeBlock := fun s id => { s with storage := s.storage.erase id }
  entangleBlocks := fun s id1 id2 =>
    if s.storage.contains id1 ∧ s.storage.contains id2 then some s else none

structure DataFlowAnalyzer where
  deinit : Unit

def DataFlowAnalyzer.create : DataFlowAnalyzer where
  deinit := ()

structure CandidateItem where
  block_id : Vector (Fin 256) HASH_SIZE
  priority : Real

structure SurpriseMemoryManager where
  storage : ContentAddressableStorage
  flow_analyzer : DataFlowAnalyzer
  surprise_records : Std.HashMap (Vector (Fin 256) HASH_SIZE) SurpriseRecord
  surprise_threshold : Real
  statistics : SurpriseMemoryStatistics
  owns_storage : Bool
  owns_analyzer : Bool

def SurpriseMemoryManager.init (storage : ContentAddressableStorage) (analyzer : DataFlowAnalyzer) : SurpriseMemoryManager :=
  { storage := storage, flow_analyzer := analyzer, surprise_records := Std.HashMap.empty, surprise_threshold := DEFAULT_SURPRISE_THRESHOLD, statistics := SurpriseMemoryStatistics.init DEFAULT_SURPRISE_THRESHOLD, owns_storage := false, owns_analyzer := false }

def SurpriseMemoryManager.initWithOwnership (storage : ContentAddressableStorage) (analyzer : DataFlowAnalyzer) (owns_storage owns_analyzer : Bool) : SurpriseMemoryManager :=
  { SurpriseMemoryManager.init storage analyzer with owns_storage := owns_storage, owns_analyzer := owns_analyzer }

def SurpriseMemoryManager.deinit (self : SurpriseMemoryManager) : Unit :=
  if self.owns_storage then ()
  else if self.owns_analyzer then ()
  else ()

def SurpriseMemoryManager.setSurpriseThreshold (self : SurpriseMemoryManager) (threshold : Real) : SurpriseMemoryManager :=
  let clamped_threshold := max 0.0 (min 1.0 threshold)
  { self with surprise_threshold := clamped_threshold, statistics := { self.statistics with surprise_threshold := clamped_threshold } }

def SurpriseMemoryManager.getSurpriseThreshold (self : SurpriseMemoryManager) : Real :=
  self.surprise_threshold

def computeContentHash (data : List (Fin 256)) : Vector (Fin 256) HASH_SIZE :=
  Vector.ofFn (fun i => data.get? i.toNat |>.getD 0)

def SurpriseMemoryManager.computeJaccardDistance (self : SurpriseMemoryManager) (data_a data_b : List (Fin 256)) : Real :=
  let set_a := data_a.foldl (init := Std.HashMap.empty) (fun m byte => m.insert byte ())
  let set_b := data_b.foldl (init := Std.HashMap.empty) (fun m byte => m.insert byte ())
  let intersection_count := set_a.fold (fun byte _ acc => if set_b.contains byte then acc + 1 else acc) 0
  let union_count := set_a.size + set_b.size - intersection_count
  if union_count = 0 then 0.0 else 1.0 - (intersection_count.toReal / union_count.toReal)

def SurpriseMemoryManager.computeHashDistance (hash_a hash_b : Vector (Fin 256) HASH_SIZE) : Real :=
  let hamming_dist := Fin.range HASH_SIZE |>.foldl (init := 0) (fun acc i => acc + Nat.popCount ((hash_a.get i) xor (hash_b.get i)).val)
  hamming_dist.toReal / HASH_BITS.toReal

partial def SurpriseMemoryManager.computeSurprise (self : SurpriseMemoryManager) (new_data : List (Fin 256)) : SurpriseMetrics :=
  if new_data.length > MAX_INPUT_SIZE then SurpriseMetrics.init 0.0 0.0 0.0 else
  let new_hash := computeContentHash new_data
  let block_count := self.storage.storage.size
  if block_count = 0 then SurpriseMetrics.init 0.0 0.0 0.0 else
  let max_samples := min block_count JACCARD_SAMPLE_SIZE
  let init_min := (1.0, 1.0)
  let sampled := self.storage.storage.fold (init := (0, init_min.1, init_min.2)) (fun acc key block =>
    if acc.1 >= max_samples then acc else
    let jaccard := self.computeJaccardDistance new_data block.data
    let hash_dist := SurpriseMemoryManager.computeHashDistance new_hash block.content_hash
    (acc.1 + 1, min jaccard acc.2, min hash_dist acc.3)
  )
  let temporal_novelty := 1.0 / (1.0 + Real.sqrt block_count.toReal)
  SurpriseMetrics.init sampled.2 sampled.3 temporal_novelty

def SurpriseMemoryManager.storeBlockInternal (self : SurpriseMemoryManager) (data : List (Fin 256)) (preferred_core : Option Nat) (surprise : SurpriseMetrics) : SurpriseMemoryManager × Vector (Fin 256) HASH_SIZE :=
  let (new_storage, block_id) := self.storage.store self.storage data preferred_core
  let record := SurpriseRecord.init block_id surprise.combined_surprise
  let new_records := self.surprise_records.insert block_id record
  ({ self with storage := new_storage, surprise_records := new_records }, block_id)

partial def SurpriseMemoryManager.storeWithSurprise (self : SurpriseMemoryManager) (data : List (Fin 256)) (preferred_core : Option Nat) : SurpriseMemoryManager × Vector (Fin 256) HASH_SIZE :=
  if data.length > MAX_INPUT_SIZE then (self, Vector.ofFn (fun _ => 0)) else
  let existing_block_id := self.storage.retrieveByContent self.storage data
  match existing_block_id with
  | some block_id =>
    match self.surprise_records.find? block_id with
    | some record =>
      let updated_record := record.recordAccess
      let new_records := self.surprise_records.insert block_id updated_record
      ({ self with surprise_records := new_records }, block_id)
    | none =>
      let surprise := self.computeSurprise data
      let new_record := SurpriseRecord.init block_id surprise.combined_surprise
      let new_records := self.surprise_records.insert block_id new_record
      let new_stats := self.statistics.addBlock surprise.combined_surprise self.surprise_threshold
      ({ self with surprise_records := new_records, statistics := new_stats }, block_id)
  | none =>
    let surprise := self.computeSurprise data
    let updated_self_surprise := self.storeBlockInternal data preferred_core surprise
    let updated_self := updated_self_surprise.1
    let block_id := updated_self_surprise.2
    let new_stats := if surprise.exceedsThreshold self.surprise_threshold then { updated_self.statistics with novel_block_allocations := updated_self.statistics.novel_block_allocations + 1 } else updated_self.statistics
    let new_stats2 := new_stats.addBlock surprise.combined_surprise self.surprise_threshold
    ({ updated_self with statistics := new_stats2 }, block_id)

def partialSort (items : List CandidateItem) (k : Nat) : List CandidateItem :=
  if items.length ≤ 1 ∨ k = 0 then items else
  let actual_k := min k items.length
  (0..actual_k).foldl (fun acc i =>
    let min_idx := (i + 1..items.length).foldl (fun m j => if (items.get! j).priority < (items.get! m).priority then j else m) i
    if min_idx ≠ i then
      let temp := acc.get! i
      let acc1 := acc.set i (acc.get! min_idx)
      acc1.set min_idx temp
    else acc
  ) items

partial def SurpriseMemoryManager.evictLowSurpriseBlocks (self : SurpriseMemoryManager) (target_capacity : Nat) : SurpriseMemoryManager × Nat :=
  if self.statistics.total_blocks ≤ target_capacity then (self, 0) else
  let to_evict := self.statistics.total_blocks - target_capacity
  let candidates := self.surprise_records.toList.map (fun p => { block_id := p.1, priority := p.2.retention_priority })
  let sorted_candidates := partialSort candidates to_evict
  let evict_list := sorted_candidates.take to_evict
  evict_list.foldl (init := (self, 0)) (fun acc candidate =>
    if acc.1.storage.containsBlock acc.1.storage candidate.block_id then
      let curr_self := acc.1
      let score_opt := curr_self.surprise_records.find? candidate.block_id |>.map (·.surprise_score)
      let new_stats := match score_opt with
        | some score => curr_self.statistics.removeBlock score curr_self.surprise_threshold
        | none => curr_self.statistics
      let new_storage := curr_self.storage.removeBlock curr_self.storage candidate.block_id
      let new_records := curr_self.surprise_records.erase candidate.block_id
      let new_self := { curr_self with storage := new_storage, surprise_records := new_records, statistics := { new_stats with evictions_due_to_low_surprise := new_stats.evictions_due_to_low_surprise + 1 } }
      (new_self, acc.2 + 1)
    else acc
  )

def SurpriseMemoryManager.organizeByEntanglement (self : SurpriseMemoryManager) : SurpriseMemoryManager × Nat :=
  let high_surprise_ids := self.surprise_records.toList.filterMap (fun p => if p.2.surprise_score > self.surprise_threshold then some p.1 else none) |>.take MAX_ENTANGLEMENT_PAIRS
  let pairs_count := (0..high_surprise_ids.length).foldl (init := 0) (fun acc i =>
    (i + 1..high_surprise_ids.length).foldl (fun acc’ j =>
      match self.storage.entangleBlocks self.storage (high_surprise_ids.get! i) (high_surprise_ids.get! j) with
      | some new_storage => { self with storage := new_storage }
      | none => self
      acc’ + 1
    ) acc
  )
  (self, pairs_count)

def SurpriseMemoryManager.getStatistics (self : SurpriseMemoryManager) : SurpriseMemoryStatistics :=
  self.statistics

def SurpriseMemoryManager.getSurpriseRecord (self : SurpriseMemoryManager) (block_id : Vector (Fin 256) HASH_SIZE) : Option SurpriseRecord :=
  self.surprise_records.find? block_id

def SurpriseMemoryManager.containsRecord (self : SurpriseMemoryManager) (block_id : Vector (Fin 256) HASH_SIZE) : Bool :=
  self.surprise_records.contains block_id

def SurpriseMemoryManager.getRecordCount (self : SurpriseMemoryManager) : Nat :=
  self.surprise_records.size

lemma SurpriseMetrics_init_clamped_jaccard_ge_zero (jaccard hash_dist temporal : Real) : (SurpriseMetrics.init jaccard hash_dist temporal).jaccard_dissimilarity ≥ 0.0 := by
  simp [SurpriseMetrics.init]
  apply max_le_iff.mpr
  constructor
  linarith
  apply min_le_left 1.0 jaccard

lemma SurpriseMetrics_init_clamped_jaccard_le_one (jaccard hash_dist temporal : Real) : (SurpriseMetrics.init jaccard hash_dist temporal).jaccard_dissimilarity ≤ 1.0 := by
  simp [SurpriseMetrics.init]
  apply min_le_right 1.0 (max 0.0 jaccard)

lemma SurpriseMetrics_init_clamped_hash_ge_zero (jaccard hash_dist temporal : Real) : (SurpriseMetrics.init jaccard hash_dist temporal).content_hash_distance ≥ 0.0 := by
  simp [SurpriseMetrics.init]
  apply max_le_iff.mpr
  constructor
  linarith
  apply min_le_left 1.0 hash_dist

lemma SurpriseMetrics_init_clamped_hash_le_one (jaccard hash_dist temporal : Real) : (SurpriseMetrics.init jaccard hash_dist temporal).content_hash_distance ≤ 1.0 := by
  simp [SurpriseMetrics.init]
  apply min_le_right 1.0 (max 0.0 hash_dist)

lemma SurpriseMetrics_init_clamped_temporal_ge_zero (jaccard hash_dist temporal : Real) : (SurpriseMetrics.init jaccard hash_dist temporal).temporal_novelty ≥ 0.0 := by
  simp [SurpriseMetrics.init]
  apply max_le_iff.mpr
  constructor
  linarith
  apply min_le_left 1.0 temporal

lemma SurpriseMetrics_init_clamped_temporal_le_one (jaccard hash_dist temporal : Real) : (SurpriseMetrics.init jaccard hash_dist temporal).temporal_novelty ≤ 1.0 := by
  simp [SurpriseMetrics.init]
  apply min_le_right 1.0 (max 0.0 temporal)

lemma SurpriseMetrics_init_combined_eq_avg (jaccard hash_dist temporal : Real) : (SurpriseMetrics.init jaccard hash_dist temporal).combined_surprise = ((max 0.0 (min 1.0 jaccard)) + (max 0.0 (min 1.0 hash_dist)) + (max 0.0 (min 1.0 temporal))) / 3.0 := by
  simp [SurpriseMetrics.init]

lemma SurpriseMetrics_init_combined_ge_zero (jaccard hash_dist temporal : Real) : (SurpriseMetrics.init jaccard hash_dist temporal).combined_surprise ≥ 0.0 := by
  simp [SurpriseMetrics.init]
  apply div_nonneg
  apply add_nonneg
  apply add_nonneg
  apply ge_trans (max_le_iff.mp (le_max_left _ _)) (min_le_left _ _)
  apply ge_trans (max_le_iff.mp (le_max_left _ _)) (min_le_left _ _)
  apply ge_trans (max_le_iff.mp (le_max_left _ _)) (min_le_left _ _)
  linarith

lemma SurpriseMetrics_init_combined_le_one (jaccard hash_dist temporal : Real) : (SurpriseMetrics.init jaccard hash_dist temporal).combined_surprise ≤ 1.0 := by
  simp [SurpriseMetrics.init]
  apply div_le_one.mpr
  linarith [(min_le_right _ _) , (min_le_right _ _) , (min_le_right _ _)]

lemma SurpriseMetrics_exceedsThreshold_iff_combined_gt (self : SurpriseMetrics) (threshold : Real) : self.exceedsThreshold threshold = true ↔ self.combined_surprise > threshold := by
  simp [SurpriseMetrics.exceedsThreshold]

lemma SurpriseRecord_init_creation_eq_last (block_id : Vector (Fin 256) HASH_SIZE) (score : Real) : (SurpriseRecord.init block_id score).creation_time = (SurpriseRecord.init block_id score).last_access_time := by
  simp [SurpriseRecord.init]

lemma SurpriseRecord_init_retention_eq_score (block_id : Vector (Fin 256) HASH_SIZE) (score : Real) : (SurpriseRecord.init block_id score).retention_priority = score := by
  simp [SurpriseRecord.init]

lemma SurpriseRecord_init_freq_one (block_id : Vector (Fin 256) HASH_SIZE) (score : Real) : (SurpriseRecord.init block_id score).access_frequency = 1 := by
  simp [SurpriseRecord.init]

lemma SurpriseRecord_updateRetention_last_now (self : SurpriseRecord) : (self.updateRetention).last_access_time = nanoTimestamp () := by
  simp [SurpriseRecord.updateRetention]

lemma SurpriseRecord_updateRetention_creation_unchanged (self : SurpriseRecord) : (self.updateRetention).creation_time = self.creation_time := by
  simp [SurpriseRecord.updateRetention]

lemma SurpriseRecord_updateRetention_score_unchanged (self : SurpriseRecord) : (self.updateRetention).surprise_score = self.surprise_score := by
  simp [SurpriseRecord.updateRetention]

lemma SurpriseRecord_updateRetention_freq_unchanged (self : SurpriseRecord) : (self.updateRetention).access_frequency = self.access_frequency := by
  simp [SurpriseRecord.updateRetention]

lemma SurpriseRecord_updateRetention_block_id_unchanged (self : SurpriseRecord) : (self.updateRetention).block_id = self.block_id := by
  simp [SurpriseRecord.updateRetention]

lemma SurpriseRecord_recordAccess_freq_inc (self : SurpriseRecord) : (self.recordAccess).access_frequency = self.access_frequency + 1 := by
  simp [SurpriseRecord.recordAccess]

lemma SurpriseRecord_recordAccess_creation_unchanged (self : SurpriseRecord) : (self.recordAccess).creation_time = self.creation_time := by
  simp [SurpriseRecord.recordAccess, SurpriseRecord.updateRetention]

lemma SurpriseRecord_recordAccess_score_unchanged (self : SurpriseRecord) : (self.recordAccess).surprise_score = self.surprise_score := by
  simp [SurpriseRecord.recordAccess, SurpriseRecord.updateRetention]

lemma SurpriseRecord_recordAccess_block_id_unchanged (self : SurpriseRecord) : (self.recordAccess).block_id = self.block_id := by
  simp [SurpriseRecord.recordAccess, SurpriseRecord.updateRetention]

lemma SurpriseRecord_getRetentionPriority_eq (self : SurpriseRecord) : self.getRetentionPriority = self.retention_priority := by
  rfl

lemma SurpriseRecord_getAccessFrequency_eq (self : SurpriseRecord) : self.getAccessFrequency = self.access_frequency := by
  rfl

lemma SurpriseMemoryStatistics_init_total_zero (threshold : Real) : (SurpriseMemoryStatistics.init threshold).total_blocks = 0 := by
  simp [SurpriseMemoryStatistics.init]

lemma SurpriseMemoryStatistics_init_high_zero (threshold : Real) : (SurpriseMemoryStatistics.init threshold).high_surprise_blocks = 0 := by
  simp [SurpriseMemoryStatistics.init]

lemma SurpriseMemoryStatistics_init_low_zero (threshold : Real) : (SurpriseMemoryStatistics.init threshold).low_surprise_blocks = 0 := by
  simp [SurpriseMemoryStatistics.init]

lemma SurpriseMemoryStatistics_init_avg_zero (threshold : Real) : (SurpriseMemoryStatistics.init threshold).average_surprise = 0.0 := by
  simp [SurpriseMemoryStatistics.init]

lemma SurpriseMemoryStatistics_init_threshold_eq (threshold : Real) : (SurpriseMemoryStatistics.init threshold).surprise_threshold = threshold := by
  simp [SurpriseMemoryStatistics.init]

lemma SurpriseMemoryStatistics_init_evictions_zero (threshold : Real) : (SurpriseMemoryStatistics.init threshold).evictions_due_to_low_surprise = 0 := by
  simp [SurpriseMemoryStatistics.init]

lemma SurpriseMemoryStatistics_init_novel_zero (threshold : Real) : (SurpriseMemoryStatistics.init threshold).novel_block_allocations = 0 := by
  simp [SurpriseMemoryStatistics.init]

lemma SurpriseMemoryStatistics_init_sum_zero (threshold : Real) : (SurpriseMemoryStatistics.init threshold).total_surprise_sum = 0.0 := by
  simp [SurpriseMemoryStatistics.init]

lemma SurpriseMemoryStatistics_addBlock_total_inc (self : SurpriseMemoryStatistics) (score thresh : Real) : (self.addBlock score thresh).total_blocks = self.total_blocks + 1 := by
  simp [SurpriseMemoryStatistics.addBlock]

lemma SurpriseMemoryStatistics_addBlock_sum_add (self : SurpriseMemoryStatistics) (score thresh : Real) : (self.addBlock score thresh).total_surprise_sum = self.total_surprise_sum + score := by
  simp [SurpriseMemoryStatistics.addBlock]

lemma SurpriseMemoryStatistics_addBlock_high_inc_if_gt (self : SurpriseMemoryStatistics) (score thresh : Real) (h : score > thresh) : (self.addBlock score thresh).high_surprise_blocks = self.high_surprise_blocks + 1 := by
  simp [SurpriseMemoryStatistics.addBlock, h]

lemma SurpriseMemoryStatistics_addBlock_high_same_if_le (self : SurpriseMemoryStatistics) (score thresh : Real) (h : score ≤ thresh) : (self.addBlock score thresh).high_surprise_blocks = self.high_surprise_blocks := by
  simp [SurpriseMemoryStatistics.addBlock, h]

lemma SurpriseMemoryStatistics_addBlock_low_inc_if_le (self : SurpriseMemoryStatistics) (score thresh : Real) (h : score ≤ thresh) : (self.addBlock score thresh).low_surprise_blocks = self.low_surprise_blocks + 1 := by
  simp [SurpriseMemoryStatistics.addBlock, h]

lemma SurpriseMemoryStatistics_addBlock_low_same_if_gt (self : SurpriseMemoryStatistics) (score thresh : Real) (h : score > thresh) : (self.addBlock score thresh).low_surprise_blocks = self.low_surprise_blocks := by
  simp [SurpriseMemoryStatistics.addBlock, h]

lemma SurpriseMemoryStatistics_addBlock_avg_correct (self : SurpriseMemoryStatistics) (score thresh : Real) : (self.addBlock score thresh).average_surprise = (self.total_surprise_sum + score) / (self.total_blocks + 1).toReal := by
  simp [SurpriseMemoryStatistics.addBlock]

lemma SurpriseMemoryStatistics_addBlock_threshold_unchanged (self : SurpriseMemoryStatistics) (score thresh : Real) : (self.addBlock score thresh).surprise_threshold = self.surprise_threshold := by
  simp [SurpriseMemoryStatistics.addBlock]

lemma SurpriseMemoryStatistics_addBlock_evictions_unchanged (self : SurpriseMemoryStatistics) (score thresh : Real) : (self.addBlock score thresh).evictions_due_to_low_surprise = self.evictions_due_to_low_surprise := by
  simp [SurpriseMemoryStatistics.addBlock]

lemma SurpriseMemoryStatistics_addBlock_novel_unchanged (self : SurpriseMemoryStatistics) (score thresh : Real) : (self.addBlock score thresh).novel_block_allocations = self.novel_block_allocations := by
  simp [SurpriseMemoryStatistics.addBlock]

lemma SurpriseMemoryStatistics_removeBlock_total_zero_unchanged (self : SurpriseMemoryStatistics) (score thresh : Real) (h : self.total_blocks = 0) : (self.removeBlock score thresh) = self := by
  simp [SurpriseMemoryStatistics.removeBlock, h]

lemma SurpriseMemoryStatistics_removeBlock_total_dec_if_pos (self : SurpriseMemoryStatistics) (score thresh : Real) (h : self.total_blocks > 0) : (self.removeBlock score thresh).total_blocks = self.total_blocks - 1 := by
  simp [SurpriseMemoryStatistics.removeBlock, h]

lemma SurpriseMemoryStatistics_removeBlock_sum_sub_max_zero (self : SurpriseMemoryStatistics) (score thresh : Real) (h : self.total_blocks > 0) : (self.removeBlock score thresh).total_surprise_sum = max 0.0 (self.total_surprise_sum - score) := by
  simp [SurpriseMemoryStatistics.removeBlock, h]

lemma SurpriseMemoryStatistics_removeBlock_high_dec_if_gt_and_pos (self : SurpriseMemoryStatistics) (score thresh : Real) (h1 : score > thresh) (h2 : self.high_surprise_blocks > 0) (h3 : self.total_blocks > 0) : (self.removeBlock score thresh).high_surprise_blocks = self.high_surprise_blocks - 1 := by
  simp [SurpriseMemoryStatistics.removeBlock, h3, h1, h2]

lemma SurpriseMemoryStatistics_removeBlock_high_same_if_le (self : SurpriseMemoryStatistics) (score thresh : Real) (h1 : score ≤ thresh) (h3 : self.total_blocks > 0) : (self.removeBlock score thresh).high_surprise_blocks = self.high_surprise_blocks := by
  simp [SurpriseMemoryStatistics.removeBlock, h3, h1]

lemma SurpriseMemoryStatistics_removeBlock_low_dec_if_le_and_pos (self : SurpriseMemoryStatistics) (score thresh : Real) (h1 : score ≤ thresh) (h2 : self.low_surprise_blocks > 0) (h3 : self.total_blocks > 0) : (self.removeBlock score thresh).low_surprise_blocks = self.low_surprise_blocks - 1 := by
  simp [SurpriseMemoryStatistics.removeBlock, h3, h1, h2]

lemma SurpriseMemoryStatistics_removeBlock_low_same_if_gt (self : SurpriseMemoryStatistics) (score thresh : Real) (h1 : score > thresh) (h3 : self.total_blocks > 0) : (self.removeBlock score thresh).low_surprise_blocks = self.low_surprise_blocks := by
  simp [SurpriseMemoryStatistics.removeBlock, h3, h1]

lemma SurpriseMemoryStatistics_removeBlock_avg_correct (self : SurpriseMemoryStatistics) (score thresh : Real) (h : self.total_blocks > 0) : (self.removeBlock score thresh).average_surprise = if self.total_blocks - 1 > 0 then max 0.0 (self.total_surprise_sum - score) / (self.total_blocks - 1).toReal else 0.0 := by
  simp [SurpriseMemoryStatistics.removeBlock, h]

lemma SurpriseMemoryStatistics_removeBlock_threshold_unchanged (self : SurpriseMemoryStatistics) (score thresh : Real) : (self.removeBlock score thresh).surprise_threshold = self.surprise_threshold := by
  simp [SurpriseMemoryStatistics.removeBlock]

lemma SurpriseMemoryStatistics_removeBlock_evictions_unchanged (self : SurpriseMemoryStatistics) (score thresh : Real) : (self.removeBlock score thresh).evictions_due_to_low_surprise = self.evictions_due_to_low_surprise := by
  simp [SurpriseMemoryStatistics.removeBlock]

lemma SurpriseMemoryStatistics_removeBlock_novel_unchanged (self : SurpriseMemoryStatistics) (score thresh : Real) : (self.removeBlock score thresh).novel_block_allocations = self.novel_block_allocations := by
  simp [SurpriseMemoryStatistics.removeBlock]

lemma computeContentHash_length (data : List (Fin 256)) : (computeContentHash data).length = HASH_SIZE := by
  simp [computeContentHash, Vector.length_ofFn]

lemma SurpriseMemoryManager_init_threshold_default (storage : ContentAddressableStorage) (analyzer : DataFlowAnalyzer) : (SurpriseMemoryManager.init storage analyzer).surprise_threshold = DEFAULT_SURPRISE_THRESHOLD := by
  simp [SurpriseMemoryManager.init]

lemma SurpriseMemoryManager_init_records_empty (storage : ContentAddressableStorage) (analyzer : DataFlowAnalyzer) : (SurpriseMemoryManager.init storage analyzer).surprise_records = Std.HashMap.empty := by
  simp [SurpriseMemoryManager.init]

lemma SurpriseMemoryManager_init_owns_false (storage : ContentAddressableStorage) (analyzer : DataFlowAnalyzer) : (SurpriseMemoryManager.init storage analyzer).owns_storage = false ∧ (SurpriseMemoryManager.init storage analyzer).owns_analyzer = false := by
  simp [SurpriseMemoryManager.init]

lemma SurpriseMemoryManager_initWithOwnership_owns (storage : ContentAddressableStorage) (analyzer : DataFlowAnalyzer) (os oa : Bool) : (SurpriseMemoryManager.initWithOwnership storage analyzer os oa).owns_storage = os ∧ (SurpriseMemoryManager.initWithOwnership storage analyzer os oa).owns_analyzer = oa := by
  simp [SurpriseMemoryManager.initWithOwnership]

lemma SurpriseMemoryManager_setSurpriseThreshold_clamped (self : SurpriseMemoryManager) (threshold : Real) : (self.setSurpriseThreshold threshold).surprise_threshold = max 0.0 (min 1.0 threshold) := by
  simp [SurpriseMemoryManager.setSurpriseThreshold]

lemma SurpriseMemoryManager_setSurpriseThreshold_statistics_updated (self : SurpriseMemoryManager) (threshold : Real) : (self.setSurpriseThreshold threshold).statistics.surprise_threshold = max 0.0 (min 1.0 threshold) := by
  simp [SurpriseMemoryManager.setSurpriseThreshold]

lemma SurpriseMemoryManager_getSurpriseThreshold_eq (self : SurpriseMemoryManager) : self.getSurpriseThreshold = self.surprise_threshold := by
  rfl

lemma SurpriseMemoryManager_computeJaccardDistance_ge_zero (self : SurpriseMemoryManager) (da db : List (Fin 256)) : self.computeJaccardDistance da db ≥ 0.0 := by
  simp [SurpriseMemoryManager.computeJaccardDistance]
  split
  linarith
  apply sub_nonneg.mpr
  apply div_le_one.mpr
  apply Nat.cast_nonneg
  apply Nat.cast_le.mpr
  apply le_refl

lemma SurpriseMemoryManager_computeJaccardDistance_le_one (self : SurpriseMemoryManager) (da db : List (Fin 256)) : self.computeJaccardDistance da db ≤ 1.0 := by
  simp [SurpriseMemoryManager.computeJaccardDistance]
  split
  linarith
  apply sub_le_self
  apply div_nonneg
  apply Nat.cast_nonneg
  apply Nat.cast_le.mpr
  apply le_refl

lemma SurpriseMemoryManager_computeHashDistance_ge_zero (ha hb : Vector (Fin 256) HASH_SIZE) : SurpriseMemoryManager.computeHashDistance ha hb ≥ 0.0 := by
  simp [SurpriseMemoryManager.computeHashDistance]
  apply div_nonneg
  apply Nat.cast_nonneg
  apply Nat.cast_nonneg

lemma SurpriseMemoryManager_computeHashDistance_le_one (ha hb : Vector (Fin 256) HASH_SIZE) : SurpriseMemoryManager.computeHashDistance ha hb ≤ 1.0 := by
  simp [SurpriseMemoryManager.computeHashDistance]
  apply div_le_one.mpr
  apply Nat.cast_le.mpr
  apply le_trans
  apply Fin.range_foldl_le (fun _ => Nat.popCount_le 8)
  simp [HASH_SIZE, HASH_BITS]

lemma SurpriseMemoryManager_computeSurprise_clamped (self : SurpriseMemoryManager) (nd : List (Fin 256)) : let m := self.computeSurprise nd 0.0 ≤ m.jaccard_dissimilarity ∧ m.jaccard_dissimilarity ≤ 1.0 ∧ 0.0 ≤ m.content_hash_distance ∧ m.content_hash_distance ≤ 1.0 ∧ 0.0 ≤ m.temporal_novelty ∧ m.temporal_novelty ≤ 1.0 := by
  simp [SurpriseMemoryManager.computeSurprise, SurpriseMetrics.init]
  split
  simp
  split
  simp
  linarith [SurpriseMetrics_init_clamped_jaccard_ge_zero, SurpriseMetrics_init_clamped_jaccard_le_one, SurpriseMetrics_init_clamped_hash_ge_zero, SurpriseMetrics_init_clamped_hash_le_one, SurpriseMetrics_init_clamped_temporal_ge_zero, SurpriseMetrics_init_clamped_temporal_le_one]

lemma partialSort_length_eq (items : List CandidateItem) (k : Nat) : (partialSort items k).length = items.length := by
  simp [partialSort]
  split
  rfl
  induction actual_k using Nat.rec
  simp
  intro
  simp
  split
  simp [List.set_length]
  simp

lemma content_hash_consistency (data : List (Fin 256)) : computeContentHash data = computeContentHash data := by
  rfl