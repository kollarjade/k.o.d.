import Mathlib.Data.String.Basic
import Mathlib.Data.Nat.Basic
import Std.Data.RBMap
import Mathlib.Data.List.Sort
import Mathlib.Data.List.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Cases
import Mathlib.Tactic.NthRewrite
import Mathlib.Init.Data.Float
import Mathlib.Data.List.Sublists

namespace Rpgu

abbrev NodeId := String
abbrev NodeData := String

structure Node where
  id : NodeId
  data : NodeData
  temporal_embedding : Float
  spatial_embedding : Float
  semantic_embedding : Float
deriving Repr, Inhabited, BEq

def Node.clone (n : Node) : Node :=
  { id := n.id, data := n.data, temporal_embedding := n.temporal_embedding, spatial_embedding := n.spatial_embedding, semantic_embedding := n.semantic_embedding }

theorem Node.clone_eq (n : Node) : n.clone = n := by
  unfold clone
  rfl

inductive EdgeQuality where
  | structural : EdgeQuality
  | temporal : EdgeQuality
  | spatial : EdgeQuality
  | semantic : EdgeQuality
deriving Repr, BEq, Inhabited

def EdgeQuality.toInt : EdgeQuality -> UInt8
  | .structural => 0
  | .temporal => 1
  | .spatial => 2
  | .semantic => 3

theorem EdgeQuality.toInt_injective : Function.Injective EdgeQuality.toInt := by
  intro a b h
  cases a <;> cases b <;> unfold toInt at h <;> try (first | contradiction | rfl)
  done

structure Edge where
  source : NodeId
  target : NodeId
  weight : Float
  quality : EdgeQuality
deriving Repr, Inhabited, BEq

def Edge.clone (e : Edge) : Edge :=
  { source := e.source, target := e.target, weight := e.weight, quality := e.quality }

theorem Edge.clone_eq (e : Edge) : e.clone = e := by
  unfold clone
  rfl

structure EdgeKey where
  source : NodeId
  target : NodeId
deriving Repr, BEq, Inhabited

instance : Ord EdgeKey where
  compare a b :=
    match compare a.source b.source with
    | .eq => compare a.target b.target
    | ord => ord

abbrev Nodes := Std.RBMap NodeId Node compare
abbrev Edges := Std.RBMap EdgeKey (List Edge) compare

structure SelfSimilarRelationalGraph where
  nodes : Nodes
  edges : Edges
deriving Inhabited

def SelfSimilarRelationalGraph.init : SelfSimilarRelationalGraph :=
  { nodes := Std.RBMap.empty, edges := Std.RBMap.empty }

theorem SelfSimilarRelationalGraph.init_empty :
  SelfSimilarRelationalGraph.init.nodes.isEmpty ∧ SelfSimilarRelationalGraph.init.edges.isEmpty := by
  unfold init
  simp [Std.RBMap.isEmpty]
  done

def SelfSimilarRelationalGraph.addNode (g : SelfSimilarRelationalGraph) (n : Node) : SelfSimilarRelationalGraph :=
  { g with nodes := g.nodes.insert n.id n }

theorem SelfSimilarRelationalGraph.addNode_preserves_edges (g : SelfSimilarRelationalGraph) (n : Node) :
  (g.addNode n).edges = g.edges := by
  unfold addNode
  simp
  done

theorem SelfSimilarRelationalGraph.node_in_addNode (g : SelfSimilarRelationalGraph) (n : Node) :
  (g.addNode n).nodes.find? n.id = some n := by
  unfold addNode
  exact Std.RBMap.find?_insert
  done

def SelfSimilarRelationalGraph.addEdge (g : SelfSimilarRelationalGraph) (e : Edge) : SelfSimilarRelationalGraph :=
  let key : EdgeKey := { source := e.source, target := e.target }
  let current_edges := g.edges.find? key |>.getD []
  { g with edges := g.edges.insert key (e :: current_edges) }

theorem SelfSimilarRelationalGraph.addEdge_preserves_nodes (g : SelfSimilarRelationalGraph) (e : Edge) :
  (g.addEdge e).nodes = g.nodes := by
  unfold addEdge
  simp
  done

def SelfSimilarRelationalGraph.nodeCount (g : SelfSimilarRelationalGraph) : Nat :=
  g.nodes.size

def SelfSimilarRelationalGraph.edgeCount (g : SelfSimilarRelationalGraph) : Nat :=
  g.edges.fold (fun acc _ es => acc + es.length) 0

inductive CoreState where
  | idle
  | processing
  | communicating
  | power_gated
deriving Repr, BEq, Inhabited

def CoreState.toString : CoreState -> String
  | .idle => "idle"
  | .processing => "processing"
  | .communicating => "communicating"
  | .power_gated => "power_gated"

def CoreState.fromString (s : String) : Option CoreState :=
  if s == "idle" then some .idle
  else if s == "processing" then some .processing
  else if s == "communicating" then some .communicating
  else if s == "power_gated" then some .power_gated
  else none

theorem CoreState.fromString_toString_iso (cs : CoreState) : CoreState.fromString (CoreState.toString cs) = some cs := by
  cases cs <;> unfold toString <;> unfold fromString <;> simp
  done

theorem CoreState.toString_fromString_iso (s : String) (h : (CoreState.fromString s).isSome) : CoreState.toString (Option.get _ h) = s := by
  unfold fromString at h
  split at h
  · rename_i h_eq
    rw [h_eq]
    unfold fromString
    simp
    unfold toString
    done
  · split at h
    · rename_i h_eq
      rw [h_eq]
      unfold fromString
      simp
      unfold toString
      done
    · split at h
      · rename_i h_eq
        rw [h_eq]
        unfold fromString
        simp
        unfold toString
        done
      · split at h
        · rename_i h_eq
          rw [h_eq]
          unfold fromString
          simp
          unfold toString
          done
        · simp at h
          done

inductive MessageType where
  | weight_update
  | graph_sync
  | isomorphism_result
  | power_control
  | data_transfer
deriving Repr, BEq, Inhabited

def MessageType.toString : MessageType -> String
  | .weight_update => "weight_update"
  | .graph_sync => "graph_sync"
  | .isomorphism_result => "isomorphism_result"
  | .power_control => "power_control"
  | .data_transfer => "data_transfer"

def MessageType.fromString (s : String) : Option MessageType :=
  if s == "weight_update" then some .weight_update
  else if s == "graph_sync" then some .graph_sync
  else if s == "isomorphism_result" then some .isomorphism_result
  else if s == "power_control" then some .power_control
  else if s == "data_transfer" then some .data_transfer
  else none

theorem MessageType.fromString_toString_iso (mt : MessageType) : MessageType.fromString (MessageType.toString mt) = some mt := by
  cases mt <;> unfold toString <;> unfold fromString <;> simp
  done

structure NoCMessage where
  source_core : Nat
  target_core : Nat
  message_type : MessageType
  payload : List UInt8
  timestamp : Int
  priority : Int
deriving Repr, Inhabited

def NoCMessage.clone (m : NoCMessage) : NoCMessage :=
  { source_core := m.source_core, target_core := m.target_core, message_type := m.message_type, payload := m.payload, timestamp := m.timestamp, priority := m.priority }

theorem NoCMessage.clone_eq (m : NoCMessage) : m.clone = m := by
  unfold clone
  rfl

structure ProcessingCore where
  core_id : Nat
  x : Nat
  y : Nat
  state : CoreState
  neighbors : List Nat
  local_graph : Option SelfSimilarRelationalGraph
  message_queue : List NoCMessage
  energy_consumed : Float
  cycles_active : Nat
  cycles_idle : Nat
deriving Inhabited

def ProcessingCore.init (core_id x y : Nat) : ProcessingCore := {
  core_id := core_id,
  x := x,
  y := y,
  state := .idle,
  neighbors := [],
  local_graph := none,
  message_queue := [],
  energy_consumed := 0.0,
  cycles_active := 0,
  cycles_idle := 0
}

theorem ProcessingCore.init_properties (core_id x y : Nat) :
  let core := ProcessingCore.init core_id x y
  core.core_id = core_id ∧ core.x = x ∧ core.y = y ∧ core.state = .idle ∧
  core.neighbors = [] ∧ core.local_graph = none ∧ core.message_queue = [] ∧
  core.energy_consumed = 0.0 ∧ core.cycles_active = 0 ∧ core.cycles_idle = 0 := by
  unfold init
  simp
  done

def ProcessingCore.addNeighbor (self : ProcessingCore) (neighbor_id : Nat) : ProcessingCore :=
  { self with neighbors := neighbor_id :: self.neighbors }

theorem ProcessingCore.addNeighbor_spec (self : ProcessingCore) (neighbor_id : Nat) :
  (self.addNeighbor neighbor_id).neighbors = neighbor_id :: self.neighbors ∧
  (self.addNeighbor neighbor_id).neighbors.length = self.neighbors.length + 1 := by
  unfold addNeighbor
  simp [List.length]
  done

def ProcessingCore.setLocalGraph (self : ProcessingCore) (graph : SelfSimilarRelationalGraph) : ProcessingCore :=
  { self with local_graph := some graph }

theorem ProcessingCore.setLocalGraph_spec (self : ProcessingCore) (graph : SelfSimilarRelationalGraph) :
  (self.setLocalGraph graph).local_graph = some graph := by
  unfold setLocalGraph
  simp
  done

def ProcessingCore.createLocalGraph (self : ProcessingCore) : ProcessingCore :=
  { self with local_graph := some SelfSimilarRelationalGraph.init }

theorem ProcessingCore.createLocalGraph_spec (self : ProcessingCore) :
  (self.createLocalGraph).local_graph = some SelfSimilarRelationalGraph.init := by
  unfold createLocalGraph
  simp
  done

def ProcessingCore.enqueueMessage (self : ProcessingCore) (message : NoCMessage) : ProcessingCore :=
  { self with message_queue := message :: self.message_queue }

theorem ProcessingCore.enqueueMessage_spec (self : ProcessingCore) (message : NoCMessage) :
  (self.enqueueMessage message).message_queue.length = self.message_queue.length + 1 := by
  unfold enqueueMessage
  simp
  done

def ProcessingCore.processMessages (self : ProcessingCore) : ProcessingCore × Nat :=
  ({ self with message_queue := [] }, self.message_queue.length)

theorem ProcessingCore.processMessages_spec (self : ProcessingCore) :
  (self.processMessages).1.message_queue = [] ∧
  (self.processMessages).2 = self.message_queue.length := by
  unfold processMessages
  simp
  done

def ProcessingCore.getWorkload (self : ProcessingCore) : Float :=
  let total := self.cycles_active + self.cycles_idle
  if total = 0 then 0.0 else Float.ofNat self.cycles_active / Float.ofNat total

theorem ProcessingCore.getWorkload_bounds (self : ProcessingCore) :
  let w := self.getWorkload
  w ≥ 0.0 ∧ w ≤ 1.0 := by
  unfold getWorkload
  split
  · simp
  · rename_i h
    have h_total_pos : (Float.ofNat (self.cycles_active + self.cycles_idle)) > 0 := by
      apply Float.ofNat_pos.mpr
      exact Nat.pos_of_ne_zero h
    have h_le : self.cycles_active ≤ self.cycles_active + self.cycles_idle := Nat.le_add_right _ _
    constructor
    · apply Float.div_nonneg
      · exact Float.ofNat_nonneg _
      · exact Float.ofNat_nonneg _
    · apply (Float.div_le_one_of_le h_total_pos).mpr
      simp [Float.ofNat_le.mpr h_le]
      done

structure MessagePriorityEntry where
  priority : Int
  sequence : Nat
  message : NoCMessage
deriving Repr, Inhabited

instance : Ord MessagePriorityEntry where
  compare a b :=
    match compare a.priority b.priority with
    | .lt => .gt
    | .gt => .lt
    | .eq => compare a.sequence b.sequence

structure RouteKey where
  source : Nat
  destination : Nat
deriving Repr, BEq, Inhabited

instance : Ord RouteKey where
  compare a b :=
    match compare a.source b.source with
    | .eq => compare a.destination b.destination
    | ord => ord

abbrev Cores := Std.RBMap Nat ProcessingCore compare
abbrev RoutingTable := Std.RBMap RouteKey (List Nat) compare
abbrev MessageBuffer := List MessagePriorityEntry

structure AsynchronousNoC where
  grid_width : Nat
  grid_height : Nat
  cores : Cores
  routing_table : RoutingTable
  message_buffer : MessageBuffer
  total_messages : Nat
  total_hops : Nat
  message_sequence : Nat
deriving Inhabited

def AsynchronousNoC.initializeCores (w h : Nat) : Cores :=
  let N := w * h
  (List.range N).foldl (
    fun map id =>
      let x := id % w
      let y := id / w
      let core := ProcessingCore.init id x y
      let neighbors :=
        (if x > 0 then [id - 1] else []) ++
        (if x < w - 1 then [id + 1] else []) ++
        (if y > 0 then [id - w] else []) ++
        (if y < h - 1 then [id + w] else [])
      let core_with_neighbors := { core with neighbors := neighbors }
      map.insert id core_with_neighbors
  ) Std.RBMap.empty

lemma AsynchronousNoC.initializeCores.fold_keys_are_range : ∀ (n : Nat),
  let w := n
  let h := 1
  let N := w * h
  let f := fun (map : Cores) (id : Nat) =>
      let x := id % w
      let y := id / w
      let core := ProcessingCore.init id x y
      let neighbors :=
        (if x > 0 then [id - 1] else []) ++
        (if x < w - 1 then [id + 1] else []) ++
        (if y > 0 then [id - w] else []) ++
        (if y < h - 1 then [id + w] else [])
      let core_with_neighbors := { core with neighbors := neighbors }
      map.insert id core_with_neighbors
  (List.foldl f Std.RBMap.empty (List.range N)).keys = List.range N := by
  intro n
  let w := n
  let h := 1
  let N := w * h
  simp [N, w, h]
  generalize f_gen : (fun (map : Cores) (id : Nat) =>
      let x := id % w
      let y := id / w
      let core := ProcessingCore.init id x y
      let neighbors :=
        (if x > 0 then [id - 1] else []) ++
        (if x < w - 1 then [id + 1] else []) ++
        (if y > 0 then [id - w] else []) ++
        (if y < h - 1 then [id + w] else [])
      let core_with_neighbors := { core with neighbors := neighbors }
      map.insert id core_with_neighbors) = f
  induction n with
  | zero =>
    simp [List.range]
    rfl
  | succ k ih =>
    rw [List.range_succ]
    rw [List.foldl_append]
    rw [List.foldl_singleton]
    have h_not_mem : k ∉ (List.foldl f Std.RBMap.empty (List.range k)).keys := by
      rw [ih]
      simp
      done
    rw [Std.RBMap.keys_insert_of_not_mem h_not_mem]
    rw [ih]
    rw [List.keys_append_of_not_mem]
    simp
    done

theorem AsynchronousNoC.initializeCores_size (w h : Nat) :
  (AsynchronousNoC.initializeCores w h).size = w * h := by
  unfold initializeCores
  generalize h_N : w * h = N
  generalize h_f : (fun map id =>
      let x := id % w
      let y := id / w
      let core := ProcessingCore.init id x y
      let neighbors :=
        (if x > 0 then [id - 1] else []) ++
        (if x < w - 1 then [id + 1] else []) ++
        (if y > 0 then [id - w] else []) ++
        (if y < h - 1 then [id + w] else [])
      let core_with_neighbors := { core with neighbors := neighbors }
      map.insert id core_with_neighbors) = f
  induction' (List.range N) with hd tl ih
  · simp [List.foldl]
    rfl
  · rw [List.foldl_cons]
    have h_not_mem : hd ∉ (List.foldl f Std.RBMap.empty tl).keys := by
      simp [List.mem_keys]
      have h_range_perm : List.Perm (hd :: tl) (List.range N) := by simp
      have h_tl_gt : ∀ (x : Nat), x ∈ tl → hd < x := by
        intro x hx
        have h_mem_range : x ∈ hd :: tl := List.mem_cons_of_mem hd hx
        rw [List.mem_range] at h_mem_range
        rw [List.mem_range]
        simp at h_mem_range
        have h_hd_val : hd = (List.range N).head (by simp) := by simp
        rw [h_hd_val]
        simp
        exact h_mem_range.1
      intro h_find
      let map_tl := List.foldl f Std.RBMap.empty tl
      have keys_tl_subset_tl : (Std.RBMap.keys map_tl) ⊆ tl := by
        induction tl with hd' tl' ih'
        · simp
        · rw [List.foldl_cons]
          rw [Std.RBMap.keys_insert]
          simp
          intro x hx
          cases hx
          · rename_i hx'
            rw [hx']
            simp
          · rename_i hx'
            simp
            apply ih'
            exact hx'
      have hd_in_keys : hd ∈ map_tl.keys := by
        simp [Std.RBMap.contains_iff_mem_keys]
        exact Std.RBMap.find?_isSome.mp h_find
      have hd_in_tl : hd ∈ tl := keys_tl_subset_tl hd_in_keys
      have := h_tl_gt hd hd_in_tl
      linarith
    rw [Std.RBMap.size_insert_of_not_mem h_not_mem]
    rw [ih]
    simp
    done

def AsynchronousNoC.computeXYRoute (w : Nat) (src_core dst_core : ProcessingCore) : List Nat :=
  let rec go_x (curr_x : Nat) (path : List Nat) : List Nat :=
    if h : curr_x = dst_core.x then path
    else
      let next_x := if curr_x < dst_core.x then curr_x + 1 else curr_x - 1
      let next_id := src_core.y * w + next_x
      go_x next_x (next_id :: path)
  termination_by (dst_core.x - curr_x).natAbs
  let path_x := go_x src_core.x [src_core.core_id]

  let rec go_y (curr_y : Nat) (path : List Nat) : List Nat :=
    if h : curr_y = dst_core.y then path
    else
      let next_y := if curr_y < dst_core.y then curr_y + 1 else curr_y - 1
      let next_id := next_y * w + dst_core.x
      go_y next_y (next_id :: path)
  termination_by (dst_core.y - curr_y).natAbs
  (go_y src_core.y path_x).reverse

def AsynchronousNoC.buildRoutingTable (w h : Nat) (cores : Cores) : RoutingTable :=
  cores.fold (fun table src_id src_core =>
    cores.fold (fun table' dst_id dst_core =>
      if src_id = dst_id then table'
      else
        let path := computeXYRoute w src_core dst_core
        table'.insert { source := src_id, destination := dst_id } path
    ) table
  ) Std.RBMap.empty

def AsynchronousNoC.init (grid_width grid_height : Nat) : AsynchronousNoC :=
  let cores := initializeCores grid_width grid_height
  let routing_table := buildRoutingTable grid_width grid_height cores
  {
    grid_width := grid_width,
    grid_height := grid_height,
    cores := cores,
    routing_table := routing_table,
    message_buffer := [],
    total_messages := 0,
    total_hops := 0,
    message_sequence := 0
  }

theorem AsynchronousNoC.init_total_cores (w h : Nat) :
  (AsynchronousNoC.init w h).cores.size = w * h := by
  unfold init
  exact initializeCores_size w h
  done

def AsynchronousNoC.sendMessage (self : AsynchronousNoC) (message : NoCMessage) : Option AsynchronousNoC :=
  if self.cores.contains message.source_core ∧ self.cores.contains message.target_core then
    let entry : MessagePriorityEntry := {
      priority := message.priority,
      sequence := self.message_sequence,
      message := message
    }
    some { self with
      message_buffer := List.mergeSorted (fun a b => a.priority > b.priority || (a.priority == b.priority && a.sequence < b.sequence)) [entry] self.message_buffer,
      message_sequence := self.message_sequence + 1,
      total_messages := self.total_messages + 1
    }
  else
    none

theorem AsynchronousNoC.sendMessage_success (self : AsynchronousNoC) (message : NoCMessage)
  (h : self.cores.contains message.source_core ∧ self.cores.contains message.target_core) :
  let new_noc := (self.sendMessage message).get (by unfold sendMessage; simp [h])
  new_noc.message_buffer.length = self.message_buffer.length + 1 ∧
  new_noc.message_sequence = self.message_sequence + 1 ∧
  new_noc.total_messages = self.total_messages + 1 := by
  unfold sendMessage
  simp only [h, and_self, cond_true]
  have h_some : Option.isSome (some _) = true := rfl
  let new_noc := Option.get (some _) h_some
  have h_len : (List.mergeSorted _ [message] self.message_buffer).length = [message].length + self.message_buffer.length := by
    exact List.length_mergeSorted _ _ _
  simp [new_noc, h_len]
  done

def AsynchronousNoC.routeMessages (self : AsynchronousNoC) : AsynchronousNoC :=
  let (new_cores, new_total_hops) := self.message_buffer.foldl (
    fun (acc_cores, acc_hops) entry =>
      let msg := entry.message
      let path_len := self.routing_table.find? { source := msg.source_core, destination := msg.target_core } |>.map List.length |>.getD 1
      let hops := if path_len > 0 then path_len - 1 else 0
      match acc_cores.find? msg.target_core with
      | some target_core =>
          let new_target_core := target_core.enqueueMessage msg
          (acc_cores.insert msg.target_core new_target_core, acc_hops + hops)
      | none => (acc_cores, acc_hops)
  ) (self.cores, self.total_hops)
  { self with
    cores := new_cores,
    total_hops := new_total_hops,
    message_buffer := []
  }

theorem AsynchronousNoC.routeMessages_clears_buffer (self : AsynchronousNoC) :
  (self.routeMessages).message_buffer = [] := by
  unfold routeMessages
  rfl
  done

def AsynchronousNoC.getCore (self : AsynchronousNoC) (core_id : Nat) : Option ProcessingCore :=
  self.cores.find? core_id

def AsynchronousNoC.getTotalCores (self : AsynchronousNoC) : Nat :=
  self.cores.size

def AsynchronousNoC.getActiveCores (self : AsynchronousNoC) : Nat :=
  self.cores.fold (fun count _ core => if core.state != .power_gated then count + 1 else count) 0

structure GraphIsomorphismProcessor where
  dummy : Unit

def GraphIsomorphismProcessor.init : GraphIsomorphismProcessor := { dummy := () }

def GraphIsomorphismProcessor.computeCanonicalForm (g : SelfSimilarRelationalGraph) : String :=
  let node_ids := g.nodes.keys.qsort (· < ·)
  let degree_sequence := node_ids.map (fun node_id =>
    let out_degree := g.edges.fold (fun acc key es => if key.source == node_id then acc + es.length else acc) 0
    let in_degree := g.edges.fold (fun acc key es => if key.target == node_id then acc + es.length else acc) 0
    (out_degree, in_degree)
  )
  let sorted_degree_sequence := degree_sequence.qsort (fun a b => a.1 < b.1 || (a.1 == b.1 && a.2 < b.2))
  let edge_qualities := g.edges.fold (fun acc _ es => acc ++ es.map (fun e => EdgeQuality.toInt e.quality)) []
  let sorted_edge_qualities := edge_qualities.qsort (· < ·)

  s!"{node_ids.size}_[{sorted_degree_sequence}]_[{sorted_edge_qualities}]"

theorem GraphIsomorphismProcessor.computeCanonicalForm_deterministic (g : SelfSimilarRelationalGraph) :
  computeCanonicalForm g = computeCanonicalForm g := by rfl

def GraphIsomorphismProcessor.areIsomorphic (g1 g2 : SelfSimilarRelationalGraph) : Bool :=
  if g1.nodeCount() != g2.nodeCount() then false
  else if g1.edgeCount() != g2.edgeCount() then false
  else computeCanonicalForm g1 == computeCanonicalForm g2

theorem GraphIsomorphismProcessor.areIsomorphic_refl (g : SelfSimilarRelationalGraph) :
  areIsomorphic g g = true := by
  unfold areIsomorphic
  unfold SelfSimilarRelationalGraph.nodeCount
  unfold SelfSimilarRelationalGraph.edgeCount
  simp [computeCanonicalForm_deterministic]
  done

structure DynamicEdgeWeighting where
  learning_rate : Float

def DynamicEdgeWeighting.init : DynamicEdgeWeighting := { learning_rate := 0.01 }

def DynamicEdgeWeighting.updateWeight (self : DynamicEdgeWeighting) (current_weight feedback : Float) : Float :=
  let delta := self.learning_rate * feedback
  let new_weight := current_weight + delta
  Float.max 0.0 (Float.min 1.0 new_weight)

theorem DynamicEdgeWeighting.updateWeight_bounds (self : DynamicEdgeWeighting) (current_weight feedback : Float) :
  let w := self.updateWeight current_weight feedback
  w ≥ 0.0 ∧ w ≤ 1.0 := by
  unfold updateWeight
  constructor
  · apply Float.le_max_of_le_left
    exact le_refl 0.0
  · apply Float.max_le
    · exact Float.min_le_left 1.0 _
    · apply Float.min_le_of_le_of_le
      · exact le_refl 0.0
      · exact Float.min_le_right _ _
  done

def DynamicEdgeWeighting.computeAdaptiveWeight (base_weight temporal_factor spatial_factor semantic_factor : Float) : Float :=
  let adaptive_weight := base_weight * temporal_factor * spatial_factor * semantic_factor
  Float.max 0.0 (Float.min 1.0 adaptive_weight)

theorem DynamicEdgeWeighting.computeAdaptiveWeight_bounds (base_weight temporal_factor spatial_factor semantic_factor : Float) :
  let w := computeAdaptiveWeight base_weight temporal_factor spatial_factor semantic_factor
  w ≥ 0.0 ∧ w ≤ 1.0 := by
  unfold computeAdaptiveWeight
  constructor
  · apply Float.le_max_of_le_left
    exact le_refl 0.0
  · apply Float.max_le
    · exact Float.min_le_left 1.0 _
    · apply Float.min_le_of_le_of_le
      · exact le_refl 0.0
      · exact Float.min_le_right _ _
  done

structure SparseActivationManager where
  sparsity_threshold : Float
  activation_map : Std.RBMap Nat Bool compare
  energy_saved : Float

def SparseActivationManager.init (sparsity_threshold : Float) : SparseActivationManager := {
  sparsity_threshold := sparsity_threshold,
  activation_map := Std.RBMap.empty,
  energy_saved := 0.0
}

def SparseActivationManager.shouldActivateCore (self : SparseActivationManager) (core_id : Nat) (workload : Float) : SparseActivationManager × Bool :=
  if workload < self.sparsity_threshold then
    ({ self with
        activation_map := self.activation_map.insert core_id false,
        energy_saved := self.energy_saved + 1.0
     }, false)
  else
    ({ self with
        activation_map := self.activation_map.insert core_id true
     }, true)

theorem SparseActivationManager.shouldActivateCore_low_workload (self : SparseActivationManager) (core_id : Nat) (workload : Float)
  (h : workload < self.sparsity_threshold) :
  let (new_manager, decision) := self.shouldActivateCore core_id workload
  decision = false ∧ new_manager.energy_saved = self.energy_saved + 1.0 ∧ new_manager.activation_map.find? core_id = some false := by
  unfold shouldActivateCore
  simp [h]
  constructor
  · rfl
  · constructor
    · rfl
    · exact Std.RBMap.find?_insert
  done

theorem SparseActivationManager.shouldActivateCore_high_workload (self : SparseActivationManager) (core_id : Nat) (workload : Float)
  (h : ¬(workload < self.sparsity_threshold)) :
  let (new_manager, decision) := self.shouldActivateCore core_id workload
  decision = true ∧ new_manager.energy_saved = self.energy_saved ∧ new_manager.activation_map.find? core_id = some true := by
  unfold shouldActivateCore
  simp [h]
  constructor
  · rfl
  · constructor
    · rfl
    · exact Std.RBMap.find?_insert
  done

structure PowerGatingController where
  gated_cores : Std.RBSet Nat compare
  power_budget : Float
  current_power : Float

def PowerGatingController.init (power_budget : Float) : PowerGatingController := {
  gated_cores := Std.RBSet.empty,
  power_budget := power_budget,
  current_power := 0.0
}

def PowerGatingController.gateCore (self : PowerGatingController) (core : ProcessingCore) : PowerGatingController × ProcessingCore × Bool :=
  if self.gated_cores.contains core.core_id then
    (self, core, false)
  else
    let new_controller := { self with
      gated_cores := self.gated_cores.insert core.core_id,
      current_power := self.current_power - 10.0
    }
    let new_core := { core with state := .power_gated }
    (new_controller, new_core, true)

theorem PowerGatingController.gateCore_success (self : PowerGatingController) (core : ProcessingCore)
  (h : ¬self.gated_cores.contains core.core_id) :
  let (new_controller, new_core, success) := self.gateCore core
  success = true ∧ new_core.state = .power_gated ∧ new_controller.gated_cores.contains core.core_id := by
  unfold gateCore
  simp [h]
  constructor
  · rfl
  · constructor
    · rfl
    · exact Std.RBSet.contains_insert
  done

def PowerGatingController.ungateCore (self : PowerGatingController) (core : ProcessingCore) : PowerGatingController × ProcessingCore × Bool :=
  if ¬self.gated_cores.contains core.core_id then
    (self, core, false)
  else if self.current_power + 10.0 > self.power_budget then
    (self, core, false)
  else
    let new_controller := { self with
      gated_cores := self.gated_cores.erase core.core_id,
      current_power := self.current_power + 10.0
    }
    let new_core := { core with state := .idle }
    (new_controller, new_core, true)

theorem PowerGatingController.ungateCore_success (self : PowerGatingController) (core : ProcessingCore)
  (h_gated : self.gated_cores.contains core.core_id) (h_budget : ¬(self.current_power + 10.0 > self.power_budget)) :
  let (new_controller, new_core, success) := self.ungateCore core
  success = true ∧ new_core.state = .idle ∧ ¬new_controller.gated_cores.contains core.core_id := by
  unfold ungateCore
  simp [h_gated, h_budget]
  constructor
  · rfl
  · constructor
    · rfl
    · exact Std.RBSet.contains_erase_of_false (by simp)
  done

structure RelationalGraphProcessingUnit where
  noc : AsynchronousNoC
  isomorphism_processor : GraphIsomorphismProcessor
  edge_weighting : DynamicEdgeWeighting
  sparse_activation : SparseActivationManager
  power_gating : PowerGatingController
  global_graph : Option SelfSimilarRelationalGraph
  execution_cycles : Nat

def RelationalGraphProcessingUnit.init (grid_width grid_height : Nat) : RelationalGraphProcessingUnit := {
  noc := AsynchronousNoC.init grid_width grid_height,
  isomorphism_processor := GraphIsomorphismProcessor.init,
  edge_weighting := DynamicEdgeWeighting.init,
  sparse_activation := SparseActivationManager.init 0.1,
  power_gating := PowerGatingController.init 1000.0,
  global_graph := none,
  execution_cycles := 0
}

theorem RelationalGraphProcessingUnit.init_spec (w h : Nat) :
  let rpgu := RelationalGraphProcessingUnit.init w h
  rpgu.noc.grid_width = w ∧ rpgu.noc.grid_height = h ∧ rpgu.execution_cycles = 0 := by
  unfold init
  unfold AsynchronousNoC.init
  simp
  done

def RelationalGraphProcessingUnit.distributeGraph (self : RelationalGraphProcessingUnit) (graph : SelfSimilarRelationalGraph) : RelationalGraphProcessingUnit :=
  let active_cores := self.noc.cores.fold (fun acc id core => if core.state != .power_gated then (id, core) :: acc else acc) []
  if h_ac : active_cores.isEmpty then self
  else
    let nodes_list := graph.nodes.keys
    let num_active_cores := active_cores.length
    let nodes_per_core := nodes_list.size / num_active_cores
    let remainder := nodes_list.size % num_active_cores

    let (final_cores, _) := active_cores.foldl (
      fun (acc_cores, start_idx) (core_id, core) =>
        let extra := if acc_cores.size < remainder then 1 else 0
        let end_idx := start_idx + nodes_per_core + extra
        let core_nodes := nodes_list.extract start_idx (end_idx - start_idx)
        let local_graph_nodes := core_nodes.foldl (fun map id =>
          match graph.nodes.find? id with
          | some n => map.insert id n
          | none => map
        ) Std.RBMap.empty
        let local_graph_edges := graph.edges.fold (fun map key es =>
          if core_nodes.contains key.source || core_nodes.contains key.target then map.insert key es else map
        ) Std.RBMap.empty
        let local_graph : SelfSimilarRelationalGraph := { nodes := local_graph_nodes, edges := local_graph_edges }
        let new_core := core.setLocalGraph local_graph
        (acc_cores.insert core_id new_core, end_idx)
    ) (self.noc.cores, 0)

    { self with noc := { self.noc with cores := final_cores } }

def SelfSimilarRelationalGraph.fromNodes (main_graph : SelfSimilarRelationalGraph) (subgraph_nodes : List NodeId) : SelfSimilarRelationalGraph :=
  let nodes := subgraph_nodes.foldl (fun acc_nodes node_id =>
    match main_graph.nodes.find? node_id with
    | some node => acc_nodes.insert node_id node
    | none => acc_nodes
  ) Std.RBMap.empty
  let edges := main_graph.edges.fold (fun acc_edges key edge_list =>
    if subgraph_nodes.contains key.source ∧ subgraph_nodes.contains key.target then
      acc_edges.insert key edge_list
    else
      acc_edges
  ) Std.RBMap.empty
  { nodes := nodes, edges := edges }

def GraphIsomorphismProcessor.findIsomorphicSubgraphs (main_graph pattern_graph : SelfSimilarRelationalGraph) : List (List NodeId) :=
  let pattern_size := pattern_graph.nodeCount
  let main_node_count := main_graph.nodeCount
  if pattern_size > main_node_count then []
  else
    let main_nodes := main_graph.nodes.keys
    let pattern_canonical := computeCanonicalForm pattern_graph
    let subnode_lists := List.sublistsLen pattern_size main_nodes
    subnode_lists.filterMap (fun node_list =>
      let subgraph := SelfSimilarRelationalGraph.fromNodes main_graph node_list
      let subgraph_canonical := computeCanonicalForm subgraph
      if subgraph_canonical == pattern_canonical then some node_list else none
    )

def RelationalGraphProcessingUnit.processIsomorphismParallel (self : RelationalGraphProcessingUnit) (pattern_graph : SelfSimilarRelationalGraph)
  : RelationalGraphProcessingUnit × List (List NodeId) :=
  let (new_noc_cores, all_matches, new_sparse_activation) := self.noc.cores.fold (
    fun (acc_cores, acc_matches, acc_sa) core_id core =>
      match core.local_graph with
      | none => (acc_cores.insert core_id core, acc_matches, acc_sa)
      | some lg =>
        if core.state == .power_gated then (acc_cores.insert core_id core, acc_matches, acc_sa)
        else
          let workload := Float.ofNat lg.nodeCount / 100.0
          let (next_sa, activate) := acc_sa.shouldActivateCore core_id workload
          if ¬activate then (acc_cores.insert core_id core, acc_matches, next_sa)
          else
            let new_core_state := { core with state := .processing }
            let matches := GraphIsomorphismProcessor.findIsomorphicSubgraphs lg pattern_graph
            let final_core := { new_core_state with
              cycles_active := new_core_state.cycles_active + 1,
              energy_consumed := new_core_state.energy_consumed + 5.0,
              state := .idle
            }
            (acc_cores.insert core_id final_core, acc_matches ++ matches, next_sa)
  ) (Std.RBMap.empty, [], self.sparse_activation)
  let new_noc := { self.noc with cores := new_noc_cores }
  let new_rpgu := { self with
      noc := new_noc,
      sparse_activation := new_sparse_activation,
      execution_cycles := self.execution_cycles + 1
  }
  (new_rpgu, all_matches)

theorem RelationalGraphProcessingUnit.processIsomorphismParallel_cycles (self : RelationalGraphProcessingUnit) (pattern : SelfSimilarRelationalGraph) :
  (self.processIsomorphismParallel pattern).1.execution_cycles = self.execution_cycles + 1 := by
  unfold processIsomorphismParallel
  simp
  done

def RelationalGraphProcessingUnit.synchronizeGraphs (self : RelationalGraphProcessingUnit) : RelationalGraphProcessingUnit :=
  let new_global_nodes := self.noc.cores.fold (fun acc _ core =>
    match core.local_graph with
    | some lg => lg.nodes.fold (fun n_acc id n => n_acc.insert id n) acc
    | none => acc
  ) Std.RBMap.empty
  let new_global_edges := self.noc.cores.fold (fun acc _ core =>
    match core.local_graph with
    | some lg => lg.edges.fold (fun e_acc key es => e_acc.insert key es) acc
    | none => acc
  ) Std.RBMap.empty
  let new_global_graph : SelfSimilarRelationalGraph := { nodes := new_global_nodes, edges := new_global_edges }
  { self with global_graph := some new_global_graph }

theorem RelationalGraphProcessingUnit.synchronizeGraphs_creates_graph (self : RelationalGraphProcessingUnit) :
  (self.synchronizeGraphs).global_graph.isSome := by
  unfold synchronizeGraphs
  simp
  done

structure RPGUStatistics where
  total_cores : Nat
  active_cores : Nat
  gated_cores : Nat
  total_energy_consumed : Float
  execution_cycles : Nat
  sparsity_ratio : Float
  total_messages : Nat
  average_message_hops : Float
  current_power : Float
  power_budget : Float

def RelationalGraphProcessingUnit.getStatistics (self : RelationalGraphProcessingUnit) : RPGUStatistics :=
  let total_energy := self.noc.cores.fold (fun acc _ c => acc + c.energy_consumed) 0.0
  let active_cores := self.noc.getActiveCores
  let gated_cores := self.power_gating.gated_cores.size
  let sparsity_ratio :=
    if h : self.sparse_activation.activation_map.size = 0 then 0.0
    else
      let inactive_count := self.sparse_activation.activation_map.fold (fun acc _ v => if ¬v then acc + 1 else acc) 0
      Float.ofNat inactive_count / Float.ofNat self.sparse_activation.activation_map.size
  let avg_hops :=
    if h : self.noc.total_messages = 0 then 0.0
    else Float.ofNat self.noc.total_hops / Float.ofNat self.noc.total_messages
  {
    total_cores := self.noc.getTotalCores,
    active_cores := active_cores,
    gated_cores := gated_cores,
    total_energy_consumed := total_energy,
    execution_cycles := self.execution_cycles,
    sparsity_ratio := sparsity_ratio,
    total_messages := self.noc.total_messages,
    average_message_hops := avg_hops,
    current_power := self.power_gating.current_power,
    power_budget := self.power_gating.power_budget
  }

theorem RelationalGraphProcessingUnit.getStatistics_total_cores (self : RelationalGraphProcessingUnit) :
  (self.getStatistics).total_cores = self.noc.cores.size := by
  unfold getStatistics
  unfold AsynchronousNoC.getTotalCores
  rfl
  done

end Rpgu