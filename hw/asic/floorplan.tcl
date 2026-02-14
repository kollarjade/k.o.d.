
set DESIGN_NAME "top_level"
set ICC_OUTPUT_DIR "./icc_output"
set NETLIST_DIR "./output"

file mkdir $ICC_OUTPUT_DIR

set TECH_FILE "/usr/local/synopsys/pdk/tsmc28/tech/tech.tf"
set REF_LIBS "/usr/local/synopsys/pdk/tsmc28/std_cells/mw_lib"
set SDC_FILE "${NETLIST_DIR}/${DESIGN_NAME}.sdc"

create_lib -technology $TECH_FILE -ref_libs $REF_LIBS ${ICC_OUTPUT_DIR}/${DESIGN_NAME}.lib
open_lib ${ICC_OUTPUT_DIR}/${DESIGN_NAME}.lib

create_block -name ${DESIGN_NAME}
read_verilog ${NETLIST_DIR}/${DESIGN_NAME}_synth.v
link_block
read_sdc $SDC_FILE

set_tlu_plus_files -max_tluplus /usr/local/synopsys/pdk/tsmc28/captable/max.tluplus -min_tluplus /usr/local/synopsys/pdk/tsmc28/captable/min.tluplus -tech2itf_map /usr/local/synopsys/pdk/tsmc28/captable/tech2itf.map

initialize_floorplan -die_size {0 0 5000 5000} -core_offset {100 100 100 100} -row_core_ratio 0.70 -flip_first_row true

set VDD_NET "VDD"
set VSS_NET "VSS"

create_supply_net $VDD_NET -reuse
create_supply_net $VSS_NET -reuse

create_pg_ring_pattern ring_pat -horizontal_layer METAL6 -horizontal_width 10.0 -horizontal_spacing 5.0 -vertical_layer METAL5 -vertical_width 10.0 -vertical_spacing 5.0
set_pg_strategy ring_strat -pattern {{name ring_pat} {nets {VDD VSS}}} -core

create_pg_strap_pattern strap_v -layer METAL5 -direction vertical -width 4.0 -spacing 30.0 -pitch 120.0 -nets {VDD VSS}
create_pg_strap_pattern strap_h -layer METAL6 -direction horizontal -width 4.0 -spacing 30.0 -pitch 120.0 -nets {VDD VSS}

set_pg_strategy vstrap_strat -pattern {{name strap_v} {nets {VDD VSS}}} -core
set_pg_strategy hstrap_strat -pattern {{name strap_h} {nets {VDD VSS}}} -core

create_pg_mesh_pattern mesh_pat -layers { {horizontal_layer METAL6 width 6.0 spacing 20.0 pitch 100.0} {vertical_layer METAL5 width 6.0 spacing 20.0 pitch 100.0} }
set_pg_strategy mesh_strat -pattern {{name mesh_pat} {nets {VDD VSS}}} -core

compile_pg -strategies {ring_strat vstrap_strat hstrap_strat mesh_strat}

connect_pg_nets -automatic
preroute_standard_cells -connect horizontal -port_filter_mode off

set MACROS [get_cells -hierarchical -filter "is_hard_macro==true"]
if {[sizeof_collection $MACROS] > 0} {
    foreach_in_collection macro $MACROS {
        create_keepout_margin -type hard -outer {30 30 30 30} $macro
    }
}

remove_individual_pin_constraints -ports [get_ports *]

set_individual_pin_constraints -ports [get_ports {axi_aw* axi_w* axi_b*}] -side bottom -allowed_layers {METAL5 METAL6} -pin_spacing 6.0
set_individual_pin_constraints -ports [get_ports {axi_ar* axi_r*}] -side right -allowed_layers {METAL5 METAL6} -pin_spacing 6.0
set_individual_pin_constraints -ports [get_ports {mem_*}] -side left -allowed_layers {METAL5 METAL6} -pin_spacing 6.0
set_individual_pin_constraints -ports [get_ports {led_* irq_out}] -side top -allowed_layers {METAL5 METAL6} -pin_spacing 6.0

if {[get_ports clk -quiet] != ""} {
    set_individual_pin_constraints -ports [get_ports clk] -side left -offset_from_die_edge 2500.0 -allowed_layers {METAL6}
}

if {[get_ports rst_n -quiet] != ""} {
    set_individual_pin_constraints -ports [get_ports rst_n] -side left -offset_from_die_edge 2550.0 -allowed_layers {METAL6}
}

place_pins -effort high

set_app_options -name place.coarse.congestion_driven -value true
create_placement -timing_driven -effort high

route_zrt_global -effort high

report_congestion -gr > ${ICC_OUTPUT_DIR}/congestion_global.rpt

update_timing -full
report_timing -path_type full_clock_expanded -max_paths 20 -delay_type max > ${ICC_OUTPUT_DIR}/timing_setup.rpt
report_timing -path_type full_clock_expanded -max_paths 20 -delay_type min > ${ICC_OUTPUT_DIR}/timing_hold.rpt
report_constraints -all_violators > ${ICC_OUTPUT_DIR}/constraints_violators.rpt

verify_pg_nets
check_pg_connectivity
check_pg_missing_vias

report_design > ${ICC_OUTPUT_DIR}/design.rpt
report_utilization -hierarchy > ${ICC_OUTPUT_DIR}/utilization.rpt
report_qor > ${ICC_OUTPUT_DIR}/qor.rpt
report_power -hierarchy > ${ICC_OUTPUT_DIR}/power.rpt

save_block -as ${DESIGN_NAME}_floorplan

write_def -version 5.8 -rows -macros -pins -blockages -nets -specialnets ${ICC_OUTPUT_DIR}/${DESIGN_NAME}_floorplan.def
write_verilog ${ICC_OUTPUT_DIR}/${DESIGN_NAME}_floorplan.v
write_floorplan -create_terminal -placement {hard final} ${ICC_OUTPUT_DIR}/${DESIGN_NAME}_floorplan.tcl

report_utilization
