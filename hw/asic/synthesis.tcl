
set DESIGN_NAME "top_level"
set CLK_PERIOD 10.0
set CLK_PORT "clk"
set RST_PORT "rst_n"
set RTL_DIR "../fpga"
set TECH_LIB_DIR "/usr/local/synopsys/pdk/tsmc28/std_cells/libs"
set OUTPUT_DIR "./output"

file mkdir $OUTPUT_DIR

set_app_var sh_enable_page_mode false
suppress_message LINT-1
suppress_message LINT-28
suppress_message LINT-29

set target_library [list ${TECH_LIB_DIR}/slow.db ${TECH_LIB_DIR}/typical.db ${TECH_LIB_DIR}/fast.db]
set link_library [list ${TECH_LIB_DIR}/slow.db ${TECH_LIB_DIR}/typical.db ${TECH_LIB_DIR}/fast.db ${TECH_LIB_DIR}/memory_compiler.db]
set symbol_library [list ${TECH_LIB_DIR}/symbols.sdb]
set mw_reference_library ${TECH_LIB_DIR}/mw_lib
set mw_design_library ${OUTPUT_DIR}/mw_${DESIGN_NAME}

read_verilog -rtl [list ${RTL_DIR}/MemoryArbiter.topEntity.v ${RTL_DIR}/SSISearch.topEntity.v ${RTL_DIR}/RankerCore.topEntity.v ${RTL_DIR}/top_level.v]

current_design $DESIGN_NAME
link
elaborate $DESIGN_NAME
current_design $DESIGN_NAME
link

check_design > ${OUTPUT_DIR}/check_design.rpt

create_clock -name $CLK_PORT -period $CLK_PERIOD [get_ports $CLK_PORT]
set_clock_uncertainty -setup 0.2 [get_clocks $CLK_PORT]
set_clock_uncertainty -hold 0.1 [get_clocks $CLK_PORT]
set_clock_transition 0.1 [get_clocks $CLK_PORT]
set_clock_latency -source 0.5 [get_clocks $CLK_PORT]
set_clock_latency 0.3 [get_clocks $CLK_PORT]

set_input_delay -clock $CLK_PORT -max 2.0 [remove_from_collection [all_inputs] [get_ports $CLK_PORT]]
set_input_delay -clock $CLK_PORT -min 0.5 [remove_from_collection [all_inputs] [get_ports $CLK_PORT]]
set_output_delay -clock $CLK_PORT -max 2.0 [all_outputs]
set_output_delay -clock $CLK_PORT -min 0.5 [all_outputs]

set_input_delay 0 -clock $CLK_PORT [get_ports $RST_PORT]
set_false_path -from [get_ports $RST_PORT]

set_output_delay 0 -clock $CLK_PORT [get_ports led_*]
set_output_delay 0 -clock $CLK_PORT [get_ports irq_out]
set_false_path -to [get_ports led_*]
set_false_path -to [get_ports irq_out]

set_multicycle_path -setup 4 -from [get_pins MemoryArbiter_topEntity/*] -to [get_pins mem_*]
set_multicycle_path -hold 3 -from [get_pins MemoryArbiter_topEntity/*] -to [get_pins mem_*]

set_multicycle_path -setup 32 -from [get_pins SSISearch_topEntity/*] -to [get_pins mem_*]
set_multicycle_path -hold 31 -from [get_pins SSISearch_topEntity/*] -to [get_pins mem_*]

set_operating_conditions -max slow -max_library slow -min fast -min_library fast
set_wire_load_model -name "estimated" -library typical
set_wire_load_mode top

set_driving_cell -lib_cell BUFX4 -library typical [remove_from_collection [all_inputs] [list [get_ports $CLK_PORT] [get_ports $RST_PORT]]]
set_load 0.05 [all_outputs]

set_max_transition 0.5 $DESIGN_NAME
set_max_fanout 16 $DESIGN_NAME
set_max_capacitance 0.5 [all_outputs]
set_max_area 0

compile_ultra -gate_clock -no_autoungroup

set_clock_gating_style -sequential_cell latch -minimum_bitwidth 4 -control_point before
compile_ultra -gate_clock -incremental

set_dynamic_optimization true
compile_ultra -incremental -only_design_rule

report_timing -max_paths 10 -transition_time -nets -attributes > ${OUTPUT_DIR}/timing.rpt
report_area -hierarchy > ${OUTPUT_DIR}/area.rpt
report_power -hierarchy > ${OUTPUT_DIR}/power.rpt
report_constraint -all_violators > ${OUTPUT_DIR}/constraints.rpt
report_qor > ${OUTPUT_DIR}/qor.rpt
report_resources > ${OUTPUT_DIR}/resources.rpt
report_clock_gating -gated -ungated > ${OUTPUT_DIR}/clock_gating.rpt

check_design > ${OUTPUT_DIR}/check_design_final.rpt

change_names -rules verilog -hierarchy
write -format verilog -hierarchy -output ${OUTPUT_DIR}/${DESIGN_NAME}_synth.v
write -format ddc -hierarchy -output ${OUTPUT_DIR}/${DESIGN_NAME}.ddc
write_sdc ${OUTPUT_DIR}/${DESIGN_NAME}.sdc
write_sdf ${OUTPUT_DIR}/${DESIGN_NAME}.sdf
write_script > ${OUTPUT_DIR}/${DESIGN_NAME}_constraints.tcl

report_qor

exit
