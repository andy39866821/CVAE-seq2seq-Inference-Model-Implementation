set TOP_DIR $TOPLEVEL
set RPT_DIR report
set NET_DIR netlist

sh rm -rf ./$TOP_DIR
sh rm -rf ./$RPT_DIR
sh rm -rf ./$NET_DIR
sh mkdir ./$TOP_DIR
sh mkdir ./$RPT_DIR
sh mkdir ./$NET_DIR

# define a lib path here
define_design_lib $TOPLEVEL -path ./$TOPLEVEL

# Read Design File (add your files here)
set HDL_DIR "../../RTL/hdl"
analyze -library $TOPLEVEL -format verilog "$HDL_DIR/CVAE_top.v $HDL_DIR/FullyConnection.v $HDL_DIR/GRU.v $HDL_DIR/Cordic.v $HDL_DIR/VLC.v $HDL_DIR/P_RHC.v $HDL_DIR/NP_RHC.v "

# elaborate your design
elaborate $TOPLEVEL -architecture verilog -library $TOPLEVEL

# Solve Multiple Instance
set uniquify_naming_style "%s_mydesign_%d"
uniquify

# link the design
current_design $TOPLEVEL
link
