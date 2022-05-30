# set your TOPLEVEL here
set TOPLEVEL "CVAE_top"

# change your timing constraint here
set TEST_CYCLE 5.12

source -echo -verbose 0_readfile.tcl 
source -echo -verbose 1_setting.tcl 
source -echo -verbose 2_compile.tcl 
source -echo -verbose 3_report.tcl 

exit
