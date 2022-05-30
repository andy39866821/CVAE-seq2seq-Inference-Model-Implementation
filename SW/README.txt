Golden:
	C++軟體模型資料夾，執行指令為:
		make
		make run
	程式會讀取Golden/data/內的浮點數Weights/Bias/Input，並計算出Outputs，最後將這些資料做quantization寫至quantized_data/內，供硬體模擬使用。
RISCV:
	在FPGA上包含RISCV CPU的系統內執行的baremetal application, 用於量測CPU的計算clock cycles作為效能比較。
	包含RISCV的assembly code，因此無法使用工作站的gcc編譯。
	須使用RISCV的riscv64-unknown-elf-gcc編譯器(工作站沒有該編譯器，我在自己環境編譯的並測試的)
	因為是使用baremetal application的方式執行，所有weights/bias/input都以c header方式預先編譯進程式。