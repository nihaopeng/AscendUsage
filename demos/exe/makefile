INSTALL_DIR=/usr/local/Ascend/ascend-toolkit/latest
RT_INC=${INSTALL_DIR}/include
RT_LIB=${INSTALL_DIR}/lib64
# -Wl,-rpath,'$$ORIGIN' 在编译阶段就把“搜索路径”写进二进制文件中,ORIGIN代表当前路径
LDFLAGS += -L. -ldevice -Wl,-rpath,'$$ORIGIN'

single:singlefile.asc
	rm -f test
	bisheng -O2 --npu-arch=dav-2201 -I${RT_INC} -L${RT_LIB} -lascendcl -lruntime singlefile.asc -o test
	./test

device:multi/device.asc
	rm -f multi/libdevice.so
	rm -f libdevice.so
	bisheng -O2 -fPIC --npu-arch=dav-2201 --cce-auto-sync --shared multi/device.asc -o multi/libdevice.so
	mv multi/libdevice.so libdevice.so

multi:device
	rm -f test
	bisheng multi/host.asc --npu-arch=dav-2201 --cce-auto-sync -I${RT_INC} -L${RT_LIB} -L. -lruntime -lascendcl -lstdc++ -ldevice -o test ${LDFLAGS}
	./test