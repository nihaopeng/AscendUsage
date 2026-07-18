source /usr/local/Ascend/ascend-toolkit/set_env.sh

rm -rf build && mkdir -p build && cd build

cmake ..

make -j

./demo 4 64