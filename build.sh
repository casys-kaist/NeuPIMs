rm -rf build
mkdir build && cd build
conan install .. --build missing
cmake ..
make -j