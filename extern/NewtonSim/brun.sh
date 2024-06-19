cd build;
make -j4;
cd ..;
./build/dramsim3main configs/HBM2_8Gb_x128_pim.ini --stream random -c 1000000;

# ./build/dramsim3main configs/HBM2_8Gb_x128_pim.ini -c 100000 -t trace.txt
