# Newton Simulator

This simulator is based on DRAMsim3.

### DRAM parameter

- $t_{FAW}$: Four-bank Activation Window
- $t_{RRD}$: Row-to-Row Activation Delay

### Newton command

- `GWRITE`: Copy a specific DRAM row to the global buffer, From DRAM's point of view, it's similar to READ
- `G_ACT#`: Activate the rows of DRAM bank group(4 banks), related to tFAW, tRCDRD
- `COMP#` : Execute MAC operation in the adder tree. Rate-matched to retrieval rate
- `READRES`: Read accumulated results.

### NeuPIMs command

- TBD: refer to the paper

### Building

```bash
# cmake out of source build
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..

# Build dramsim3 library and executables
make -j4
```

### Runnning

```bash
# help
./build/dramsim3main -h

# Running random stream with a config file
./build/dramsim3main configs/DDR4_8Gb_x8_3200.ini --stream random -c 100000

# Running a trace file
./build/dramsim3main configs/DDR4_8Gb_x8_3200.ini -c 100000 -t sample_trace.txt

# Running with gem5
--mem-type=dramsim3 --dramsim3-ini=configs/DDR4_4Gb_x4_2133.ini

```
