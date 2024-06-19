# NeuPIMs Simulator

This is NeuPIMs cycle-accurate simulator. We integrate open-source [NPU simulator](https://github.com/PSAL-POSTECH/ONNXim) and our in-house PIM simulator based on DRAMsim3 to implement NeuPIMs simulator.

### Publication

- Paper: https://dl.acm.org/doi/10.1145/3620666.3651380
- Authors: Guseul Heo, Sangyeop Lee, Jaehong Cho, Hyunmin Choi, and Sanghyeon Lee (KAIST); Hyungkyu Ham and Gwangsun Kim (POSTECH); Divya Mahajan (Georgia Tech); Jongse Park (KAIST)

---

### Python Package

- torch >= 1.10.1
- conan == 1.57.0
- onnxruntime >= 1.10.0

### Package

- cmake >= 3.22.1 (You need to build manually)
- gcc == 8.3

---

# Getting Started

## method 1 (Docker Image)

```
$ git clone https://github.com/casys-kaist/NeuPIMs.git
$ cd NeuPIMs
$ docker build . -t neupims
$ docker run -it -v .:/workspace/neupims-sim neupims
(docker) cd neupims-sim
(docker) git submodule update --recursive --init
(docker) ./build.sh
```

build docker image and installation

```
$ docker run -it -v .:/workspace/neupims-sim neupims
(docker) cd neupims-sim
(docker) ./brun.sh
```

run docker image

## method 2 (Mannual)

### Installation

```
$ git clone https://github.com/casys-kaist/NeuPIMs.git
$ cd NeuPIMs
$ git submodule update --recursive --init
```

### Build

```
$ mkdir build && cd build
$ conan install .. --build missing
$ cmake ..
$ make -j
```

### Run Simulator

```
$ cd ..
$ ./brun.sh
```

### Baselines

1. NPU-only: Codes on `npu-only` branch, all operations in LLM batched inference are executed on NPU.
2. NPU+PIM: Codes on `npu+pim` branch, attention GEMV operations on PIM. PIM is single row buffered PIM on this baseline.
3. NeuPIMs: Codes on `main` branch, we use dual row buffer PIM and sub-batch interleaving technique. (Sub-batch interleaving is enabled only when batch size>=256)

### Citation

If you use NeuPIMs for your research, please cite our paper:

```
@inproceedings{10.1145/3620666.3651380,
author = {Heo, Guseul and Lee, Sangyeop and Cho, Jaehong and Choi, Hyunmin and Lee, Sanghyeon and Ham, Hyungkyu and Kim, Gwangsun and Mahajan, Divya and Park, Jongse},
title = {NeuPIMs: NPU-PIM Heterogeneous Acceleration for Batched LLM Inferencing},
year = {2024},
doi = {10.1145/3620666.3651380},
booktitle = {Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3},
series = {ASPLOS '24}
}
```
