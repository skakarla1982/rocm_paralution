Paralution-Open-Source
The Open Source version of Paralution: A library that enables you to perform various sparse iterative solvers and preconditioners on multi/many-core CPU and GPU devices.

Paralution build Steps for rocm
1) Clone the repo
        git clone https://github.com/srinivaskakarlaamd/rocm_paralution.git
2) Create build  in side rocm_paralution directory
        cd  rocm_paralution
        mkdir build
3) Build and  Compilation of paralution with HIP on ROCm
        cmake -DSUPPORT_HIP=ON ..
        make -j
4) Testing
       Go to build/bin
        Download the input file from the below link
              wget https://www.paralution.com/downloads/gr_30_30.mtx
              (https://www.paralution.com/downloads/)
Example
./binary < input file> < no of threads>
./amg   gr_30_30.mtx  1

