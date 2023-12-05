nvcc \
-I/home/xihan1/workspace/PTE/include/ \
-I/home/xihan1/opt/anaconda3/envs/py3/lib/python3.10/site-packages/torch/include \
-I/home/xihan1/opt/anaconda3/envs/py3/lib/python3.10/site-packages/torch/include/torch/csrc/api/include \
-I/home/xihan1/opt/anaconda3/envs/py3/lib/python3.10/site-packages/torch/include/TH \
-I/home/xihan1/opt/anaconda3/envs/py3/lib/python3.10/site-packages/torch/include/THC \
-I/usr/local/cuda/include \
-I/home/xihan1/opt/anaconda3/envs/py3/include/python3.10 \
-c src/main.cu \
-o main.o \
-D__CUDA_NO_HALF_OPERATORS__ \
-D__CUDA_NO_HALF_CONVERSIONS__ \
-D__CUDA_NO_BFLOAT16_CONVERSIONS__ \
-D__CUDA_NO_HALF2_OPERATORS__ \
--expt-relaxed-constexpr \
--compiler-options ''"'"'-fPIC'"'"'' \
-O2 \
-DTORCH_API_INCLUDE_EXTENSION_H \
'-DPYBIND11_COMPILER_TYPE="_gcc"' \
'-DPYBIND11_STDLIB="_libstdcpp"' \
'-DPYBIND11_BUILD_ABI="_cxxabi1011"' \
-DTORCH_EXTENSION_NAME=pte \
-D_GLIBCXX_USE_CXX11_ABI=0 \
-gencode=arch=compute_75,code=compute_75 \
-gencode=arch=compute_75,code=sm_75 \
-std=c++17

g++ \
-pthread \
-B /home/xihan1/opt/anaconda3/envs/py3/compiler_compat \
-shared \
-Wl,-rpath,/home/xihan1/opt/anaconda3/envs/py3/lib \
-Wl,-rpath-link,/home/xihan1/opt/anaconda3/envs/py3/lib \
-L/home/xihan1/opt/anaconda3/envs/py3/lib \
-Wl,-rpath,/home/xihan1/opt/anaconda3/envs/py3/lib \
-Wl,-rpath-link,/home/xihan1/opt/anaconda3/envs/py3/lib \
-L/home/xihan1/opt/anaconda3/envs/py3/lib \
main.o \
-L/home/xihan1/opt/anaconda3/envs/py3/lib/python3.10/site-packages/torch/lib \
-L/usr/local/cuda/lib64 \
-lc10 \
-ltorch \
-ltorch_cpu \
-ltorch_python \
-lcudart \
-lc10_cuda \
-ltorch_cuda \
-o pte.so
