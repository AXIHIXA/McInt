# PTE 

- PyTorch CUDA Extension Demo. 
- Implements a sample Monte-Carlo integration routine of: 
  - Green's function for 2D Laplacian operator; 
  - Over a complex polygonal domain (without self intersection, with holes); 
  - With respect to a huge number of center points; 
  - Differentiable with respect to domain boundary. 
- Fixes missing parts in libtorch's offical CMake config module for CUDA extensions 
  - in a brute-force manner. 
- **For testing purpose**, we have a much simpler example available at [HERE](https://github.com/AXIHIXA/Memo/tree/master/code/CMakeLists/PTE). **Check this first for a runnable demo**.
  - If the hyperref above does not work: Under `code/CMakeLists/PTE` directory of [AXIHIXA/Memo](https://github.com/AXIHIXA/Memo/). 
