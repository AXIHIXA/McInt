if [ -d "build" ]; then
    rm -r build
fi

mkdir build
cmake -DCMAKE_BUILD_TYPE=Release -DPython_ROOT_DIR="${HOME}/opt/anaconda3/envs/py3" -DCUDA_ARCHS="70;75;80" -B build
cmake --build build -- -j16
