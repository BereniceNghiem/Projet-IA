import cupy
print("📍 CuPy path:", cupy.__file__)
print("🎯 GPU device:", cupy.cuda.runtime.getDeviceProperties(0)['name'])