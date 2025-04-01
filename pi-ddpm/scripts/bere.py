import cupy 
print("✅ GPU:", cupy.cuda.runtime.getDeviceProperties(0)['name'])