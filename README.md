# WORK IN PROGRESS

https://github.com/kijai/ComfyUI-OpenDiTWrapper/assets/40791699/1cb3aa06-1576-43ad-a01e-f5a63f59127d


https://github.com/kijai/ComfyUI-OpenDiTWrapper/assets/40791699/422565ae-aea9-43a7-b9d5-a8a8e7963141



Tested to work in Linux, and on Windows with current ComfyUI portable (pytorch ):

pytorch version: 2.3.1+cu121

xformers version: 0.0.26.post1

VRAM requirements are pretty hefty, but far less than the original repo due to offloading.
For example 48 frames at 768x512 fit in 15GB, less frames/smaller res can fit 10GB.

## Installing:

`pip install -r requirements.txt`

`pip install xformers --no-deps`

for portable (in ComfyUI_windows_portable -folder):

`python_embeded\python.exe -m pip install -r .\ComfyUI\custom_nodes\ComfyUI-OpenDiTWrapper\requirements.txt`

`python_embeded\python.exe -m pip install xformers --no-deps`

Original repo:

https://github.com/NUS-HPC-AI-Lab/OpenDiT/
