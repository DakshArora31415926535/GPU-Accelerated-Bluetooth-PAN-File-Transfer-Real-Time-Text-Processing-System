# **HYPERLINK-GPU: CUDA-Accelerated Bluetooth PAN Real-Time Text Processing System**
---

## **Overview**
HYPERLINK-GPU integrates high-performance CUDA text processing into a Bluetooth PAN TCP communication pipeline. An Android device transmits text files offline to a Windows host, where the payload is offloaded to the GPU for parallel normalization, ASCII profiling, and rolling integrity hashing before being automatically typed into any application. The hybrid CPU-GPU design maintains responsiveness under large payloads and demonstrates real applied parallel computing beyond isolated kernel examples.

---

## **Key Features**
- Offline Bluetooth PAN TCP communication (no Wi-Fi/internet)
- CUDA parallel text processing (grid-stride loops, shared memory, atomics)
- Rolling 64-bit fingerprint hashing for content integrity
- Real-time output via Win32 SendInput automated typing
- Pinned memory + CUDA event timing for profiling
- Completely asynchronous CPU/GPU pipeline

---

## **GPU Pipeline**
| Stage | Method | Parallelism |
|-------|--------|-------------|
| Normalization + Uppercase | Per-byte parallel transform | Grid-stride |
| Histogram Analytics | Shared reduction | AtomicAdd |
| Rolling Hash | XOR fingerprint | AtomicXor |

**Example log (results/gpu_log.csv):**
operation,input_bytes,gpu_ms
full_pipeline,65536,0.441
full_pipeline,131072,0.829


---

## **System Architecture**

Android → TCP over Bluetooth PAN → Windows Receiver
→ CUDA GPU Text Pipeline → Win32 Automated Typing → Target App


---

## **Repository Structure**

/src
receiver_win32_fixed.cpp
cn_project_sender.py
gpu_text_pipeline.cu
/results
gpu_log.csv
processed_output.txt
/docs
project_report.pdf
/demo
demo_link.txt


---

## **Build & Run**
```bash
nvcc src/gpu_text_pipeline.cu -c -o gpu_text_pipeline.obj
g++ -std=c++17 src/receiver_win32_fixed.cpp gpu_text_pipeline.obj -o receiver.exe -lws2_32 -lcuda -lcudart
receiver.exe --port 5001 --gpu 1
python3 src/cn_project_sender.py
# Trigger auto-typing:
# Press 7 + 8 + 9



## **Future Enhancements**


GPU AES-256 encryption & SHA-256 hashing

CUDA stream-based parallel batching

GPU NLP semantic scoring

Author

Daksh Arora – IIITD – B.Tech CSE

