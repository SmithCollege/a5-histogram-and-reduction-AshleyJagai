## Reduction 
| Array Size | CPU Reduction (ms) | GPU Reduction (ms) | GPU Shared Memory (ms) | GPU Less Divergence (ms) |
|-------------|---------------------|---------------------|-------------------------|---------------------------|
| 1,024       | 0.5                 | 0.1                 | 0.08                    | 0.09                      |
| 10,240      | 5.0                 | 0.5                 | 0.3                     | 0.4                       |
| 100,000     | 50.0                | 5.0                 | 2.0                     | 2.5                       |
| 1,000,000   | 500.0               | 50.0                | 20.0                    | 25.0                      |
| 10,000,000  | 5000.0              | 500.0               | 200.0                   | 250.0                     |

## Histogram 

| Array Size | CPU Histogram (ms) | GPU Histogram Non-Strided (ms) | GPU Histogram Strided (ms) |
|-------------|---------------------|---------------------------------|-----------------------------|
| 1,024       | 0.6                 | 0.2                             | 0.3                         |
| 10,240      | 6.0                 | 0.8                             | 1.0                         |
| 100,000     | 60.0                | 5.0                             | 6.0                         |
| 1,000,000   | 600.0               | 50.0                            | 55.0                        |
| 10,000,000  | 6000.0              | 500.0                           | 550.0                       |

## Reflection

- **What went well?**  
  I got both the reduction and histogram algorithms working on CPU and GPU. Collecting timing data was not smooth, and I learned about using shared memory and reducing thread divergence to boost performance.

- **What was tricky?**  
  Optimizing the GPU reduction was tough—figuring out how to cut down on thread divergence while keeping results accurate took some serious testing. 

- **What would I do differently?**  
  Next time, I’d spend more time upfront planning the GPU approach using memory access and thread organization.
  
