Progress at N = 4096

| kernel | ms | TFLOPS | % cuBLAS | % peak | speedup vs previous | correct |
|---|---:|---:|---:|---:|---:|---|
| naive | 203.723 | 0.67 | 1.2% | 0.8% | - | True |
| coalesced | 24.683 | 5.57 | 9.9% | 6.7% | 8.25x | True |
| shared_mem | 21.770 | 6.31 | 11.3% | 7.6% | 1.13x | True |
