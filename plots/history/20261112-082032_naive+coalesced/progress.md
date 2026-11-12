Progress at N = 8192

| kernel | ms | TFLOPS | % cuBLAS | % peak | speedup vs previous | correct |
|---|---:|---:|---:|---:|---:|---|
| naive | 1611.641 | 0.68 | 1.2% | 0.8% | - | True |
| coalesced | 230.633 | 4.77 | 8.7% | 5.8% | 6.99x | True |
