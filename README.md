# sc-jnmf
![main_fig](/fig/fig.png)

## Requirement
- Ubuntu==18.04
- CUDA==10.0 (gpu required)
- Python>=3.6

(This tool is developed under the docker image [nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04](https://hub.docker.com/layers/nvidia/cuda/10.0-cudnn7-devel-ubuntu18.04/images/sha256-3ce355415152dfc1eb9145d3d8bbb4b901de16b2a005b068ed48e5e182375a1d?context=explore))

## Installing
` git clone https://github.com/agis09/sc-jnmf`  
` pip install ./sc-jnmf`

## Usage
```python3
from sc_jnmf import sc_JNMF
import numpy as np

d1 = np.random.randint(4, size=(10, 10))

d2 = np.random.randint(5, size=(10, 10))


sc_jnmf = sc_JNMF(d1, d2, rank=3)
sc_jnmf.factorize()


```

## Sample
Read the [notebook](/example/example.ipynb)  

or  

change current directory to "example/"  
`cd example/ `  

to use sc-jnmf  
`python example1.py`  

## Documentation
https://agis09.github.io/sc-jnmf/index.html#document-sc_jnmf
