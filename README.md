SPHConvolve
===========

A small package used to convolve an SPH kernel with your data.

Requirements
------------

+ `python3`
+ `numpy`

For the tests,

+ `pytest`
+ `matplotlib`


Usage
-----

```python3
from sphconvolve import convolve_positions

convolved = convolve_positions(
	y=y_values,
	h=smoothing_length,
	N_neigh=48,
	dim=3
)
```

License
-------

MIT.
