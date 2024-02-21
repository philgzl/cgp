# cgp

A Python implementation of the correlation based glimpse proportion (CGP) index proposed in [1] for speech intelligibility prediction.

Supports batched inputs.

# Installation

```
pip install python-cgp
```

# Usage

```python
import cgp
import soundfile as sf

x, fs = sf.read("path/to/input/signal")
y, fs = sf.read("path/to/reference/signal")

z = cgp.cgp(x, y, fs)
```

For batched inputs:
```python
import cgp
import numpy as np
import soundfile as sf

x_1, fs = sf.read("path/to/input/signal/1")
y_1, fs = sf.read("path/to/reference/signal/1")

x_2, fs = sf.read("path/to/input/signal/2")
y_2, fs = sf.read("path/to/reference/signal/2")

# x_1 and x_2 might have different lengths so we pad before creating the batch
lengths = [len(x_1), len(x_2)]
n = max(lengths)
x_1 = np.pad(x_1, (0, n - len(x_1)))
y_1 = np.pad(y_1, (0, n - len(y_1)))
x_2 = np.pad(x_2, (0, n - len(x_2)))
y_2 = np.pad(y_2, (0, n - len(y_2)))

# create the batch
x = np.stack([x_1, x_2])
y = np.stack([y_1, y_2])

# provide the lengths argument to ensure the same result as for unbatched processing
z = cgp.cgp(x, y, fs, axis=1, lengths=lengths)
```

# Note

This implementation corrects an indexing error in the original MATLAB code in `removeSilentFrames.m` which causes the last frame in the voice activity detection stage to be discarded.
This means the results can differ slightly from the original implementation.
To obtain the same results as the original implementation, set the `_discard_last_frame` argument to `True`.

# Speed comparison with `pystoi`

The paper claims CGP takes substantially less time to execute compared to baselines such as ESTOI [2].
Below is a comparison with `pystoi` for unbatched inputs.

TODO

# References

[1] A. Alghamdi, L. Moen, W.-Y. Chan, D. Fogerty and J. Jensen, "Correlation based glimpse proportion index", in Proc. WASPAA, 2023.\
[2] J. Jensen and C. H. Taal, "An algorithm for predicting the intelligibility of speech masked by modulated noise maskers", in IEEE/ACM Trans. Audio, Speech, Lang. Process., 2016.
