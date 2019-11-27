# BE Processes
``pyUSID.Process`` extensions for BE SHO and Loop fitters in Pycroscopy

### Main points:
1. The scientific algorithms themselves remain unaltered
2. Extending ``pyUSID.Process`` allow these classes to extend beyond 1 node in a cluster via ``MPI``
3. The classes are also a **LOT** more modular and readable since they eschew the 
   ``pycroscopy.analysis.GuessMethods``, ``pycroscopy.analysis.FitMethods``, ``pycroscopy.analysis.Optimizer``
4. These classes will eventually replace their original counterparts in pycroscopy soon (late 2019 / early 2020) *upon verification / validation by CNMS staff*.

### Usage:
```python
import h5py
import pyUSID as usid

# Since the Fitters are not part of pycroscopy (yet), import them manually 
from be_sho_fitter import BESHOfitter
from be_loop_fitter import BELoopFitter

h5_path = '/path/to/your/be_measurement_data.h5'

h5_f = h5py.File(h5_path, mode='r+')
h5_main = usid.USIDataset(h5_f['Measurement_000/Channel_000/Raw_Data'])

sho_proc = BESHOfitter(h5_main, verbose=False)

# A set up needs to be performed before the actual computation (do_guess)
# This is where you set parameters, etc.
sho_proc.set_up_guess()
h5_sho_guess = sho_proc.do_guess()

# Set up fit before fitting
sho_proc.set_up_fit()
h5_sho_fit = sho_proc.do_fit()

# Repeat the same procedure for Loop fitting:
loop_proc = BELoopFitter(h5_sho_fit, verbose=False)

loop_proc.set_up_guess()
h5_loop_guess = loop_proc.do_guess()

loop_proc.set_up_fit()
h5_loop_fit = loop_proc.do_fit()

# Verify via visualization / analyze further ...
```