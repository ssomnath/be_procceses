# BE Processes
``pyUSID.Process`` extensions for BE SHO and Loop fitters in Pycroscopy

Main points:
1. The scientific algorithms themselves remain unaltered
2. Extending ``pyUSID.Process`` allow these classes to extend beyond 1 node in a cluster via ``MPI``
3. The classes are also a **LOT** more modular and readable since they eschew the 
   ``pycroscopy.analysis.GuessMethods``, ``pycroscopy.analysis.FitMethods``, ``pycroscopy.analysis.Optimizer``
4. These classes will eventually replace their original counterparts in pycroscopy soon (late 2019 / early 2020)

