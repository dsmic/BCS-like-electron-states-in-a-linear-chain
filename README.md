# BCS-like-electron-states-in-a-linear-chain
Quantum mechanical multi particle states similar to BCS

The [documentation](documentation/BCSwithLinearChain.pdf) is found in the folder.

You can run a sample calculation by
```
python BCSstate.py FullRun_fast.json
```

which will produces simelar results as in the results folder.

If you do have a lot of computational power, you could use FullRun.json :)


BCS_3D_Al.py is a prove of concept to run a 3D material (Al) with realistic pseudo potentials in a 5x5x1 supercell calculating a gap. Due to the small supercell the gap is probably overestimated.

Run it by:
```
python BCS_3D_Al.py
```
