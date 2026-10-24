# gpu_chemistry


## Performance

**Note!** When running the gpuChemistryModel using multiple mpi-processes, make sure to enable nvidia-cuda-mps to ensure that the cpu processes use gpu resources concurrently. The scheduler can be launched without root use privileges by:
```
nvidia-cuda-mps-control -f > my_mps_output.log&
```

Some performance results are presented below. The results are from ```../tutorials/scalingTest``` which solves a simple shear layer combustion case in a cubic domain. The cell count in the test case is N=100^3 for the HPC node test and N=30^3 in the desktop PC case. As can be seen, the GPU implementation is faster, especially, when the number of cells per mpi process is high.

##### On a HPC node: 2 x AMD Rome 7H12 CPU (128 CPU cores) with 4 x A100 GPU
<img src="../tutorials/scalingTest/gri_results_a100.png" alt="asd" width="370"/>
<img src="../tutorials/scalingTest/h2_results_a100.png" alt="asd" width="370"/>

##### On Desktop PC: With AMD Ryzen Threadripper 3960X (24 CPU cores) with a RTX 3080Ti GPU
<img src="../tutorials/scalingTest/gri_results_rtx3080ti.png" alt="asd" width="370"/>
<img src="../tutorials/scalingTest/h2_results_rtx3080ti.png" alt="asd" width="370"/>

## Supported features
Not everything from the official OpenFOAM release is supported. If you need more support for features, please make a feature request.

#### Supported thermo models
Currently, the following thermodynamics model combination is supported.

```
thermoType
{
    type            heRhoThermo / hePsiThermo;
    mixture         <SomeMultiComponentMixture>;
    transport       <SomeTransportType>;
    thermo          janaf;
    energy          sensibleEnthalpy / sensibleInternalEnergy;
    equationOfState perfectGas;
    specie          specie;
}
```
This solves the energy equation for sensible enthalpy or internal energy and the transport properties are computed based on any of the available transport models in stock OpenFOAM. Heat capacity is taken from Janaf polynomials and the ideal gas law is assumed for the density.


#### Supported ODE solvers
* Rosenbrock12
* Rosenbrock23
* Rosenbrock34

Only the Rosenbrock family of ODE solvers is supported at this point.


#### Supported reaction types
The following reaction rate types are supported, each as a reversible and an irreversible reaction, e.g. ```reversibleArrhenius``` and ```irreversibleArrhenius```:

* Arrhenius
* ThirdBodyArrhenius
* LandauTeller
* Janev
* PLog (up to 16 pressures)
* ArrheniusLindemannFallOff
* ArrheniusTroeFallOff
* ArrheniusSRIFallOff
* ArrheniusLindemannChemicallyActivated
* ArrheniusTroeChemicallyActivated
* ArrheniusSRIChemicallyActivated

The non-equilibrium reversible reactions and the surface, Michaelis-Menten, Langmuir-Hinshelwood and powerSeries rates are not supported and stop the run with an error.

#### Supported chemistry model controls
As the OpenFOAM-dev ```standard``` chemistry model, ```gpuChemistryModel``` supports:

* the optional ```cellZone``` entry of ```chemistryProperties```, which limits the integration of the chemistry to the cells of the zone
* local time stepping (LTS), i.e. the ```localEuler``` ddt scheme
* the ```Treact``` temperature below which the chemistry is not solved
* the ```pivot``` and ```rowTol``` controls of the LU decomposition in the ```ode``` dictionary

The pressure is constant during the integration of the chemistry in each cell.


## Unit tests
The unit tests compare the gpuFoam models against the OpenFOAM reference implementations. They are compiled by ```Allwmake``` into ```Test-gpuChemistry``` which runs the tests on the CPU ```[CPU]``` and on the GPU ```[GPU]```:
```
Test-gpuChemistry          # all tests
Test-gpuChemistry "[CPU]"  # CPU tests only
```

#### Running the GPU kernel tests on the CPU
The script ```unittest/hostEmulation/Allrun``` builds and runs the unit tests, including the ```[GPU]``` kernel tests, on the CPU without a GPU or a GPU compiler. The device sources (```.cu```) are compiled as C++ with the host compiler, using the headers in ```unittest/hostEmulation/include``` in place of the GPU runtime headers of ```gpu_utils/common```: the kernels are run serially and the device memory is host memory.
```
cd unittest/hostEmulation
./Allrun                    # build into ./build and run all tests
./Allrun -norun             # build only
./Allrun -- "[GPU]"         # run the GPU kernel tests only
./Allclean
```
This requires a sourced OpenFOAM-dev environment and a C++17 host compiler. It checks the device code paths against the OpenFOAM reference kernels on the host, but it does not replace compiling with nvcc or hipcc and running the tests on a GPU.
