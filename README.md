# nanoconfinement-md

What does this code do
* The code enables simulations of ions confined between nanoparticles (NPs) or other material surfaces
    * Length of confinement is of the order of nanometers
* Materials represent nanoparticles (NPs) or biomacromolecules
    * NP surfaces are treated as planar walls due to the large size difference between ions and NPs 
* Users can extract the ionic structure (density profile) for a wide variety of ionic and environmental parameters
* Unpolarized surfaces are assumed and standard molecular dynamics is used to propagrate the dynamics of ions

Necessary Modules

* Load the necessary modules: module load gsl && module load openmpi/3.0.1 && module load boost/1_67_0
* Make sure to export BOOST_LIBDIR environment variable with location to the lib directory: export BOOST_LIBDIR=/opt/boost/gnu/openmpi_ib/lib/
* Also make sure to export OMP_NUM_THREADS environment variable with maximum threads available in your CPU: export OMP_NUM_THREADS=16

Install instructions

* Copy or git clone nanoconfinement-md project in to a directory.
* Go to nanoconfinement-md directory and (cd nanoconfinement-md)
* You should provide the following make command to make the project. This will create the executable and Install the executable (md_simulation_confined_ions) into bin directory (That is nanoconfinement-md/bin)
   * make local-install
* Next, go to the bin directory: cd bin
* Now you are ready to run the executable with aprun command using the following method: time mpirun -np 2 -N 16 ./md_simulation_confined_ions -Z 3 -p 1 -n -1 -c 0.5 -d 0.714 -S 1000000
* All outputs from the simulation will be stored in the bin folder when the simulation is completed.
   * Check and compare files (ex: energy.out) inside the bin/outfiles directory.
   * If you want to clean everything and create a new build, use: ```make clean``
   * Once the simulation has finished, data and outflies folders will contain the simulation results. You may check final density profile form data folder against the example desity profile provided in nanoconfinement-md/examples folder.

* ADD LAMMPS INPUT

For further details please refer to the [documentation](https://softmaterialslab.github.io/nanoconfinement-md/) 

## NanoHUB app page:
* https://nanohub.org/tools/nanoconfinement
