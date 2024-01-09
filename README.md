# Predicting-Saturation-Vapour-Pressure
This is the project me and my teammates done in the course of University of Helsinki. 

## Overview
The term project is on the GeckoQ dataset, which has the atomic structures of 31,637 atmospherically relevant molecules resulting from pinene, toluene, and decane oxidation. The training set used in this competition is a subset of 27,147 molecules from the GeckoQ dataset. The test set in this competition is previously unpublished data, fresh from the LUMI supercomputer.

The GeckoQ dataset is built to complement data-driven research in atmospheric science. It provides molecular data pertinent to aerosol particle growth and new particle formation (NPF). A critical molecular property related to aerosol particle growth is the saturation vapour pressure (pSat), a measure of a molecule's ability to condense to the liquid phase. Molecules with low pSat and low-volatile organic compounds (LVOCs) are particularly interesting for NPF research; low saturation vapour pressure means a molecule is "sticky" and condenses easily. All the data in GeckoQ pertains to LVOCs. For more information, read:

```{python}Besel V, Todorović M, Kurtén T, Rinke P, Vehkamäki H. Atomic structures, conformers and thermodynamic properties of 32k atmospheric molecules. Sci Data. 2023 Jul 12;10(1):450 https://doi.org/10.1038/s41597-023-02366-x

GeckoQ features important thermodynamic properties for each molecule, computed by time-consuming quantum chemistry simulations: saturation vapour pressures [Pa], the chemical potential [kJ/mol], the free energy of the molecule in the mixture [kJ/mol], and the heat of vaporisation [kJ/mol]. Of these four, the saturation vapour pressures will be the focus of your term project.
