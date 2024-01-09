# Predicting-Saturation-Vapour-Pressure
This is the project me and my teammates done in the course of University of Helsinki. 

## Overview
The term project is on the GeckoQ dataset, which has the atomic structures of 31,637 atmospherically relevant molecules resulting from pinene, toluene, and decane oxidation. The training set used in this competition is a subset of 27,147 molecules from the GeckoQ dataset. The test set in this competition is previously unpublished data, fresh from the LUMI supercomputer.

The GeckoQ dataset is built to complement data-driven research in atmospheric science. It provides molecular data pertinent to aerosol particle growth and new particle formation (NPF). A critical molecular property related to aerosol particle growth is the saturation vapour pressure (pSat), a measure of a molecule's ability to condense to the liquid phase. Molecules with low pSat and low-volatile organic compounds (LVOCs) are particularly interesting for NPF research; low saturation vapour pressure means a molecule is "sticky" and condenses easily. All the data in GeckoQ pertains to LVOCs. For more information, read:

```Besel Besel V, Todorović M, Kurtén T, Rinke P, Vehkamäki H. Atomic structures, conformers and thermodynamic properties of 32k atmospheric molecules. Sci Data. 2023 Jul 12;10(1):450 https://doi.org/10.1038/s41597-023-02366-x```

GeckoQ features important thermodynamic properties for each molecule, computed by time-consuming quantum chemistry simulations: saturation vapour pressures [Pa], the chemical potential [kJ/mol], the free energy of the molecule in the mixture [kJ/mol], and the heat of vaporisation [kJ/mol]. Of these four, the saturation vapour pressures will be the focus of your term project.

All data need for are placed in the file. (remember to put the data in the same root of the source code file if you are running directly)

Submission format should be like:
```Id,target
1000001,-4.52712625777568
1000002,-4.03796711059814
1000003,-5.51356645081844
1000004,-1.22768576052495
1000005,-2.68370396933252
etc.
```

Columns
**Id** - A unique molecule index used in naming files.

**MW** - The molecule's molecular weight (g/mol).

**pSat_Pa**- The pSat of the molecule calculated by COSMOtherm (Pa).

**NumOfAtoms** - The number of atoms in the molecule.

**NumOfC** - The number of carbon atoms in the molecule.

**NumOfO**- The number of oxygen atoms in the molecule.

**NumOfN**- The number of nitrogen atoms in the molecule.

**NumHBondDonors** - "The number of hydrogen bond donors in the molecule i.e. hydrogens bound to oxygen."

**parentspecies**- Either “decane”, “toluene”, “apin” for α-pinene, or a combination of these connected by an underscore to indicate ambiguous descent. In 243 cases, the parent species is “None” because retrieving it was impossible.

**NumOfConf**- The number of stable conformers found and successfully calculated by COSMOconf.

**NumOfConfUsed**- The number of conformers used to calculate the thermodynamic properties.

**C = C (non-aromatic)**- The number of non-aromatic C = C bounds found in the molecule.

**C = C-C = O in non-aromatic ring**- The number of C = C-C = O structures found in non-aromatic rings in the molecule.

**hydroxyl** (alkyl) - The number of the alkylic hydroxyl groups found in the molecule.

**aldehyde**- The number of aldehyde groups in the molecule.

**ketone** - The number of ketone groups in the molecule.

**carboxylic acid** - The number of carboxylic acid groups in the molecule.

**ester** - The number of ester groups in the molecule.

**ether** (alicyclic)- The number of alicyclic ester groups in the molecule.

**nitrate** - The number of alicyclic nitrate groups in the molecule.

**nitro** - The number of nitro ester groups in the molecule.

**aromatic hydroxyl** - The number of alicyclic aromatic hydroxyl groups in the molecule.

**carbonylperoxynitrate** - The number of carbonylperoxynitrate groups in the molecule.

**peroxide** - The number of peroxide groups in the molecule.

**hydroperoxide**- The number of hydroperoxide groups in the molecule.

**carbonylperoxyacid**- The number of carbonylperoxyacid groups found in the molecule

**nitroester**- The number of nitroester groups found in the molecule

**Target variable**
**pSat_Pa**- saturation vapour pressure. The target variable is log10(pSat_Pa).



More information can be found in: 
https://www.kaggle.com/competitions/iml_term_project_2023
