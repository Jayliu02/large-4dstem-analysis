# Fe candidate structures and historical Ti references

The current per-pattern workflow uses byte-for-byte copies of the user's files
from `C:/Users/jayliu/Documents/Data`. Lattice parameters are taken directly
from these files without adjustment. Both structures expand to ordered elemental
Fe; neutral `Fe0+` in the FCC file is normalized before simulation.

| File | Space group | Cubic a (Å) | Expanded sites | SHA-256 |
| --- | --- | --- | --- | --- |
| Fe-BCC.cif | 229, Im-3m | 2.86303550 | 2 | `10c51e7c30f582e3712fa70332663493e7608c43af86ceace35e579339147cda` |
| Fe-FCC.cif | 225, Fm-3m | 3.65555117 | 4 | `5fd92bd8e8859fd72bdc1289fd057d0ff29b86db474752c1730e1a649755b736` |

The original Ti candidate choice was incorrect for the current three MIB scans.
The Ti files below are retained for historical traceability and legacy examples.

## Historical Ti references

These files are reference candidates, not evidence that a phase is present in
the experimental scans. Runtime structure audits contain SHA-256 fingerprints,
expanded atomic sites, symmetry and cell parameters.

| File | Source | Interpretation |
| --- | --- | --- |
| Ti-alpha-COD9008517.cif | https://www.crystallography.net/cod/9008517.cif | Alpha Ti, hcp, space group 194; Wyckoff (1963), Crystal Structures 1, 7–83; a=2.95 Å, c=4.686 Å. COD reference data, CC0: https://www.crystallography.net/cod/9008517.html |
| Ti-beta.cif | Byte-for-byte copy of user-provided data/Ti-bcc.cif | P1-expanded Ti structure; expected recovered symmetry Im-3m, 229. |
| Ti-omega.cif | Byte-for-byte copy of user-provided data/Ti-hcp.cif | Despite its original filename, three Ti sites and its cell describe omega Ti; expected recovered symmetry P6/mmm, 191. |

The original user files are not modified. The omega structure assignment is
consistent with the atomic coordinates discussed in:
https://doi.org/10.1016/S0921-5093(99)00365-2
