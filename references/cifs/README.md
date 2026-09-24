# Fe candidate structures

The current per-pattern workflow uses byte-for-byte copies of the user's files
from `C:/Users/jayliu/Documents/Data`. Lattice parameters are taken directly
from these files without adjustment. Both structures expand to ordered elemental
Fe; neutral `Fe0+` in the FCC file is normalized before simulation.

| File | Space group | Cubic a (Å) | Expanded sites | SHA-256 |
| --- | --- | --- | --- | --- |
| Fe-BCC.cif | 229, Im-3m | 2.86303550 | 2 | `10c51e7c30f582e3712fa70332663493e7608c43af86ceace35e579339147cda` |
| Fe-FCC.cif | 225, Fm-3m | 3.65555117 | 4 | `5fd92bd8e8859fd72bdc1289fd057d0ff29b86db474752c1730e1a649755b736` |

The original Ti candidate choice was incorrect for the current three MIB scans.
The three historical Ti CIFs have been removed from the repository; their
contents remain available in Git history. Tests generate their own wrong-element
fixture. The former Ti matching examples are disabled, and no current example
configuration loads those CIFs. User-owned source files and previous outputs
are preserved.
