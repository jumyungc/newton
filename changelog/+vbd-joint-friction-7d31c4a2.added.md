Add Coulomb friction for revolute, prismatic, and D6 joints in VBD, including joints in mimic relationships.

Use a projected compliant-ALM multiplier for exact static sticking and saturated sliding, with an inverse-Delassus rho policy and a positive secant solve metric. Legacy AVBD retains its smooth near-rest law. Joint-coordinate gradients handle rotating frames and multi-axis joints. VBD's mimic solve uses assembled body Hessians and retained constraint reactions so follower friction contributes to force balance.
