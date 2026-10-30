Improve CUDA paired-backbone KKT solves with scalar-entry matrix reductions and
a uniform cooperative schedule, including eligible single-loop islands. Preserve
existing structural-island admission under the payload cap by allocating optional
single-loop work space only from remaining capacity. Local VBD equations, public
controls, CPU schedules and open-chain routes are unchanged.
