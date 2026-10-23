# Publication notes

Date: 2026-10-24. Original code license: MIT. Owner: 1-e-1.

This is a method-focused source export of an existing local research project.
Shared numerical, geometric and runtime dependencies keep their original package
names, and are marked in EXPORT_MANIFEST.json; they are not claimed as separate
new contributions. Model equations and trained-parameter files are not modified.
Historical READMEs and reports may refer to paths or artifacts outside this export.
Data, checkpoints, recordings, CAD/USD meshes, textures, videos, caches, personal
work logs and external library trees are not redistributed. No historical research
training or physical robot execution was rerun for this publication. CPU checks
cover only the functions listed in PUBLICATION_CHECKS.md after verification.

NumPy compatibility: the trapezoidal integration calls use np.trapezoid when available and otherwise np.trapz. This preserves the integration rule while supporting both the original NumPy 1.26 environment and NumPy 2.4. No model equations or stored results were changed.
