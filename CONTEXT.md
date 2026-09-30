# Killing assay glossary

LiSCA product terms (Workspace, Position, Frame, Timepoint, Sample, Pattern, ROI,
Trace, Task, Step, …): `~/workspace/lisca/CONTEXT.md`. This repo uses them with the
same meaning and adds only the killing-specific terms below.

| Term | Meaning here |
| --- | --- |
| **ROI stack** | `roi/Pos{n}/Roi{m}.tif` in a LiSCA Workspace; brightfield and TOTO-3 pages interleaved, two pages per Frame |
| **Cell** | The biological cell on one ROI's Pattern (one label per ROI) |
| **Morphology death time** | Frame where the cell is judged dead from brightfield shape |
| **TOTO-3 death time** | Frame where TOTO-3 (membrane-permeability reporter) rises |
| **STS** | Staurosporine; killing agent for review Fig. 6 D–E |
| **death_frame** | Label in `labels.json`; a Frame index. Equal to the ROI's Frame count means the cell survived |
