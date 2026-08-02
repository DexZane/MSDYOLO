# Data workspace

This directory is reserved for local data artifacts and is not a source-data
distribution. Keep large or immutable files out of Git:

- `raw/`: original, unmodified downloads;
- `processed/`: derived or split data;
- `dataset/DOTA/`: the cloud workflow's current DOTA v1.5 location.

The DOTA preparation script writes pixel-coordinate polygon labels into
`dataset/DOTA/split/{train,val}/labelTxt`. Dataset downloads and generated
patches are ignored by Git and must be recreated with `bash scripts/setup.sh
--prepare-only`.
