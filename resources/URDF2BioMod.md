# URDF => BioMod Conversion

## Known Issues
- Negative rotation axes are not supported. Workaround: Invert joint limits/ranges.
- Mesh scaling it something inappropriate, since ROS interprets meshes at 1000 scale when mesh scale is 1 in URDF. Workaround: Dívide all meshscales by 1000 in bioMod.
- Only visuals are converted, not collisions. The converter expects a visual for each collision.
