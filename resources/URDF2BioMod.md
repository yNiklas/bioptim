# URDF => BioMod Conversion

## Known Issues
- Negative rotation axes are not supported. Workaround: Invert joint limits/ranges, e.g. 0.15->0.8 => -0.8->0.15.
- Mesh scaling it something inappropriate, since ROS interprets meshes at 1000 scale when mesh scale is 1 in URDF. Workaround: Dívide all meshscales by 1000 in bioMod.
- Only visuals are converted, not collisions. The converter expects a visual for each collision.
- The rotation of `meshrt` in bioMod is interpreted as local (xyz) axes, while `rpy` in URDF is global. This is wrongly converted currently and affects visual meshes.
