# MuJoCo => BioMod Conversion

## Differences
### Bodies Become Segments
- MuJoCo: `body` elements can be nested, creating a hierarchy of bodies.
- BioMod: `segment` elements are flat and cannot be nested. Instead, the hierarchy is defined through the `parent` attribute of a `segment`.

### Geoms Become Segments
- MuJoCo: A `body` can contain multiple `geom` elements, each with an own mesh and mass. Only the body has a COM and inertia property (via the inertial tag).
- BioMod: `segment` elements can only have one mesh and mass. If a `body` has multiple `geom` elements, multiple `segments` have to be created and connected. Therefore, a static transform of `rt 0 0 0 xyz 0 0 0` is used. If the mesh in MuJoCo is translated (via the pos or euler or quat attrbute), the `meshrt <roll> <pitch> <yaw> xyz <transX> <transY> <transZ>` attribute is used. The inertia tensor and COM are only entered to the one segment.

### Independent Joints
- MuJoCo: Joints are defined explicitly (via `joint` element) within a `body` and connect it to its parent body. Each joint has its own type and properties.
- BioMod: Joints are defined in `segments` via the `rotations <axis>` or `translations <axis>` attribute.

### Sites Become Markers
- MuJoCo: `site` elements are used to define points of interest on a body, such as attachment points for sensors or actuators.
- BioMod: `marker` elements serve a similar purpose, defining points of interest on a segment.

### Tendons
- MuJoCo: `tendon` elements define tendons that can connect multiple sites.
- BioMod: The first and last of the MuJoCo tendon are used to define the `originPosition` and `insertionPosition` of a `tendon`. Intermediate sites are defined as `tendonRoutingPoint`.
