#!/usr/bin/env python3
"""
URDF to bioMod Converter

Converts URDF (Unified Robot Description Format) files to bioMod format.
Key difference: URDF joints are independent, but bioMod joints are part of segments (via 'rotations' attribute).
"""

import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class Visual:
    """Represents a visual element in a URDF link"""
    mesh_file: Optional[str] = None
    mesh_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    mesh_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    mesh_scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass
class Link:
    """Represents a URDF link (becomes a bioMod segment)"""
    name: str
    mass: float = 0.0
    inertia: np.ndarray = field(default_factory=lambda: np.eye(3))
    com: Tuple[float, float, float] = (0, 0, 0)
    visuals: List[Visual] = field(default_factory=list)


@dataclass
class Joint:
    """Represents a URDF joint"""
    name: str
    type: str  # revolute, prismatic, fixed, etc.
    parent: str
    child: str
    origin_xyz: Tuple[float, float, float] = (0, 0, 0)
    origin_rpy: Tuple[float, float, float] = (0, 0, 0)
    axis: Tuple[float, float, float] = (1, 0, 0)
    limit_lower: float = -np.pi
    limit_upper: float = np.pi


class URDFParser:
    """Parses URDF XML files"""

    def __init__(self, urdf_file: str):
        self.urdf_file = Path(urdf_file)
        self.tree = ET.parse(urdf_file)
        self.root = self.tree.getroot()
        self.links: Dict[str, Link] = {}
        self.joints: Dict[str, Joint] = {}

    def parse(self):
        """Parse the URDF file"""
        self._parse_links()
        self._parse_joints()

    def _parse_links(self):
        """Extract all links from URDF"""
        for link_elem in self.root.findall('link'):
            name = link_elem.get('name')
            link = Link(name=name)

            # Parse inertial properties
            inertial = link_elem.find('inertial')
            if inertial is not None:
                mass_elem = inertial.find('mass')
                if mass_elem is not None:
                    link.mass = float(mass_elem.get('value', 0))

                origin = inertial.find('origin')
                if origin is not None:
                    link.com = self._parse_xyz(origin)

                inertia_elem = inertial.find('inertia')
                if inertia_elem is not None:
                    link.inertia = self._parse_inertia_matrix(inertia_elem)

            # Parse all visuals
            for visual_elem in link_elem.findall('visual'):
                geometry = visual_elem.find('geometry')
                if geometry is not None:
                    mesh = geometry.find('mesh')
                    if mesh is not None:
                        mesh_file = mesh.get('filename')
                        
                        origin = visual_elem.find('origin')
                        mesh_offset = (0.0, 0.0, 0.0)
                        mesh_rpy = (0.0, 0.0, 0.0)
                        if origin is not None:
                            mesh_offset = self._parse_xyz(origin)
                            mesh_rpy = self._parse_rpy(origin)

                        scale_str = mesh.get('scale', '1 1 1')
                        try:
                            mesh_scale = tuple(float(x) for x in scale_str.split())
                            if len(mesh_scale) != 3:
                                mesh_scale = (1.0, 1.0, 1.0)
                        except Exception:
                            mesh_scale = (1.0, 1.0, 1.0)

                        link.visuals.append(Visual(
                            mesh_file=mesh_file,
                            mesh_offset=mesh_offset,
                            mesh_rpy=mesh_rpy,
                            mesh_scale=mesh_scale
                        ))

            self.links[name] = link

    def _parse_joints(self):
        """Extract all joints from URDF"""
        for joint_elem in self.root.findall('joint'):
            name = joint_elem.get('name')
            joint_type = joint_elem.get('type')

            parent_elem = joint_elem.find('parent')
            child_elem = joint_elem.find('child')
            parent = parent_elem.get('link') if parent_elem is not None else None
            child = child_elem.get('link') if child_elem is not None else None

            joint = Joint(name=name, type=joint_type, parent=parent, child=child)

            # Parse origin (transformation from parent to child)
            origin = joint_elem.find('origin')
            if origin is not None:
                joint.origin_xyz = self._parse_xyz(origin)
                joint.origin_rpy = self._parse_rpy(origin)

            # Parse axis
            axis = joint_elem.find('axis')
            if axis is not None:
                joint.axis = self._parse_xyz(axis)

            # Parse limits
            limit = joint_elem.find('limit')
            if limit is not None:
                joint.limit_lower = float(limit.get('lower', -np.pi))
                joint.limit_upper = float(limit.get('upper', np.pi))

            self.joints[name] = joint

    @staticmethod
    def _parse_xyz(elem) -> Tuple[float, float, float]:
        """Parse xyz attribute"""
        xyz = elem.get('xyz', '0 0 0')
        return tuple(float(x) for x in xyz.split())

    @staticmethod
    def _parse_rpy(elem) -> Tuple[float, float, float]:
        """Parse rpy (roll, pitch, yaw) attribute"""
        rpy = elem.get('rpy', '0 0 0')
        return tuple(float(x) for x in rpy.split())

    @staticmethod
    def _parse_inertia_matrix(elem) -> np.ndarray:
        """Parse 3x3 inertia matrix from URDF"""
        ixx = float(elem.get('ixx', 0))
        iyy = float(elem.get('iyy', 0))
        izz = float(elem.get('izz', 0))
        ixy = float(elem.get('ixy', 0))
        ixz = float(elem.get('ixz', 0))
        iyz = float(elem.get('iyz', 0))

        return np.array([
            [ixx, ixy, ixz],
            [ixy, iyy, iyz],
            [ixz, iyz, izz]
        ])


class TransformationHelper:
    """Helper class for rotation and transformation matrix operations"""

    @staticmethod
    def rpy_to_matrix(rpy: Tuple[float, float, float]) -> np.ndarray:
        """Convert RPY (roll, pitch, yaw) to 3x3 rotation matrix"""
        roll, pitch, yaw = rpy

        # Rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])

        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        return Rz @ Ry @ Rx

    @staticmethod
    def matrix_to_xyz_rpy(rotation: np.ndarray) -> Tuple[float, float, float]:
        """Convert 3x3 rotation matrix back to RPY"""
        # Assuming ZYX order (yaw, pitch, roll)
        sy = np.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2)

        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(rotation[2, 1], rotation[2, 2])
            y = np.arctan2(-rotation[2, 0], sy)
            z = np.arctan2(rotation[1, 0], rotation[0, 0])
        else:
            x = np.arctan2(-rotation[1, 2], rotation[1, 1])
            y = np.arctan2(-rotation[2, 0], sy)
            z = 0

        return (x, y, z)

    @staticmethod
    def get_rotation_axis_and_angle(rpy: Tuple[float, float, float],
                                    axis_hint: Tuple[float, float, float]) -> str:
        """
        Determine primary rotation axis from RPY and joint axis.
        Returns rotation character: 'x', 'y', 'z', or combination like 'xy'
        """
        roll, pitch, yaw = rpy
        ax, ay, az = axis_hint

        # Normalize axis hint
        norm = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
        if norm > 0:
            ax, ay, az = ax / norm, ay / norm, az / norm

        # Determine dominant rotation axes based on joint axis
        axes = []
        if abs(ax) > 0.9:
            axes.append('x')
        if abs(ay) > 0.9:
            axes.append('y')
        if abs(az) > 0.9:
            axes.append('z')

        return ''.join(axes) if axes else 'x'


class BioModSegment:
    """Represents a bioMod segment"""

    def __init__(self, name: str, parent: Optional[str] = None):
        self.name = name
        self.parent = parent
        self.rotations: List[str] = []
        self.translations: List[str] = []
        self.ranges: List[Tuple[float, float]] = []
        self.mass: float = 0.0
        self.inertia: np.ndarray = np.eye(3)
        self.com: Tuple[float, float, float] = (0, 0, 0)
        self.mesh_file: Optional[str] = None
        self.mesh_scale: Tuple[float, float, float] = (1, 1, 1)
        self.mesh_rt: Tuple[float, float, float, Tuple[float, float, float]] = (0, 0, 0, (0, 0, 0))
        self.rt_in_matrix: int = 0
        self.rt_matrix: np.ndarray = np.eye(4)

    def format_for_biomod(self) -> str:
        """Format segment as bioMod text"""
        lines = [f"segment {self.name}"]

        if self.parent:
            lines.append(f"    parent {self.parent}")

        # Add rotations
        if self.rotations:
            lines.append(f"    rotations {' '.join(self.rotations)}")

        # Add translations
        if self.translations:
            lines.append(f"    translations {' '.join(self.translations)}")

        # Add ranges
        if self.ranges:
            lines.append("    ranges")
            for lower, upper in self.ranges:
                lines.append(f"        {self._format_angle(lower)} {self._format_angle(upper)}")

        # Add RT matrix or individual rt
        if self.parent:  # Only child segments have transformations
            lines.append(f"    rtinmatrix {self.rt_in_matrix}")
            if self.rt_in_matrix:
                lines.append("    rt")
                for row in self.rt_matrix:
                    lines.append(f"        {row[0]:8.5f}    {row[1]:8.5f}    {row[2]:8.5f}    {row[3]:8.5f}")
            else:
                rpy = self.rt_matrix[:3, :3]
                xyz = tuple(self.rt_matrix[:3, 3])
                lines.append(
                    f"    rt {rpy[0, 0]:.1f} {rpy[0, 1]:.1f} {rpy[0, 2]:.1f} xyz {xyz[0]:.6g} {xyz[1]:.6g} {xyz[2]:.6g}")

        # Add mass and inertia
        if self.mass > 0 or self.parent:
            lines.append(f"    mass")
            lines.append(f"        {self.mass:.6g}")

        if self.parent or self.mass > 0:
            lines.append("    inertia")
            for row in self.inertia:
                lines.append(f"        {row[0]:12.5g}     {row[1]:12.5g}     {row[2]:12.5g}")

        # Add COM
        if self.parent or self.mass > 0:
            lines.append(f"    com")
            lines.append(f"        {self.com[0]:12.5g}    {self.com[1]:12.5g}   {self.com[2]:12.5g}")

        # Add mesh
        if self.mesh_file:
            lines.append(f"    meshFile {self.mesh_file}")
            lines.append(
                f"    meshscale {self.mesh_scale[0]:.6g}    {self.mesh_scale[1]:.6g}    {self.mesh_scale[2]:.6g}")
            rpy_val = self.mesh_rt[0:3]
            xyz_val = self.mesh_rt[3]
            lines.append(
                f"    meshrt {rpy_val[0]:.6g} {rpy_val[1]:.6g} {rpy_val[2]:.6g} xyz {xyz_val[0]:.6g} {xyz_val[1]:.6g} {xyz_val[2]:.6g}")

        lines.append("endsegment")
        lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _format_angle(angle: float) -> str:
        """Format angle, handling pi notation"""
        if abs(angle) < 1e-6:
            return "0"
        if abs(abs(angle) - np.pi) < 1e-6:
            return "-pi" if angle < 0 else "pi"
        if abs(angle / np.pi - round(angle / np.pi)) < 1e-6:
            coeff = round(angle / np.pi)
            if coeff == 1:
                return "pi"
            elif coeff == -1:
                return "-pi"
            else:
                return f"{coeff}*pi"
        return f"{angle:.6g}"


class URDFToBioModConverter:
    """Main converter class"""

    def __init__(self, urdf_file: str):
        self.parser = URDFParser(urdf_file)
        self.parser.parse()
        self.segments: Dict[str, BioModSegment] = {}
        self.transform_helper = TransformationHelper()

    def convert(self) -> str:
        """Convert URDF to bioMod format"""
        # Find root link (link with no parent)
        root_link = self._find_root_link()

        # Build hierarchy
        self._build_segments(root_link)

        # Generate bioMod content
        return self._generate_biomod()

    def _find_root_link(self) -> str:
        """Find the root link (has no parent in any joint)"""
        child_links = {joint.child for joint in self.parser.joints.values()}
        for link_name in self.parser.links.keys():
            if link_name not in child_links:
                return link_name
        return next(iter(self.parser.links.keys()))

    def _build_segments(self, link_name: str, parent_segment: Optional[str] = None,
                        parent_joint: Optional[Joint] = None):
        """Recursively build segments from links and joints"""
        link = self.parser.links[link_name]
        segment = BioModSegment(link_name, parent=parent_segment)

        # Copy link properties to segment
        segment.mass = link.mass
        segment.inertia = link.inertia
        segment.com = link.com
        
        if link.visuals:
            first_visual = link.visuals[0]
            segment.mesh_file = first_visual.mesh_file
            segment.mesh_scale = first_visual.mesh_scale
            segment.mesh_rt = (
                first_visual.mesh_rpy[0],
                first_visual.mesh_rpy[1],
                first_visual.mesh_rpy[2],
                first_visual.mesh_offset
            )

        # Apply joint properties if this segment comes from a joint
        if parent_joint is not None:
            self._add_joint_to_segment(parent_joint, segment)

        self.segments[link_name] = segment

        # Generate dummy segments for additional visuals
        if len(link.visuals) > 1:
            for i, visual in enumerate(link.visuals[1:], start=1):
                dummy_name = f"{link_name}_visual_{i}"
                dummy_segment = BioModSegment(dummy_name, parent=link_name)
                # Set up properties for visual-only dummy segment
                dummy_segment.mass = 0.0
                dummy_segment.inertia = np.zeros((3, 3))
                dummy_segment.com = (0, 0, 0)
                dummy_segment.mesh_file = visual.mesh_file
                dummy_segment.mesh_scale = visual.mesh_scale
                dummy_segment.mesh_rt = (
                    visual.mesh_rpy[0],
                    visual.mesh_rpy[1],
                    visual.mesh_rpy[2],
                    visual.mesh_offset
                )
                
                # Rigidity relative to parent segment: Identity transformation matrix
                dummy_segment.rt_matrix = np.eye(4)
                dummy_segment.rt_in_matrix = 1
                
                self.segments[dummy_name] = dummy_segment

        # Find all joints where this link is the parent
        for joint in self.parser.joints.values():
            if joint.parent == link_name:
                # Recursively process child link, passing the joint info
                self._build_segments(joint.child, parent_segment=link_name, parent_joint=joint)

    def _add_joint_to_segment(self, joint: Joint, child_segment: BioModSegment):
        """Add joint information to the child segment"""

        # Determine rotation/translation axes
        if joint.type == 'revolute':
            axis_char = self.transform_helper.get_rotation_axis_and_angle(
                joint.origin_rpy, joint.axis
            )
            child_segment.rotations.append(axis_char)
            child_segment.ranges.append((joint.limit_lower, joint.limit_upper))

        elif joint.type == 'prismatic':
            axis_char = self.transform_helper.get_rotation_axis_and_angle(
                (0, 0, 0), joint.axis
            )
            child_segment.translations.append(axis_char)
            child_segment.ranges.append((joint.limit_lower, joint.limit_upper))

        # Set transformation matrix
        rotation_matrix = self.transform_helper.rpy_to_matrix(joint.origin_rpy)
        child_segment.rt_matrix[:3, :3] = rotation_matrix
        child_segment.rt_matrix[:3, 3] = joint.origin_xyz
        child_segment.rt_in_matrix = 1  # Use matrix form

    def _generate_biomod(self) -> str:
        """Generate the complete bioMod file content"""
        lines = ["version 4", ""]

        # Add ground segment
        lines.append("segment ground")
        lines.append("endsegment")
        lines.append("")

        # Add all segments in order
        for segment in self.segments.values():
            lines.append(segment.format_for_biomod())

        return "\n".join(lines)

    def save(self, output_file: str):
        """Save converted bioMod to file"""
        content = self.convert()
        Path(output_file).write_text(content)
        print(f"[OK] Saved to {output_file}")


def main():
    """Example usage"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python urdf_to_biomod_converter.py <urdf_file> [output_file]")
        print("\nExample:")
        print("  python urdf_to_biomod_converter.py robot.urdf robot.bioMod")
        sys.exit(1)

    urdf_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else urdf_file.replace('.urdf', '.bioMod')

    converter = URDFToBioModConverter(urdf_file)
    converter.save(output_file)

    print(f"\n[OK] Conversion complete!")
    print(f"  Input:  {urdf_file}")
    print(f"  Output: {output_file}")


if __name__ == '__main__':
    main()
