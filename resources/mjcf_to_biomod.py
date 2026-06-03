import os
import sys
import numpy as np
import xml.etree.ElementTree as ET

def parse_vector(s, default=None):
    if not s:
        return default
    return [float(x) for x in s.split()]

def parse_quat(quat_str):
    if not quat_str:
        return np.array([1.0, 0.0, 0.0, 0.0])
    q = np.array(parse_vector(quat_str))
    if len(q) == 4:
        return q
    return np.array([1.0, 0.0, 0.0, 0.0])

def parse_euler(euler_str, degrees=True):
    if not euler_str:
        return np.zeros(3)
    e = np.array(parse_vector(euler_str))
    if len(e) == 3:
        return e
    return np.zeros(3)

def quat_to_matrix(q):
    w, x, y, z = q
    norm = (w*w + x*x + y*y + z*z)**0.5
    if norm > 0:
        w, x, y, z = w/norm, x/norm, y/norm, z/norm
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [    2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z,     2*y*z - 2*w*x],
        [    2*x*z - 2*w*y,     2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])

def euler_to_matrix(seq_vals, seq="xyz", degrees=True):
    rad = np.radians(seq_vals) if degrees else np.array(seq_vals)
    cx, cy, cz = np.cos(rad)
    sx, sy, sz = np.sin(rad)
    
    Rx = np.array([
        [1, 0, 0],
        [0, cx, -sx],
        [0, sx, cx]
    ])
    Ry = np.array([
        [cy, 0, sy],
        [0, 1, 0],
        [-sy, 0, cy]
    ])
    Rz = np.array([
        [cz, -sz, 0],
        [sz, cz, 0],
        [0, 0, 1]
    ])
    return Rx @ Ry @ Rz

def matrix_to_euler_xyz(R):
    r13 = R[0, 2]
    r13 = max(-1.0, min(1.0, r13))
    
    if abs(r13) < 0.99999:
        y = np.arcsin(r13)
        x = np.arctan2(-R[1, 2], R[2, 2])
        z = np.arctan2(-R[0, 1], R[0, 0])
    else:
        y = np.pi/2 if r13 > 0 else -np.pi/2
        x = 0.0
        z = np.arctan2(R[1, 0], R[1, 1])
    return np.array([x, y, z])

def matrix_to_quat(R):
    tr = np.trace(R)
    if tr > 0:
        S = (tr + 1.0)**0.5 * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = (1.0 + R[0, 0] - R[1, 1] - R[2, 2])**0.5 * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = (1.0 + R[1, 1] - R[0, 0] - R[2, 2])**0.5 * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = (1.0 + R[2, 2] - R[0, 0] - R[1, 1])**0.5 * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    return np.array([qw, qx, qy, qz])

def map_axis(axis_vector):
    if not axis_vector or len(axis_vector) < 3:
        return 'z'
    abs_axis = [abs(x) for x in axis_vector]
    idx = abs_axis.index(max(abs_axis))
    return ['x', 'y', 'z'][idx]

def convert_range(range_vector, is_rotation, angle_in_degrees):
    if not range_vector or len(range_vector) < 2:
        return None
    min_val, max_val = range_vector
    if is_rotation and angle_in_degrees:
        min_val = np.radians(min_val)
        max_val = np.radians(max_val)
    return [min_val, max_val]

class BodyNode:
    def __init__(self, name, parent_name=None):
        self.name = name
        self.parent_name = parent_name
        self.pos = np.zeros(3)
        self.quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.euler = None
        self.joints = []
        self.inertial = None
        self.geoms = []
        self.sites = []

def parse_body(element, parent_name, body_nodes, meshes, compiler_angle_degree):
    name = element.get('name')
    if not name:
        name = f"body_{len(body_nodes)}"
    
    node = BodyNode(name, parent_name)
    body_nodes[name] = node
    
    node.pos = np.array(parse_vector(element.get('pos'), [0.0, 0.0, 0.0]))
    
    if element.get('quat'):
        node.quat = parse_quat(element.get('quat'))
    elif element.get('euler'):
        node.euler = parse_euler(element.get('euler'), compiler_angle_degree)
        R = euler_to_matrix(node.euler, seq="xyz", degrees=compiler_angle_degree)
        node.quat = matrix_to_quat(R)
        
    for child in element:
        if child.tag == 'joint':
            j_name = child.get('name', f"{name}_joint_{len(node.joints)}")
            j_type = child.get('type', 'hinge')
            j_axis_vec = parse_vector(child.get('axis'), [0.0, 0.0, 1.0])
            j_axis = map_axis(j_axis_vec)
            j_range_vec = parse_vector(child.get('range'))
            j_range = convert_range(j_range_vec, is_rotation=(j_type in ['hinge', 'ball']), angle_in_degrees=compiler_angle_degree)
            
            if j_type == 'ball':
                for axis in ['x', 'y', 'z']:
                    node.joints.append({
                        'name': f"{j_name}_{axis}",
                        'type': 'hinge',
                        'axis': axis,
                        'range': j_range or [-np.pi, np.pi]
                    })
            else:
                node.joints.append({
                    'name': j_name,
                    'type': j_type,
                    'axis': j_axis,
                    'range': j_range
                })
                
        elif child.tag == 'inertial':
            mass = float(child.get('mass', 0.0))
            com = np.array(parse_vector(child.get('pos'), [0.0, 0.0, 0.0]))
            
            R_inertial = np.eye(3)
            if child.get('quat'):
                q_in = parse_quat(child.get('quat'))
                R_inertial = quat_to_matrix(q_in)
            elif child.get('euler'):
                e_in = parse_euler(child.get('euler'), compiler_angle_degree)
                R_inertial = euler_to_matrix(e_in, seq="xyz", degrees=compiler_angle_degree)
                
            I_local = np.zeros((3, 3))
            if child.get('diaginertia'):
                diag = parse_vector(child.get('diaginertia'))
                I_local = np.diag(diag)
            elif child.get('fullinertia'):
                full = parse_vector(child.get('fullinertia'))
                ixx, iyy, izz, ixy, ixz, iyz = full
                I_local = np.array([
                    [ixx, ixy, ixz],
                    [ixy, iyy, iyz],
                    [ixz, iyz, izz]
                ])
                
            I_body = R_inertial @ I_local @ R_inertial.T
            node.inertial = {
                'mass': mass,
                'com': com,
                'inertia': I_body
            }
            
        elif child.tag == 'geom':
            g_type = child.get('type', 'sphere')
            g_pos = np.array(parse_vector(child.get('pos'), [0.0, 0.0, 0.0]))
            g_quat = parse_quat(child.get('quat'))
            g_euler = None
            if child.get('euler'):
                g_euler = parse_euler(child.get('euler'), compiler_angle_degree)
                
            meshfile = None
            mesh_scale = [1.0, 1.0, 1.0]
            if g_type == 'mesh':
                mesh_name = child.get('mesh')
                if mesh_name in meshes:
                    meshfile = meshes[mesh_name].get('file')
                    mesh_scale = meshes[mesh_name].get('scale', [1.0, 1.0, 1.0])
                    
            rgba = parse_vector(child.get('rgba'))
            node.geoms.append({
                'type': g_type,
                'pos': g_pos,
                'quat': g_quat,
                'euler': g_euler,
                'meshfile': meshfile,
                'scale': mesh_scale,
                'rgba': rgba,
                'mass': child.get('mass')
            })
            
        elif child.tag == 'site':
            s_name = child.get('name')
            if not s_name:
                s_name = f"{name}_site_{len(node.sites)}"
            s_pos = np.array(parse_vector(child.get('pos'), [0.0, 0.0, 0.0]))
            node.sites.append({
                'name': s_name,
                'pos': s_pos
            })
            
        elif child.tag == 'body':
            parse_body(child, name, body_nodes, meshes, compiler_angle_degree)


def compute_default_inertial(node):
    total_mass = sum(float(g.get('mass', 0.1)) if g.get('mass') else 0.1 for g in node.geoms)
    if total_mass <= 0:
        total_mass = 1.0
    
    com = np.zeros(3)
    geom_count = 0
    for g in node.geoms:
        com += g['pos']
        geom_count += 1
    if geom_count > 0:
        com /= geom_count
        
    inertia = np.diag([0.01, 0.01, 0.01]) * total_mass
    node.inertial = {
        'mass': total_mass,
        'com': com,
        'inertia': inertia
    }

def write_segment(f, name, parent, rt_matrix, rotations=None, translations=None, ranges=None, mass=None, inertia=None, com=None, meshfile=None, meshcolor=None, meshscale=None, meshrt=None):
    f.write(f"segment {name}\n")
    if parent:
        f.write(f"    parent {parent}\n")
        
    f.write("    RTinMatrix 1\n")
    f.write("    RT\n")
    for row in rt_matrix:
        f.write(f"        {row[0]:.6f}\t{row[1]:.6f}\t{row[2]:.6f}\t{row[3]:.6f}\n")
        
    if translations:
        f.write(f"    translations {translations}\n")
    if rotations:
        f.write(f"    rotations {rotations}\n")
        
    if ranges:
        f.write("    ranges\n")
        for r in ranges:
            f.write(f"        {r[0]:.6f} {r[1]:.6f}\n")
            
    if mass is not None:
        f.write(f"    mass {mass:.6f}\n")
    if com is not None:
        f.write(f"    com {com[0]:.6f} {com[1]:.6f} {com[2]:.6f}\n")
    if inertia is not None:
        f.write("    inertia\n")
        f.write(f"        {inertia[0][0]:.6f} {inertia[0][1]:.6f} {inertia[0][2]:.6f}\n")
        f.write(f"        {inertia[1][0]:.6f} {inertia[1][1]:.6f} {inertia[1][2]:.6f}\n")
        f.write(f"        {inertia[2][0]:.6f} {inertia[2][1]:.6f} {inertia[2][2]:.6f}\n")
        
    if meshfile:
        f.write(f"    meshfile {meshfile}\n")
        if meshcolor is not None:
            f.write(f"    meshcolor {meshcolor[0]:.6f} {meshcolor[1]:.6f} {meshcolor[2]:.6f}\n")
        if meshscale is not None:
            f.write(f"    meshscale {meshscale[0]:.6f} {meshscale[1]:.6f} {meshscale[2]:.6f}\n")
        if meshrt is not None:
            f.write(f"    meshrt {meshrt[0]:.6f} {meshrt[1]:.6f} {meshrt[2]:.6f} xyz {meshrt[3]:.6f} {meshrt[4]:.6f} {meshrt[5]:.6f}\n")
            
    f.write("endsegment\n\n")

def convert(mjcf_path, biomod_path):
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    
    # Compiler
    compiler_el = root.find('compiler')
    meshdir = ""
    compiler_angle_degree = True
    if compiler_el is not None:
        meshdir = compiler_el.get('meshdir', '')
        compiler_angle_degree = (compiler_el.get('angle', 'degree') == 'degree')
        
    # Gravity
    gravity = [0.0, 0.0, -9.81]
    option_el = root.find('option')
    if option_el is not None and option_el.get('gravity') is not None:
        gravity = parse_vector(option_el.get('gravity'))
        
    # Meshes
    meshes = {}
    asset_el = root.find('asset')
    if asset_el is not None:
        for mesh in asset_el.findall('mesh'):
            m_name = mesh.get('name')
            m_file = mesh.get('file')
            m_scale_str = mesh.get('scale')
            m_scale = parse_vector(m_scale_str, [1.0, 1.0, 1.0])
            
            if not m_name and m_file:
                m_name = os.path.splitext(os.path.basename(m_file))[0]
                
            if m_name:
                if m_file and meshdir and not os.path.isabs(m_file):
                    m_file = os.path.join(meshdir, m_file)
                meshes[m_name] = {
                    'file': m_file,
                    'scale': m_scale
                }
                
    # Worldbody traversal
    worldbody = root.find('worldbody')
    if worldbody is None:
        raise ValueError("worldbody element not found in MJCF file")
        
    body_nodes = {}
    for element in worldbody:
        if element.tag == 'body':
            parse_body(element, 'ground', body_nodes, meshes, compiler_angle_degree)
            
    # Site mapping for site_name -> (parent_body_name, pos_array)
    site_mapping = {}
    for node in body_nodes.values():
        for site in node.sites:
            site_mapping[site['name']] = (node.name, site['pos'])
            
    # Parse tendons
    tendons = []
    tendon_el = root.find('tendon')
    if tendon_el is not None:
        for spatial in tendon_el.findall('spatial'):
            name = spatial.get('name')
            if not name:
                name = f"tendon_{len(tendons)}"
            
            sites = []
            for child in spatial:
                if child.tag == 'site':
                    site_name = child.get('site')
                    sites.append(site_name)
                    
            if len(sites) >= 2:
                tendons.append({
                    'name': name,
                    'sites': sites
                })
                
    # Write biomod file
    with open(biomod_path, 'w') as f:
        f.write("version 4\n")
        f.write(f"gravity {gravity[0]:.6f} {gravity[1]:.6f} {gravity[2]:.6f}\n\n")
        
        # Segment ground
        f.write("segment ground\nendsegment\n\n")
        
        # Write all body nodes as segments
        for node in body_nodes.values():
            R_body = quat_to_matrix(node.quat)
            rt_matrix = np.eye(4)
            rt_matrix[:3, :3] = R_body
            rt_matrix[:3, 3] = node.pos
            
            trans_axes = []
            rot_axes = []
            ranges = []
            
            for j in node.joints:
                if j['type'] == 'slide':
                    trans_axes.append(j['axis'])
                    ranges.append(j['range'] or [-10.0, 10.0])
                    
            for j in node.joints:
                if j['type'] == 'hinge':
                    rot_axes.append(j['axis'])
                    ranges.append(j['range'] or [-np.pi, np.pi])
                    
            translations_str = "".join(trans_axes) if trans_axes else None
            rotations_str = "".join(rot_axes) if rot_axes else None
            
            if not node.inertial:
                compute_default_inertial(node)
                
            main_geom = node.geoms[0] if len(node.geoms) > 0 else None
            meshfile = main_geom['meshfile'] if main_geom else None
            meshcolor = main_geom['rgba'][:3] if (main_geom and main_geom['rgba']) else None
            meshscale = main_geom['scale'] if main_geom else None
            meshrt = None
            if main_geom:
                g_R = quat_to_matrix(main_geom['quat'])
                if main_geom['euler'] is not None:
                    g_R = euler_to_matrix(main_geom['euler'], seq="xyz", degrees=compiler_angle_degree)
                g_euler_rad = matrix_to_euler_xyz(g_R)
                meshrt = [g_euler_rad[0], g_euler_rad[1], g_euler_rad[2], main_geom['pos'][0], main_geom['pos'][1], main_geom['pos'][2]]
                
            write_segment(
                f,
                name=node.name,
                parent=node.parent_name,
                rt_matrix=rt_matrix,
                rotations=rotations_str,
                translations=translations_str,
                ranges=ranges if ranges else None,
                mass=node.inertial['mass'],
                com=node.inertial['com'],
                inertia=node.inertial['inertia'],
                meshfile=meshfile,
                meshcolor=meshcolor,
                meshscale=meshscale,
                meshrt=meshrt
            )
            
            for idx, geom in enumerate(node.geoms[1:]):
                sub_name = f"{node.name}_geom_{idx + 1}"
                sub_rt_matrix = np.eye(4)
                
                g_R = quat_to_matrix(geom['quat'])
                if geom['euler'] is not None:
                    g_R = euler_to_matrix(geom['euler'], seq="xyz", degrees=compiler_angle_degree)
                g_euler_rad = matrix_to_euler_xyz(g_R)
                sub_meshrt = [g_euler_rad[0], g_euler_rad[1], g_euler_rad[2], geom['pos'][0], geom['pos'][1], geom['pos'][2]]
                
                write_segment(
                    f,
                    name=sub_name,
                    parent=node.name,
                    rt_matrix=sub_rt_matrix,
                    meshfile=geom['meshfile'],
                    meshcolor=geom['rgba'][:3] if geom['rgba'] else None,
                    meshscale=geom['scale'],
                    meshrt=sub_meshrt
                )
                
        # Write markers
        for node in body_nodes.values():
            for site in node.sites:
                f.write(f"marker {site['name']}\n")
                f.write(f"    parent {node.name}\n")
                f.write(f"    position {site['pos'][0]:.6f} {site['pos'][1]:.6f} {site['pos'][2]:.6f}\n")
                f.write("endmarker\n\n")
                
        # Write tendons
        for t in tendons:
            name = t['name']
            sites = t['sites']
            
            origin_site_name = sites[0]
            if origin_site_name in site_mapping:
                origin_parent_body, origin_pos = site_mapping[origin_site_name]
            else:
                continue
                
            insertion_site_name = sites[-1]
            if insertion_site_name in site_mapping:
                insertion_parent_body, insertion_pos = site_mapping[insertion_site_name]
            else:
                continue
                
            f.write(f"tendon {name}\n")
            f.write(f"    origin {origin_parent_body}\n")
            f.write(f"    insertion {insertion_parent_body}\n")
            f.write(f"    originPosition {origin_pos[0]:.6f} {origin_pos[1]:.6f} {origin_pos[2]:.6f}\n")
            f.write(f"    insertionPosition {insertion_pos[0]:.6f} {insertion_pos[1]:.6f} {insertion_pos[2]:.6f}\n")
            f.write("endtendon\n\n")
            
            for site_name in sites[1:-1]:
                if site_name in site_mapping:
                    rp_parent_body, rp_pos = site_mapping[site_name]
                    f.write(f"tendonRoutingPoint {site_name}\n")
                    f.write(f"    tendon {name}\n")
                    f.write(f"    parent {rp_parent_body}\n")
                    f.write(f"    position {rp_pos[0]:.6f} {rp_pos[1]:.6f} {rp_pos[2]:.6f}\n")
                    f.write("    frictionLoss 0.0\n")
                    f.write("endtendonRoutingPoint\n\n")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python mjcf_to_biomod.py <input_mjcf_file> <output_biomod_file>")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
