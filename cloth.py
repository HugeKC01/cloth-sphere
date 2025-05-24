# Minimal cloth simulation falling on a sphere using OpenGL and GLSL in a single file
# Requires: pip install PyOpenGL glfw numpy

import glfw
from OpenGL.GL import *
import numpy as np
import ctypes
import sys
import os
try:
    from PIL import Image
except ImportError:
    Image = None

# Cloth parameters
CLOTH_SIZE = 0.75
CLOTH_RES = 20
CLOTH_MASS = 1.0
GRAVITY = np.array([0, -9.8, 0], dtype=np.float32)
DT = 0.016

# Sphere parameters
SPHERE_CENTER = np.array([0, 0.0, 0], dtype=np.float32)  # Raised sphere higher
SPHERE_RADIUS = 0.25

# Vertex and fragment shaders
VERT_SHADER = """
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texcoord;
out vec3 fragNormal;
out vec2 fragTexcoord;
uniform mat4 MVP;
void main() {
    fragNormal = normal;
    fragTexcoord = texcoord;
    gl_Position = MVP * vec4(position, 1.0);
}
"""
FRAG_SHADER = """
#version 330 core
in vec3 fragNormal;
in vec2 fragTexcoord;
out vec4 color;
uniform vec3 lightDir;
uniform sampler2D normalMap;
uniform sampler2D customTex;
uniform bool useCustomTex;
uniform bool isFloor; // NEW: tells if fragment is floor
uniform bool isCloth; // NEW: tells if fragment is cloth
uniform vec3 sphereCenter; // NEW: for shadow
uniform float sphereRadius; // NEW: for shadow
void main() {
    vec3 nmap = texture(normalMap, fragTexcoord).rgb * 2.0 - 1.0;
    vec3 perturbedNormal = normalize(fragNormal + 0.3 * nmap);
    float light = dot(normalize(perturbedNormal), normalize(lightDir));
    vec4 baseColor = vec4(0.7, 0.7, 1.0, 1.0);
    if (isCloth && useCustomTex) {
        baseColor = texture(customTex, fragTexcoord);
    }
    if (isFloor) {
        // Floor color
        baseColor = vec4(0.5, 0.5, 0.5, 1.0); // changed to grey
        // Simple shadow: project sphere onto floor (y = -1.0)
        float shadow = 1.0;
        float floorY = -1.0;
        // Compute world pos from texcoord (floor quad is from -1 to 1 in x/z)
        float x = fragTexcoord.x * 2.0 - 1.0;
        float z = fragTexcoord.y * 2.0 - 1.0;
        vec2 floorPos = vec2(x, z);
        vec2 sphereXZ = sphereCenter.xz;
        float dist = length(floorPos - sphereXZ);
        float shadowRadius = sphereRadius * 1.2;
        if (dist < shadowRadius) {
            float fade = smoothstep(shadowRadius, shadowRadius * 0.7, dist);
            shadow = mix(0.4, 1.0, fade); // 0.4 = shadow darkness
        }
        float floorLight = max(0.0, dot(normalize(fragNormal), normalize(lightDir)));
        color = baseColor * (0.3 + 0.7 * floorLight) * shadow;
        return;
    }
    if (light > 0.0) {
        color = baseColor * light;
    } else {
        color = vec4(0.2, 0.2, 0.3, 1.0) * (-light) * 0.2;
    }
}
"""

# Helper: compile shader
def compile_shader(src, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, src)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        raise RuntimeError(glGetShaderInfoLog(shader).decode())
    return shader

def create_shader_program():
    vs = compile_shader(VERT_SHADER, GL_VERTEX_SHADER)
    fs = compile_shader(FRAG_SHADER, GL_FRAGMENT_SHADER)
    prog = glCreateProgram()
    glAttachShader(prog, vs)
    glAttachShader(prog, fs)
    glLinkProgram(prog)
    if not glGetProgramiv(prog, GL_LINK_STATUS):
        raise RuntimeError(glGetProgramInfoLog(prog).decode())
    glDeleteShader(vs)
    glDeleteShader(fs)
    return prog

# Cloth mesh generation
def create_cloth_mesh(n=None, size=None, thickness=0.01):
    if n is None:
        n = CLOTH_RES
    if size is None:
        size = CLOTH_SIZE
    positions = np.zeros((n*n, 3), dtype=np.float32)
    normals = np.zeros((n*n, 3), dtype=np.float32)
    texcoords = np.zeros((n*n, 2), dtype=np.float32)
    for y in range(n):
        for x in range(n):
            i = y*n + x
            positions[i] = [size*(x/(n-1)-0.5), 0.8, size*(y/(n-1)-0.5)]
            normals[i] = [0,1,0]
            texcoords[i] = [x/(n-1), y/(n-1)]
    return positions, normals, texcoords

def create_cloth_mesh_indices(n):
    indices = []
    # Top face
    for y in range(n-1):
        for x in range(n-1):
            i = y*n + x
            indices += [i, i+1, i+n, i+1, i+n+1, i+n]
    # Bottom face (reversed winding)
    for y in range(n-1):
        for x in range(n-1):
            i = y*n + x
            j = n*n + i
            indices += [j, j+n, j+1, j+1, j+n, j+n+1]
    # Sides
    for y in range(n-1):
        for x in range(n-1):
            i0 = y*n + x
            i1 = i0 + 1
            i2 = i0 + n
            i3 = i2 + 1
            j0 = n*n + i0
            j1 = n*n + i1
            j2 = n*n + i2
            j3 = n*n + i3
            # Four sides per cell
            indices += [i0, j0, i1, i1, j0, j1]  # x side
            indices += [i0, i2, j0, j0, i2, j2]  # z side
            indices += [i2, i3, j2, j2, i3, j3]  # x+1 side
            indices += [i1, j1, i3, i3, j1, j3]  # z+1 side
    return np.array(indices, dtype=np.uint32)

def create_cloth_top_indices(n):
    indices = []
    for y in range(n-1):
        for x in range(n-1):
            i = y*n + x
            indices += [i, i+1, i+n, i+1, i+n+1, i+n]
    return np.array(indices, dtype=np.uint32)

def update_cloth_thickness(pos, normals, thickness=0.1):
    n = pos.shape[0]
    positions = np.zeros((n*2, 3), dtype=np.float32)
    all_normals = np.zeros((n*2, 3), dtype=np.float32)
    positions[:n] = pos
    positions[n:] = pos - thickness * normals
    all_normals[:n] = normals
    all_normals[n:] = -normals
    return positions, all_normals

# Mass-spring system for cloth
def get_all_springs(n, positions):
    springs = []
    rest_lengths = []
    # Structural springs
    for y in range(n):
        for x in range(n):
            i = y*n + x
            if x < n-1:
                j = i+1
                springs.append((i, j))
                rest_lengths.append(np.linalg.norm(positions[i] - positions[j]))
            if y < n-1:
                j = i+n
                springs.append((i, j))
                rest_lengths.append(np.linalg.norm(positions[i] - positions[j]))
    # Shear springs
    for y in range(n-1):
        for x in range(n-1):
            i = y*n + x
            springs.append((i, i+n+1))
            rest_lengths.append(np.linalg.norm(positions[i] - positions[i+n+1]))
            springs.append((i+1, i+n))
            rest_lengths.append(np.linalg.norm(positions[i+1] - positions[i+n]))
    # Bending springs
    for y in range(n):
        for x in range(n):
            i = y*n + x
            if x < n-2:
                j = i+2
                springs.append((i, j))
                rest_lengths.append(np.linalg.norm(positions[i] - positions[j]))
            if y < n-2:
                j = i+2*n
                springs.append((i, j))
                rest_lengths.append(np.linalg.norm(positions[i] - positions[j]))
    return springs, np.array(rest_lengths, dtype=np.float32)

def simulate_cloth(pos, old_pos, springs, rest_lengths, fixed, dt):
    n = int(np.sqrt(len(pos)))
    # Verlet integration
    acc = np.tile(GRAVITY, (len(pos),1))
    next_pos = pos + (pos - old_pos) + acc * dt * dt
    # Initial collision (optional, for stability)
    for i in range(len(next_pos)):
        v = next_pos[i]
        to_center = v - SPHERE_CENTER
        dist = np.linalg.norm(to_center)
        if dist < SPHERE_RADIUS:
            next_pos[i] = SPHERE_CENTER + to_center/dist * SPHERE_RADIUS
        # Offset floor collision to prevent z-fighting
        if next_pos[i][1] < -0.99:
            next_pos[i][1] = -0.99
    # Spring constraints (structural)
    for _ in range(3):  # Reduced from 3 to 2 for speed
        for idx, (a, b) in enumerate(springs):
            rest = rest_lengths[idx]
            delta = next_pos[b] - next_pos[a]
            d = np.linalg.norm(delta)
            if d == 0: continue
            diff = (d - rest) * 0.5 * delta/d
            if not fixed[a]: next_pos[a] += diff
            if not fixed[b]: next_pos[b] -= diff
        # Collision after each spring iteration
        for i in range(len(next_pos)):
            v = next_pos[i]
            to_center = v - SPHERE_CENTER
            dist = np.linalg.norm(to_center)
            if dist < SPHERE_RADIUS:
                next_pos[i] = SPHERE_CENTER + to_center/dist * SPHERE_RADIUS
            # Offset floor collision to prevent z-fighting
            if next_pos[i][1] < -0.99:
                next_pos[i][1] = -0.99
    # Pin fixed points
    for i in range(len(next_pos)):
        if fixed[i]:
            next_pos[i] = pos[i]
    return next_pos

def compute_normals(pos, indices):
    normals = np.zeros_like(pos)
    for i in range(0, len(indices), 3):
        a, b, c = indices[i:i+3]
        v1 = pos[b] - pos[a]
        v2 = pos[c] - pos[a]
        n = np.cross(v1, v2)
        normals[a] += n
        normals[b] += n
        normals[c] += n
    nrm = np.linalg.norm(normals, axis=1)
    normals[nrm>0] /= nrm[nrm>0][:,None]
    return normals

# OpenGL buffer setup
def create_vbos(positions, normals, texcoords, indices):
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    vbo_pos = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_pos)
    glBufferData(GL_ARRAY_BUFFER, positions.nbytes, positions, GL_DYNAMIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, False, 0, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    vbo_nrm = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_nrm)
    glBufferData(GL_ARRAY_BUFFER, normals.nbytes, normals, GL_DYNAMIC_DRAW)
    glVertexAttribPointer(1, 3, GL_FLOAT, False, 0, ctypes.c_void_p(0))
    glEnableVertexAttribArray(1)
    vbo_tex = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_tex)
    glBufferData(GL_ARRAY_BUFFER, texcoords.nbytes, texcoords, GL_STATIC_DRAW)
    glVertexAttribPointer(2, 2, GL_FLOAT, False, 0, ctypes.c_void_p(0))
    glEnableVertexAttribArray(2)
    ebo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
    glBindVertexArray(0)
    return vao, vbo_pos, vbo_nrm, vbo_tex

def update_vbos(vbo_pos, vbo_nrm, positions, normals):
    glBindBuffer(GL_ARRAY_BUFFER, vbo_pos)
    glBufferSubData(GL_ARRAY_BUFFER, 0, positions.nbytes, positions)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_nrm)
    glBufferSubData(GL_ARRAY_BUFFER, 0, normals.nbytes, normals)

# Simple MVP matrix
def get_mvp():
    import math
    eye = np.array([0,0.2,3], dtype=np.float32)
    center = np.array([0,0,0], dtype=np.float32)
    up = np.array([0,1,0], dtype=np.float32)
    f = center-eye; f/=np.linalg.norm(f)
    s = np.cross(f, up); s/=np.linalg.norm(s)
    u = np.cross(s, f)
    view = np.identity(4, dtype=np.float32)
    view[:3,0] = s; view[:3,1] = u; view[:3,2] = -f; view[:3,3] = eye
    view = np.linalg.inv(view)
    proj = np.identity(4, dtype=np.float32)
    fov = math.radians(45)
    aspect = 1.0
    near, far = 0.1, 10.0
    f = 1/np.tan(fov/2)
    proj[0,0] = f/aspect
    proj[1,1] = f
    proj[2,2] = (far+near)/(near-far)
    proj[2,3] = (2*far*near)/(near-far)
    proj[3,2] = -1
    proj[3,3] = 0
    return proj @ view

# Draw sphere (simple, not using indices)
def draw_sphere(center, radius, shader, mvp):
    lats, longs = 16, 16
    vertices = []
    normals = []
    indices = []
    for i in range(lats+1):
        lat = np.pi * i / lats
        for j in range(longs+1):
            lon = 2 * np.pi * j / longs
            x = np.sin(lat) * np.cos(lon)
            y = np.cos(lat)
            z = np.sin(lat) * np.sin(lon)
            vertices.append(center + radius * np.array([x,y,z]))
            normals.append([x,y,z])
    for i in range(lats):
        for j in range(longs):
            first = i * (longs + 1) + j
            second = first + longs + 1
            indices += [first, second, first + 1]
            indices += [second, second + 1, first + 1]
    vertices = np.array(vertices, dtype=np.float32)
    normals = np.array(normals, dtype=np.float32)
    indices = np.array(indices, dtype=np.uint32)
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    nbo = glGenBuffers(1)
    ebo = glGenBuffers(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, False, 0, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    glBindBuffer(GL_ARRAY_BUFFER, nbo)
    glBufferData(GL_ARRAY_BUFFER, normals.nbytes, normals, GL_STATIC_DRAW)
    glVertexAttribPointer(1, 3, GL_FLOAT, False, 0, ctypes.c_void_p(0))
    glEnableVertexAttribArray(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
    glUseProgram(shader)
    loc = glGetUniformLocation(shader, "MVP")
    glUniformMatrix4fv(loc, 1, GL_FALSE, mvp.T)
    glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)
    glBindVertexArray(0)
    glDeleteBuffers(1, [vbo])
    glDeleteBuffers(1, [nbo])
    glDeleteBuffers(1, [ebo])
    glDeleteVertexArrays(1, [vao])

# Draw a simple colored quad (for floor and wall)
def draw_quad(p1, p2, p3, p4, color, shader, mvp):
    quad_vertices = np.array([p1, p2, p3, p4], dtype=np.float32)
    normal = np.cross(np.array(p2)-np.array(p1), np.array(p3)-np.array(p1))
    normal = normal / np.linalg.norm(normal)
    normals = np.tile(normal, (4,1)).astype(np.float32)
    texcoords = np.array([[0,0],[1,0],[1,1],[0,1]], dtype=np.float32)  # NEW: for floor shadow
    glUseProgram(shader)
    loc = glGetUniformLocation(shader, "MVP")
    glUniformMatrix4fv(loc, 1, GL_FALSE, mvp.T)
    glEnableVertexAttribArray(0)
    glEnableVertexAttribArray(1)
    glEnableVertexAttribArray(2)
    glVertexAttribPointer(0, 3, GL_FLOAT, False, 0, quad_vertices)
    glVertexAttribPointer(1, 3, GL_FLOAT, False, 0, normals)
    glVertexAttribPointer(2, 2, GL_FLOAT, False, 0, texcoords)
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4)
    glDisableVertexAttribArray(0)
    glDisableVertexAttribArray(1)
    glDisableVertexAttribArray(2)

# Main
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cloth_texture', type=str, default=None, help='Path to custom texture image for the cloth top layer')
    args = parser.parse_args()
    # If no argument is given, use a default image in the project folder if it exists
    if args.cloth_texture is None:
        default_img = os.path.join(os.path.dirname(__file__), 'cloth_texture.jpg')
        if os.path.isfile(default_img):
            args.cloth_texture = default_img
    if not glfw.init():
        print("Failed to initialize GLFW")
        sys.exit(1)
    window = glfw.create_window(800, 800, "Cloth on Sphere", None, None)
    if not window:
        glfw.terminate()
        print("Failed to create window")
        sys.exit(1)
    glfw.make_context_current(window)
    shader = create_shader_program()
    positions, normals, texcoords = create_cloth_mesh(CLOTH_RES, CLOTH_SIZE)  # only top layer
    top_indices = create_cloth_top_indices(CLOTH_RES)
    indices = create_cloth_mesh_indices(CLOTH_RES)
    springs, rest_lengths = get_all_springs(CLOTH_RES, positions)
    fixed = np.zeros(len(positions), dtype=bool)
    old_pos = positions.copy()
    normals = compute_normals(positions, create_cloth_top_indices(CLOTH_RES))
    thick_positions, thick_normals = update_cloth_thickness(positions, normals, thickness=0.01)
    # Duplicate texcoords for both sides
    thick_texcoords = np.vstack([texcoords, texcoords])
    vao, vbo_pos, vbo_nrm, vbo_tex = create_vbos(thick_positions, thick_normals, thick_texcoords, indices)
    glEnable(GL_DEPTH_TEST)

    # --- Normal map setup (procedural fluffy normal map) ---
    def generate_fluffy_normal_map(size=128):
        # Simple random noise normal map
        np.random.seed(42)
        noise = np.random.normal(0, 0.5, (size, size, 2)).astype(np.float32)
        base = np.zeros((size, size, 3), dtype=np.float32)
        base[...,0] = noise[...,0]
        base[...,1] = noise[...,1]
        base[...,2] = np.sqrt(1.0 - np.clip(base[...,0]**2 + base[...,1]**2, 0, 1))
        base = (base * 0.5 + 0.5)  # Map from [-1,1] to [0,1]
        return (base * 255).astype(np.uint8)
    normal_map_data = generate_fluffy_normal_map(128)
    normal_map_tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, normal_map_tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 128, 128, 0, GL_RGB, GL_UNSIGNED_BYTE, normal_map_data)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glBindTexture(GL_TEXTURE_2D, 0)
    # --- Custom texture setup ---
    custom_tex = None
    use_custom_tex = False
    if args.cloth_texture and Image is not None and os.path.isfile(args.cloth_texture):
        img = Image.open(args.cloth_texture).convert('RGBA')
        img_data = np.array(img)[::-1]  # Flip vertically for OpenGL
        custom_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, custom_tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.width, img.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glBindTexture(GL_TEXTURE_2D, 0)
        use_custom_tex = True

    frame_counter = 0
    while not glfw.window_should_close(window):
        glfw.poll_events()
        glClearColor(0.1,0.1,0.15,1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # Simulate
        new_pos = simulate_cloth(positions, old_pos, springs, rest_lengths, fixed, DT)
        # Only update normals every other frame for speed
        if frame_counter % 2 == 0:
            normals = compute_normals(new_pos, top_indices)
            thick_positions, thick_normals = update_cloth_thickness(new_pos, normals, thickness=0.01)
            update_vbos(vbo_pos, vbo_nrm, thick_positions, thick_normals)
        else:
            thick_positions, thick_normals = update_cloth_thickness(new_pos, normals, thickness=0.01)
            update_vbos(vbo_pos, vbo_nrm, thick_positions, thick_normals)
        old_pos, positions = positions, new_pos
        frame_counter += 1
        # Draw cloth
        glUseProgram(shader)
        mvp = get_mvp()
        loc = glGetUniformLocation(shader, "MVP")
        glUniformMatrix4fv(loc, 1, GL_FALSE, mvp.T)
        # Set light direction from the top (straight down)
        light_dir_loc = glGetUniformLocation(shader, "lightDir")
        light_dir = np.array([0.0, 1.0, 0.0])  # from above, pointing down
        glUniform3f(light_dir_loc, *light_dir)
        # Bind normal map
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, normal_map_tex)
        normal_map_loc = glGetUniformLocation(shader, "normalMap")
        glUniform1i(normal_map_loc, 0)
        # Bind custom texture if available
        use_custom_tex_loc = glGetUniformLocation(shader, "useCustomTex")
        glUniform1i(use_custom_tex_loc, 1 if use_custom_tex else 0)
        is_cloth_loc = glGetUniformLocation(shader, "isCloth")
        glUniform1i(is_cloth_loc, 1)  # Set isCloth true for cloth
        if use_custom_tex:
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, custom_tex)
            custom_tex_loc = glGetUniformLocation(shader, "customTex")
            glUniform1i(custom_tex_loc, 1)
        glBindVertexArray(vao)
        glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
        glBindTexture(GL_TEXTURE_2D, 0)
        if use_custom_tex:
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, 0)
        # Draw sphere
        glUseProgram(shader)
        loc = glGetUniformLocation(shader, "MVP")
        glUniformMatrix4fv(loc, 1, GL_FALSE, mvp.T)
        light_dir_loc = glGetUniformLocation(shader, "lightDir")
        glUniform3f(light_dir_loc, *light_dir)
        glUniform1i(use_custom_tex_loc, 0)  # Don't use custom tex for sphere
        glUniform1i(is_cloth_loc, 0)        # Not cloth
        draw_sphere(SPHERE_CENTER, SPHERE_RADIUS, shader, mvp)
        # Draw floor (y = -1.0)
        glUseProgram(shader)
        loc = glGetUniformLocation(shader, "MVP")
        glUniformMatrix4fv(loc, 1, GL_FALSE, mvp.T)
        light_dir_loc = glGetUniformLocation(shader, "lightDir")
        glUniform3f(light_dir_loc, *light_dir)
        # Set floor uniforms
        is_floor_loc = glGetUniformLocation(shader, "isFloor")
        glUniform1i(is_floor_loc, 1)
        glUniform1i(is_cloth_loc, 0)        # Not cloth
        sphere_center_loc = glGetUniformLocation(shader, "sphereCenter")
        glUniform3f(sphere_center_loc, *SPHERE_CENTER)
        sphere_radius_loc = glGetUniformLocation(shader, "sphereRadius")
        glUniform1f(sphere_radius_loc, SPHERE_RADIUS)
        draw_quad(
            [-1.0, -1.0, -1.0],
            [ 1.0, -1.0, -1.0],
            [ 1.0, -1.0,  1.0],
            [-1.0, -1.0,  1.0],
            [0.3, 0.8, 0.3, 1.0], shader, mvp
        )
        glUniform1i(is_floor_loc, 0)  # reset for other draws
        glfw.swap_buffers(window)
    glfw.terminate()
