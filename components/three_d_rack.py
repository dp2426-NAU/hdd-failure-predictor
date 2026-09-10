"""
components/three_d_rack.py

A tasteful, purposeful use of 3D: a rack of physical drives you can rotate,
each one colored by its current predicted risk tier. This is NOT a 3D chart
encoding data as depth/perspective (3D bar/pie charts distort the actual
numbers and are avoided throughout this project) -- it's a literal small
model of physical hardware, which is one of the few cases where 3D helps
rather than hurts: it reads instantly as "a room full of drives" and the
color coding does the actual data communication.

Actual quantitative charts (risk trend, feature importance) stay flat 2D
elsewhere in the dashboard, per standard data-viz practice.
"""

import json

import streamlit.components.v1 as components

MAX_DRIVES_RENDERED = 180  # keep the scene light and legible


def render_rack(drives: list[dict], height: int = 440):
    """
    drives: list of {"serial": str, "tier": "healthy"|"elevated"|"critical", "risk": float}
    """
    payload = json.dumps(drives[:MAX_DRIVES_RENDERED])

    html = f"""
    <div id="rack-wrap" style="width:100%;height:{height}px;border-radius:12px;overflow:hidden;
         background:radial-gradient(circle at 50% 0%, #1c2528 0%, #101618 70%);position:relative;">
      <canvas id="rack-canvas" style="display:block;width:100%;height:100%;"></canvas>
      <div id="rack-tooltip" style="position:absolute;pointer-events:none;display:none;
           background:#1a2225;border:1px solid #2b3538;color:#e9efed;padding:6px 10px;
           border-radius:8px;font:12px 'IBM Plex Mono',monospace;z-index:10;"></div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script>
    (function() {{
        const drives = {payload};
        const tierColor = {{ healthy: 0x4caf7d, elevated: 0xe0ab54, critical: 0xe0705a }};

        const wrap = document.getElementById('rack-wrap');
        const canvas = document.getElementById('rack-canvas');
        const tooltip = document.getElementById('rack-tooltip');

        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(45, wrap.clientWidth / {height}, 0.1, 1000);

        const renderer = new THREE.WebGLRenderer({{ canvas: canvas, antialias: true, alpha: true }});
        renderer.setSize(wrap.clientWidth, {height});
        renderer.setPixelRatio(window.devicePixelRatio || 1);

        scene.add(new THREE.AmbientLight(0xffffff, 0.55));
        const key = new THREE.DirectionalLight(0xffffff, 0.9);
        key.position.set(6, 10, 8);
        scene.add(key);
        const rim = new THREE.DirectionalLight(0x57d6c4, 0.35);
        rim.position.set(-8, 4, -6);
        scene.add(rim);

        const n = drives.length;
        const cols = Math.max(1, Math.ceil(Math.sqrt(n * 1.6)));
        const rows = Math.ceil(n / cols);
        const spacing = 0.62;
        const boxGeo = new THREE.BoxGeometry(0.46, 0.9, 0.28);

        const group = new THREE.Group();
        const meshes = [];
        drives.forEach((d, i) => {{
            const col = i % cols;
            const row = Math.floor(i / cols);
            const color = tierColor[d.tier] !== undefined ? tierColor[d.tier] : 0x4caf7d;
            const mat = new THREE.MeshStandardMaterial({{
                color: color, metalness: 0.35, roughness: 0.45,
                emissive: color, emissiveIntensity: d.tier === 'critical' ? 0.35 : 0.08,
            }});
            const mesh = new THREE.Mesh(boxGeo, mat);
            mesh.position.set(
                (col - cols / 2) * spacing,
                0,
                (row - rows / 2) * spacing
            );
            mesh.userData = d;
            group.add(mesh);
            meshes.push(mesh);
        }});
        scene.add(group);

        const rackWidth = cols * spacing;
        const dist = Math.max(6, rackWidth * 0.9);
        camera.position.set(dist * 0.55, dist * 0.6, dist * 0.9);
        camera.lookAt(0, 0, 0);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.08;
        controls.autoRotate = true;
        controls.autoRotateSpeed = 0.6;
        controls.enablePan = false;
        controls.minDistance = dist * 0.4;
        controls.maxDistance = dist * 2;

        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();

        function onMove(event) {{
            const rect = canvas.getBoundingClientRect();
            mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
            mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
            raycaster.setFromCamera(mouse, camera);
            const hits = raycaster.intersectObjects(meshes);
            if (hits.length > 0) {{
                const d = hits[0].object.userData;
                tooltip.style.display = 'block';
                tooltip.style.left = (event.clientX - rect.left + 12) + 'px';
                tooltip.style.top = (event.clientY - rect.top + 8) + 'px';
                tooltip.innerHTML = d.serial + '<br>risk: ' + Math.round(d.risk * 100) + '% (' + d.tier + ')';
            }} else {{
                tooltip.style.display = 'none';
            }}
        }}
        canvas.addEventListener('mousemove', onMove);
        canvas.addEventListener('mouseleave', () => tooltip.style.display = 'none');

        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}
        animate();

        window.addEventListener('resize', () => {{
            const w = wrap.clientWidth;
            camera.aspect = w / {height};
            camera.updateProjectionMatrix();
            renderer.setSize(w, {height});
        }});
    }})();
    </script>
    """
    components.html(html, height=height, scrolling=False)
