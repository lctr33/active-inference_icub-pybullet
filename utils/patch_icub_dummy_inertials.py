import xml.etree.ElementTree as ET
from pathlib import Path
import shutil

SRC = Path("/home/lctr/miniforge3/envs/icub-pybullet/share/iCub/robots/iCubGazeboV2_5_visuomanip/model.urdf")
DST = SRC.with_name("model_patched.urdf")

DUMMY_MASS = "1e-6"
DUMMY_INERTIA = "1e-9"

shutil.copy2(SRC, DST)

tree = ET.parse(DST)
root = tree.getroot()

patched = []

for link in root.findall("link"):
    name = link.attrib["name"]

    has_inertial = link.find("inertial") is not None
    has_visual = link.find("visual") is not None
    has_collision = link.find("collision") is not None

    if has_inertial:
        continue

    # Solo parchear links auxiliares sin geometría.
    # Esto evita tocar partes físicas reales.
    if has_visual or has_collision:
        print(f"[SKIP] {name}: no tiene inertial, pero tiene visual/collision")
        continue

    inertial = ET.Element("inertial")

    origin = ET.SubElement(inertial, "origin")
    origin.set("xyz", "0 0 0")
    origin.set("rpy", "0 0 0")

    mass = ET.SubElement(inertial, "mass")
    mass.set("value", DUMMY_MASS)

    inertia = ET.SubElement(inertial, "inertia")
    inertia.set("ixx", DUMMY_INERTIA)
    inertia.set("ixy", "0")
    inertia.set("ixz", "0")
    inertia.set("iyy", DUMMY_INERTIA)
    inertia.set("iyz", "0")
    inertia.set("izz", DUMMY_INERTIA)

    link.insert(0, inertial)
    patched.append(name)

tree.write(DST, encoding="utf-8", xml_declaration=True)

print(f"URDF original:  {SRC}")
print(f"URDF parcheado: {DST}")
print(f"Links parcheados: {len(patched)}")

for name in patched:
    print(f"  {name}")