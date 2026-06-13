import xml.etree.ElementTree as ET
from pathlib import Path

URDF = Path("")

tree = ET.parse(URDF)
root = tree.getroot()

links = {}
for link in root.findall("link"):
    name = link.attrib["name"]
    links[name] = {
        "has_inertial": link.find("inertial") is not None,
        "has_visual": link.find("visual") is not None,
        "has_collision": link.find("collision") is not None,
        "parent_joint": None,
        "parent_link": None,
        "joint_type": None,
    }

for joint in root.findall("joint"):
    joint_name = joint.attrib.get("name", "")
    joint_type = joint.attrib.get("type", "")
    parent = joint.find("parent")
    child = joint.find("child")

    if parent is None or child is None:
        continue

    parent_link = parent.attrib["link"]
    child_link = child.attrib["link"]

    if child_link in links:
        links[child_link]["parent_joint"] = joint_name
        links[child_link]["parent_link"] = parent_link
        links[child_link]["joint_type"] = joint_type

missing = {name: data for name, data in links.items() if not data["has_inertial"]}

print(f"URDF: {URDF}")
print(f"Total links: {len(links)}")
print(f"Links sin inertial: {len(missing)}")
print()

for name, data in missing.items():
    print(
        f"{name:35s} "
        f"joint_type={str(data['joint_type']):10s} "
        f"parent={str(data['parent_link']):30s} "
        f"visual={data['has_visual']} "
        f"collision={data['has_collision']}"
    )