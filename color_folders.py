import os
import subprocess

colors = {
    "01_Data": "#E74C3C", # Red
    "02_Experiments": "#2ECC71", # Green
    "03_Core": "#3498DB", # Blue
    "04_Logs_and_Results": "#F1C40F", # Yellow
    "05_Docs": "#9B59B6" # Purple
}

svg_template = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" width="256" height="256">
  <path fill="{color}" d="M40 12H22l-4-4H8c-2.2 0-4 1.8-4 4v8h40v-4c0-2.2-1.8-4-4-4z"/>
  <path fill="{color}" d="M40 16H8c-2.2 0-4 1.8-4 4v16c0 2.2 1.8 4 4 4h32c2.2 0 4-1.8 4-4V20c0-2.2-1.8-4-4-4z"/>
  <path fill="#000000" opacity="0.15" d="M40 16H8c-2.2 0-4 1.8-4 4v16c0 2.2 1.8 4 4 4h32c2.2 0 4-1.8 4-4V20c0-2.2-1.8-4-4-4z"/>
</svg>"""

for folder, color in colors.items():
    icon_path = os.path.abspath(f".{folder}_icon.svg")
    with open(icon_path, "w") as f:
        f.write(svg_template.format(color=color))
    
    cmd = ["gio", "set", "-t", "string", folder, "metadata::custom-icon", f"file://{icon_path}"]
    subprocess.run(cmd)
    
print("Successfully applied color-coded folder icons using GIO metadata!")
