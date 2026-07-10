import os
import base64
import zipfile
import io

sol_dir = r"c:\Users\Asus\Downloads\sber_mesh_qc\sber_mesh_qc\solution"
buf = io.BytesIO()

with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
    for r, d, fs in os.walk(sol_dir):
        if "__pycache__" in r:
            continue
        for f in fs:
            if f.endswith(".py"):
                full_path = os.path.join(r, f)
                rel_path = os.path.relpath(full_path, sol_dir).replace("\\", "/")
                zf.write(full_path, rel_path)

# Write solution.zip directly to workspace (Step 10)
zip_data = buf.getvalue()
with open(os.path.join(os.path.dirname(sol_dir), "solution.zip"), "wb") as f:
    f.write(zip_data)

b64_str = base64.b64encode(zip_data).decode("utf-8")

out_path = r"c:\Users\Asus\Downloads\sber_mesh_qc\sber_mesh_qc\self_extract_cell.py"
with open(out_path, "w", encoding="utf-8") as out:
    out.write("import base64, zipfile, io, os\n\n")
    out.write(f'b64_data = "{b64_str}"\n\n')
    out.write("zip_bytes = base64.b64decode(b64_data.encode('utf-8'))\n")
    out.write("os.makedirs('solution', exist_ok=True)\n")
    out.write("with open('solution.zip', 'wb') as f:\n")
    out.write("    f.write(zip_bytes)\n\n")
    out.write("with zipfile.ZipFile('solution.zip', 'r') as zip_ref:\n")
    out.write("    zip_ref.extractall('solution')\n\n")
    out.write("print('Successfully extracted solution/ package (v7.2 Frontier)!')\n")

print(f"[OK] Universal Self-extracting cell script created at {out_path} ({len(b64_str)} chars)")
