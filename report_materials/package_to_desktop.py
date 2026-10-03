import os
import shutil
import zipfile

desktop_paths = [
    r"C:\Users\DELL\OneDrive\Desktop",
    r"C:\Users\DELL\Desktop"
]

# Create staging directory
staging_dir = "report_materials/package_staging"
os.makedirs(staging_dir, exist_ok=True)
os.makedirs(os.path.join(staging_dir, "screenshots"), exist_ok=True)

# Copy Word report
shutil.copy("report_materials/MLOps_A01_Report.docx", os.path.join(staging_dir, "MLOps_A01_Report.docx"))

# Copy standalone screenshot images
fig_dir = "report_materials/figures"
for f in os.listdir(fig_dir):
    if f.endswith(".png"):
        shutil.copy(os.path.join(fig_dir, f), os.path.join(staging_dir, "screenshots", f))

# Copy text logs
shutil.copy("report_materials/git_tree.txt", os.path.join(staging_dir, "git_tree.txt"))
shutil.copy("report_materials/merge_conflict_terminal_log.txt", os.path.join(staging_dir, "merge_conflict_terminal_log.txt"))

# Build ZIP file
zip_name = "MLOps_A01_Screenshots_and_Report.zip"
zip_path = os.path.join("report_materials", zip_name)

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
    for root, _, files in os.walk(staging_dir):
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, staging_dir)
            zipf.write(full_path, rel_path)

print(f"Staged and created {zip_path}")

# Deploy to Desktop locations
for d in desktop_paths:
    if os.path.exists(d):
        target_zip = os.path.join(d, zip_name)
        target_doc = os.path.join(d, "MLOps_A01_Report.docx")
        shutil.copy(zip_path, target_zip)
        shutil.copy("report_materials/MLOps_A01_Report.docx", target_doc)
        print(f"Successfully copied to: {d}")
