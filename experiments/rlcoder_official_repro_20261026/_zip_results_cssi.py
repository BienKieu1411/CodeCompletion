
import shutil, os
import tarfile

src_dir = "/content/cadre-results/cadre_cssi_guided_8004"
out_tar = "/content/cadre_cssi_guided_8004.tar"
if os.path.exists(src_dir):
    print(f"Archiving {src_dir} to {out_tar}...")
    with tarfile.open(out_tar, "w") as tar:
        tar.add(src_dir, arcname="cadre_cssi_guided_8004")
    print("Archive complete, size:", os.path.getsize(out_tar))
