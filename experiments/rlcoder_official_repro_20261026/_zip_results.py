
import shutil, os
src = "/content/cadre-results/cadre_v28_8004"
dst = "/content/cadre-results/cadre_v28_8004"
shutil.make_archive(dst, 'tar', root_dir="/content/cadre-results", base_dir="cadre_v28_8004")
print("Archive created at", dst + ".tar", flush=True)
