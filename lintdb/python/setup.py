from setuptools import setup, find_packages
import os
import shutil
import platform

import codecs
from pathlib import Path

def read(rel_path):
    project_root = Path(__file__).parents[0]
    with codecs.open(os.path.join(str(project_root), rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    """ this depends on cmake copying version.txt from the root directory."""
    for line in read(rel_path).splitlines():
        return line.strip()
    else:
        raise RuntimeError("Unable to find version string.")
    

shutil.rmtree("lintdb", ignore_errors=True)
os.mkdir("lintdb")
shutil.copyfile("__init__.py", "lintdb/__init__.py")

ext = ".pyd" if platform.system() == 'Windows' else ".so"
prefix = "Release/" * (platform.system() == 'Windows')

pylintdb_lib = f"{prefix}_pylintdb{ext}"

"""
I don't know we need to copy file like this.
"""
print(f"Copying {pylintdb_lib}")
shutil.copyfile("lintdb.py", "lintdb/lintdb.py")
shutil.copyfile("document.py", "lintdb/document.py")
shutil.copyfile("schema.py", "lintdb/schema.py")
try:
    shutil.copyfile(pylintdb_lib, f"lintdb/_pylintdb{ext}") # we use pylintdb as cmake's python target.
except:
    # if using cmake presets, we still put the library in the Release directory.
    shutil.copyfile("Release/"+pylintdb_lib, f"lintdb/_pylintdb{ext}")

long_description="""
PylintDB is a library for efficient late interaction similarity search. It
implements ColBERTv2 with the Plaid Engine. 
"""
setup(
    name='lintdb',
    version=get_version("version.txt"),
    description='A library for efficient late interaction',
    long_description=long_description,
    author='DeployQL',
    author_email='matt@deployql.com',
    license='MIT',
    keywords='search nearest neighbors colbert',
    install_requires=['numpy', 'packaging'],
    packages=['lintdb'],
    package_data={
        'lintdb': ['*.so', '*.pyd'],
    },
    zip_safe=False,
)