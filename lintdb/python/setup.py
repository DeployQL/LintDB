from setuptools import setup, find_packages
import os
import shutil
import platform

shutil.rmtree("lintdb", ignore_errors=True)
os.mkdir("lintdb")
shutil.copyfile("__init__.py", "lintdb/__init__.py")

ext = ".pyd" if platform.system() == 'Windows' else ".so"
prefix = "Release/" * (platform.system() == 'Windows')

pylintdb_lib = f"{prefix}_pylintdb{ext}"

print(f"Copying {pylintdb_lib}")
shutil.copyfile("lintdb.py", "lintdb/lintdb.py")
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
    version='0.0.1',
    description='A library for efficient late interaction',
    long_description=long_description,
    author='Matt Barta',
    author_email='matt@deployql.com',
    license='MIT',
    keywords='search nearest neighbors',

    install_requires=['numpy', 'packaging'],
    packages=['lintdb'],
    package_data={
        'lintdb': ['*.so', '*.pyd'],
    },
    zip_safe=False,
)