import setuptools, shutil
from distutils.command.clean import clean as CleanCommand

with open("README.md", "r") as fh:
    long_description = fh.read()

class Clean(CleanCommand):
    def run(self):
        super().run()
        shutil.rmtree('./build', ignore_errors=True)
        shutil.rmtree('./dist', ignore_errors=True)
        shutil.rmtree('./emodeconnection.egg-info', ignore_errors=True)

setuptools.setup(
    name="emodeconnection",
    version="1.0.8",
    author="EMode Photonix LLC",
    author_email="hello@emodephotonix.com",
    description="Python connection for EMode",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/emode-photonix/emodeconnection",
    packages=setuptools.find_packages(),
    cmdclass={'clean': Clean},
    install_requires=[
        "numpy", "dill", "scipy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
    ],
    python_requires='>=3.6',
)

