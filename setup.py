"""Setup script."""

from pathlib import Path
import re
import setuptools


if __name__ == "__main__":
     # Read description from README
    with Path(Path(__file__).parent, "README.md").open(encoding="utf-8") as file:
        long_description = file.read()
	
	# reading long description from file
	with Path(Path(__file__).parent, "REQUIREMENTS.txt").open(encoding="utf-8") as file:
	    REQUIREMENTS = file.read()

    # Run setup
    setuptools.setup(
        name="image_embeddings",
        author=Ashish Verma,
        version="0.1.0",
        tests_require=["pytest", "black"],
        dependency_links=[],
        data_files=[(".", ["requirements.txt", "README.md"])],
        entry_points={"console_scripts": ["image_embeddings = image_embeddings.cli.main:main"]},
        packages=setuptools.find_packages(),
        description=long_description.split("\n")[0],
        long_description=long_description,
        long_description_content_type="text/markdown",
		url="hhttps://github.com/ashishverma30/image_embedding",
		classifiers=CLASSIFIERS,
		install_requires=REQUIREMENTS,
        classifiers=[
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "Development Status :: 5 - Production/Stable",
            "Topic :: Scientific/Engineering :: Deep Learning",
        ],
		package_dir={"": "src"},
		packages=setuptools.find_packages(where="src"),
		python_requires=">=3.6",
    )	
	