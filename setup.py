from setuptools import setup, find_packages

requirements = []
setup(
    name="deep-uncertainty",
    author="B. Kelly, S. Malamud, T. A. Xu",
    zip_safe=False,
    requires=requirements,
    packages=find_packages(),
)
