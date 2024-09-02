from setuptools import setup
from setuptools import find_packages
with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content]
setup(
    name="ValTool",
    version="0.0.10",
    author="Georgy Levin; Tomas Benavente; Larry Miller",
    description="AI-Powered Valuation Tool",
    packages=find_packages(),
    install_requires=requirements
)
