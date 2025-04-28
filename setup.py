from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = '-e .'
def get_requirements(file_path:str)->List[str]:
    """
    This function will return the list of requirements
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name="DestekYanimda.AI",
    version="0.0.1",
    author="Bedirhan Durmus",
    author_email="bedirhan.durmus@gmail.com",
    description="A machine learning project for tech sales prediction and recommendation system",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)