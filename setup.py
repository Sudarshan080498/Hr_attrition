from setuptools import setup, find_packages
from typing import List


Project_name = 'HR_Department_project'
Version = "0.0.1"
Description = 'Attrition analysis'
Author = 'Sudarshan'
Author_email = "sudarshan080498@gmail.com"
Requirement_file_name = "requirements.txt"



Hyphen_e_dot = "-e ."




def get_requirements_list()->List[str]:
    with open(Requirement_file_name) as requirement_file:
        requirement_list = requirement_file.readline()
        requirement_list = [requirement_name.replace("\n","")for requirement_name in requirement_list]


        if Hyphen_e_dot in requirement_list:
            requirement_list.remove(Hyphen_e_dot)


            return requirement_list





setup(name = Project_name,
      version = Version,
      description=Description,
      author=Author,
      author_email=Author_email,
      packages=find_packages(),
      install_requires = get_requirements_list(),

      )