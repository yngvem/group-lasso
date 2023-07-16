from setuptools import find_packages, setup

setup(name="group-lasso",
      version="1.5.0",
      description="Fast group lasso regularised linear models in a sklearn-style API",
      author="Yngve Mardal Moe",
      author_email="yngve.m.moe@gmail.com",
      url="https://group-lasso.readthedocs.io/en/latest/",
      packages=find_packages(
            where="src",
            include=["group_lasso*"]
      ),
      package_dir={"": "src"},
      include_package_data=True
      )
