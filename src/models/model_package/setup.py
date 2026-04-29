from setuptools import setup, find_packages

setup(
    version='0.1.0',
    name='mlp_package',
    packages=find_packages(),
    package_data={
        'model_package': ['model_package/models/*.pth', 'model_package/models/*.joblib'],
    },
    include_package_data=True,
    install_requires=[
        "torch>=2.10.0",
        "pandas>=2.3.3",
        "scikit-learn>=1.8.0"
    ],
    description="a package for the tech challenge",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)