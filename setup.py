from setuptools import setup, find_packages

setup(
    name='experiments',  # Replace with your package's name
    version='0.1',  # Package version
    author='Alexander Wagner',  # Your name or the name of the organization
    author_email='alexander.wagner@cea.fr',  # Your email or the organization's email
    description='A short description of your package',  # Short description
    long_description='A longer description of your package',  # Optional longer description
    url='https://github.com/err4re/experiments',  # Link to your package's repository or website
    packages=find_packages(),  # Automatically find all packages in the directory
    classifiers=[
        # Trove classifiers (https://pypi.org/classifiers/)
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    # Additional metadata
)