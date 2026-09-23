from pathlib import Path

from setuptools import setup, find_packages

long_description = (Path(__file__).parent / 'README.md').read_text(encoding='utf-8')

setup(
    name='experiments',
    version='0.1',
    author='Alexander Wagner',
    author_email='alexander.wagner@cea.fr',
    description='Measurement-control and analysis tools for flux-map and two-tone spectroscopy of superconducting circuits.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/err4re/experiments',
    packages=find_packages(),
    classifiers=[
        # No LICENSE file is checked in yet -- add one and restore a
        # 'License :: OSI Approved :: ...' classifier once the license is decided.
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',  # experiment_data.py uses @dataclass(kw_only=True)
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'h5py',
        'qutip',
        'scikit-learn',
        'lmfit',
        # resonator is on PyPI (currently at 1.1.0), but the only version this
        # codebase has actually been run against is 0.8.0 (see
        # requirements-lock.txt) -- capped here until someone verifies the
        # analysis.py fitting calls (shunt.LinearShuntFitter, see.triptych,
        # background.MagnitudeSlopeOffsetPhaseDelay) still work on 1.x.
        'resonator<1.0',
        'PyQt6',
        'ipython',
        'tqdm',
        # Note: the `instruments` package (experiment_config.py, experiment.py,
        # flux_map_experiment.py, two_tone_experiment.py all import from it) is
        # this lab's own instrument-driver package
        # (https://github.com/err4re/instruments), not published on PyPI --
        # install it separately, e.g.:
        #   pip install -e git+https://github.com/err4re/instruments.git#egg=instruments
        # before this package will actually import. It can't be listed here
        # without a resolvable index/URL.
    ],
)
