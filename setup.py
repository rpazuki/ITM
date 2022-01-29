from distutils.core import setup

# http://www.diveintopython3.net/packaging.html
# https://pypi.python.org/pypi?:action=list_classifiers

setup(
    name='ITM',
    packages=['itm.distributions', 'itm.measures', 'itm.utils'],
    include_package_data=True,
    version='0.0.1',
    description='Information Theoretic Measures',
    author='Roozbeh H. Pazuki',
    author_email='r.pazuki@gmail.com',
    url='https://github.com/rpazuki/ITM',
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Development Status :: 5 - Production/Stable",
        "Operating System :: POSIX :: Linux",
        "Intended Audience :: Science/Research",
        "Environment :: Console",
        "Environment :: Other Environment",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ]
)
