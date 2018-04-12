from setuptools import setup, find_packages


setup(
    name='textvec',
    use_scm_version=True,
    description='Supervised text features extraction',
    version = '0.0.1',
    url='',
    author='Alex Zveryansky',
    author_email='',
    license='',
    packages=find_packages(),
    install_requires=['scikit-learn','numpy','scipy'],
)
