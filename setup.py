from setuptools import setup, find_packages


setup(
    name='textvec',
    use_scm_version=True,
    description='Supervised text features extraction',
    version='3.0',
    url='https://github.com/textvec/textvec',
    author='Alex Zverianskii',
    author_email='alex@zverianskii.com',
    license='MIT',
    classifiers=[
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.2',
    'Programming Language :: Python :: 3.3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9'
    ],
    keywords='text nlp vectorization scikit-learn',
    packages=find_packages(exclude=['examples']),
    install_requires=['scikit-learn', 'numpy', 'scipy', 'gensim'],
)
