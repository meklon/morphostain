from setuptools import setup, find_packages

setup(
    name='morphostain',
    version='1.0.8',
    description='DAB-chromagen analysis tool',
    longer_description='''
MorphoStain counts the stained area with
DAB-chromagen using the typical immunohystochemistry protocols.
After the analysis user can measure the difference of proteins
content in tested samples. It could also measure areas covered
with other stains.
''',
    maintainer='Ivan Gumenyuk',
    maintainer_email='meklon@gmail.com',
    url='https://github.com/meklon/morphostain',
    packages=find_packages(),
    install_requires=['pandas>=0.17.1', 'numpy>=1.11.0', 'scipy',
        'scikit-image', 'matplotlib', 'seaborn'],
    setup_requires=['pytest-runner'],
    include_package_data = True,
    tests_require=['pytest'],
    license='GPLv3',
    classifiers=['Environment :: Console'],
    package_data={
        'morphostain': ['resources/*'],
    },
    entry_points={'console_scripts': ['morphostain = morphostain:main']},
    )
