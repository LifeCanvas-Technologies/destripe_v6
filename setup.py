import setuptools

VERSION = "1.0.0"

with open('README.md') as file:
    long_description = file.read()

# REQUIREMENTS = ['pystripe']

CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'Programming Language :: Python :: 3.11',
]

# calling the setup function 
setuptools.setup(name='cl_destripe_6',
      version=VERSION,
      description='A command line auto-destriping tool',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/lifecanvastechnologies',
      author='LifeCanvas Technologies',
      license='MIT',
      packages=setuptools.find_packages(),
      include_package_data=True,
      package_data={'': ['data/config.ini', 'data/lct.ico', 'data/DestripeGUI.exe']},
      classifiers=CLASSIFIERS,
      # install_requires=REQUIREMENTS,
      entry_points={
        'console_scripts' : ['command_line_destripe=destripegui.destripegui:main',
                             'cpu-destripe=destripegui.destripe.core:main',
                              'gpu-destripe=destripegui.destripe.core_gpu:main',
                              'revert_from_orig=destripegui.revert_from_orig:main',]
      }
)
