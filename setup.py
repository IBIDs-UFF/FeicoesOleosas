import setuptools


setuptools.setup(
    name='oil',
    version='0.1.0',
    author='Leandro A. F. Fernandes',
    author_email='laffernandes@ic.uff.br',
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        'matplotlib',
        'opencv-python',
        'opencv-contrib-python',
        'tqdm',
    ],
    zip_safe=False)
