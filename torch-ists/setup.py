from setuptools import setup, find_packages

install_requires = [
    'ray==2.3.0',
    'numpy==1.23.5',
    'pandas==1.5.3',
    'matplotlib==3.7.1',
    'scipy==1.10.1',
    'scikit-learn==1.2.2',
    'tqdm==4.65.0',
    'sktime==0.16.1',
    'stribor==0.1.0',
    'torchcde==0.2.5',
    'torchdiffeq==0.2.1',
    'torchdyn==1.0.4',
    'torchmetrics==0.9.3',
    'torchsde==0.2.5',
]

# with open('README.md', 'r') as f:
#     long_description = f.read()

setup(name='torch_ists',
      version='0.5.0',
      description='Pytorch ISTS',
      # long_description=long_description,
      # long_description_content_type='text/markdown',
      url='',
      author='',
      author_email='',
      packages=find_packages(),
      install_requires=install_requires,
      python_requires='>=3.9',
      zip_safe=False,
)