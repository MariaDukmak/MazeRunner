from setuptools import setup, find_packages

install_requires = [
    'gym',
    'numpy',
    'Pillow',
    'opencv-python',
]

setup(
    name='fancymazerunner',
    description='A fancymazerunner agent based simulation',
    version='0.0.1',
    packages=find_packages(),
    install_requires=install_requires,
    python_requires='>=3.6',
    author='Maria Dukmak',
    author_email='Maria.dukmak@student.hu.nl',
    keyword=['agents', 'agent-based'],
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    include_package_data=True,
)
