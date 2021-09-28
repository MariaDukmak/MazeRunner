import setuptools

install_requires = [
    'gym',
    'numpy',
    'Pillow',
    'opencv-python',
    'matplotlib',
    'seaborn',
    'pandas'
]

extras = {
    "dev": [
        'flake8',
        'flake8-blind-except',
        "flake8-builtins",
        "flake8-docstrings",
        "flake8-logging-format",
        "mypy",
        "pytest"]
}

setuptools.setup(
    name='fancymazerunner',
    description='A fancymazerunner agent based simulation',
    version='0.0.1',
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    extras_require=extras,
    python_requires='>=3.6',
    author='Maria Dukmak',
    author_email='Maria.dukmak@student.hu.nl',
    keyword=['agents', 'agent-based'],
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    include_package_data=True,
)
