import os.path
from setuptools import setup, find_packages
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

setup(
    name="autosyn",
    version="1.0",
    author="all",
    author_email="",
    description="Stay true",

    # 你要安装的包，通过 setuptools.find_packages 找到当前目录下有哪些包
    packages=find_packages()
)