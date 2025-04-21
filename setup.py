# setup.py
from setuptools import setup, find_packages

setup(
    name='LaTeX_OCR',                 # pip安装时的名字 (不要和已有包重名)
    version='0.1.0',                  # 版本号
    description='Handwritten LaTeX OCR recognition with ResNet + Transformer Decoder',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Jiashu Guo''Junhao Fu',
    author_email='jacob1365826588@163.com',
    url='https://github.com/GuoJiashu/LaTex_Equation_Generator',  # 可选
    license='MIT',                    # 许可证
    packages=find_packages(),          # 自动找到子包
    install_requires=[
        'torch',
        'torchvision',
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'tqdm',
        'pillow',
        'nltk',
        'python-Levenshtein',
        'collections',
        'PIL',
        'os',
        'math',
        're',
        'pickle',
        'random',
        'torch.nn'
    ],
    python_requires='>=3.9',            # 最低Python版本
    include_package_data=True,
    classifiers=[                      # PyPI分类
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
