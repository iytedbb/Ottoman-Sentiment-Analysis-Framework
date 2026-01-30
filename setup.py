"""
Ottoman Sentiment Analysis Framework
=================================

A comprehensive NLP framework for analyzing Late Ottoman Turkish memoirs (1900-1950).

Features:
    - Named Entity Recognition (NER)
    - Classical Sentiment Analysis
    - Cross-Individual Sentiment Analysis (CISA)
    - End-to-end pipelines

Supported by TÜBİTAK Project No: 323K372
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="Ottoman-Sentiment-Analysis-Framework",
    version="1.0.0",
    author="Mustafa İlter, Emre Onuç, Doğan Evecen, Buket Erşahin, Yasemin Özcan Gönülal, Sezen Karabulut, İbrahim Berci, Selma Tekir",
    author_email="mustafailter@iyte.edu.tr",
    description="NLP framework for analyzing historical Turkish texts from Late Ottoman period",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/iytedbb/Ottoman-Sentiment-Analysis-Framework",
    project_urls={
        "Bug Tracker": "https://github.com/iytedbb/Ottoman-Sentiment-Analysis-Framework/issues",
        "Documentation": "https://github.com/iytedbb/Ottoman-Sentiment-Analysis-Framework#readme",
        "Source Code": "https://github.com/iytedbb/Ottoman-Sentiment-Analysis-Framework",
        "HuggingFace Models": "https://huggingface.co/dbbiyte",
    },
    packages=find_packages(exclude=["tests", "docs", "examples"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "License :: Other/Proprietary License",  # CC BY-NC 4.0
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
    },
    include_package_data=True,
    package_data={
        "ottoman_sentiment_analysis": ["*.md", "*.txt", "*.bib", "*.json"], "datasets": ["*.json"],
    },
    entry_points={
        "console_scripts": [
            "ottoman-sentiment=ottoman_sentiment_analysis.cli:main",
        ],
    },
    keywords=[
        "nlp",
        "turkish",
        "historical",
        "ottoman",
        "named-entity-recognition",
        "sentiment-analysis",
        "entity-based-sentiment-analysis",
        "cross-individual-sentiment-analysis",
        "digital-humanities",
        "bert",
        "transformer",
        "pytorch",
    ],
    license="CC BY-NC 4.0",
    zip_safe=False,
)
