"""Setup script for Bamboo."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="bamboo",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Bolstered Assistance for Managing and Building Operations and Oversight (BAMBOO)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/bamboo",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=[
        "langgraph>=0.0.38",
        "langchain>=0.1.0",
        "langchain-openai>=0.0.5",
        "langchain-anthropic>=0.1.0",
        "graph_db>=5.16.0",
        "vector_db-client>=1.7.0",
        "pydantic>=2.5.0",
        "pydantic-settings>=2.1.0",
        "python-dotenv>=1.0.0",
        "httpx>=0.26.0",
        "tenacity>=8.2.3",
        "email-validator>=2.1.0",
        "click>=8.1.7",
        "rich>=13.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "black>=23.12.1",
            "ruff>=0.1.9",
            "mypy>=1.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "bamboo=bamboo.cli:cli",
            "bamboo-populate=bamboo.scripts.populate_knowledge:main",
            "bamboo-analyze=bamboo.scripts.analyze_task:main",
        ],
    },
)

