from setuptools import setup, find_packages

setup(
    name="dr_classification",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.12.0",
        "numpy>=1.22.0",
        "pandas>=1.5.0",
        "matplotlib>=3.5.0",
        "scikit-learn>=1.0.0",
        "opencv-python>=4.5.0",
        "pillow>=9.0.0",
        "seaborn>=0.12.0",
        "tqdm>=4.64.0",
        "h5py>=3.7.0",
        "albumentations>=1.3.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Diabetic Retinopathy Classification using Deep Learning",
    keywords="deep learning, medical imaging, diabetic retinopathy",
    python_requires=">=3.8",
)