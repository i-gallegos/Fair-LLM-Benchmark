from setuptools import setup

setup(
    name="bias-bench",
    version="0.1.0",
    description="An Empirical Survey of the Effectiveness of Debiasing Techniques for Pre-trained Language Models",
    url="https://github.com/mcgill-nlp/bias-bench",
    packages=["bias_bench"],
    install_requires=[
        "torch==1.10.2",
        "transformers==4.16.2",
        "scipy==1.7.3",
        "scikit-learn==1.0.2",
        "nltk==3.7.0",
        "datasets==1.18.3",
        "accelerate==0.5.1",
    ],
    include_package_data=True,
    zip_safe=False,
)
