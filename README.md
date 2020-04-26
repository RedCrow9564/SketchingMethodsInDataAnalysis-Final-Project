# Sketching Methods for Analysis of Matrices and Data (0372-4004) - Final Project
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RedCrow9564/SketchingMethodsInDataAnalysis-Final-Project/blob/master/Sketching_Methods_Final_Project.ipynb) 
[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)
![Run Unit-Tests](https://github.com/RedCrow9564/SketchingMethodsInDataAnalysis-Final-Project/workflows/Run%20Unit-Tests/badge.svg?branch=master)
![Compute Code Metrics](https://github.com/RedCrow9564/SketchingMethodsInDataAnalysis-Final-Project/workflows/Compute%20Code%20Metrics/badge.svg?branch=master)
![GitHub last commit](https://img.shields.io/github/last-commit/RedCrow9564/SketchingMethodsInDataAnalysis-Final-Project)

This is a project submitted as a requirement for this course. [The course](https://www30.tau.ac.il/yedion/syllabus.asp?course=0372400401) was administered in Fall 2019-2020 (before the Coronavirus outbreak...) in [Tel-Aviv University - School of Mathematical Sciences](https://en-exact-sciences.tau.ac.il/math), and taught by [Prof. Haim Avron](https://english.tau.ac.il/profile/haimav). 
This project is a reconstruction of experiments of [[1]](#1) about an algorithm for a faster computation least-square 
solutions accurately. A complete documentation of the code is available [here](doc/doc-html/documentation_homepage.html)(open it in a web browser).

## Getting Started
The code can be fetched from [this repo](https://github.com/RedCrow9564/SketchingMethodsInDataAnalysis-Final-Project). 
The Jupyter Notebook does the same work, and can be deployed to [Google Colab](https://colab.research.google.com/github/RedCrow9564/SketchingMethodsInDataAnalysis-Final-Project/blob/master/Sketching_Methods_Final_Project.ipynb). 
While the the notebook version can be used immediately, this code has some prerequisites.
Any questions about this project may be sent by mail to 'eladeatah' at mail.tau.ac.il (replace 'at' by @).

### Prerequisites

This code was developed and tested using the following Python 3.7 dependencies. These dependencies are listed in [requirements.txt](requirements.txt).
All these packages can be installed using the 'pip' package manager (when the command window is in the main directory where requirements.txt is located):
```
pip install -r requirements.txt
```
All the packages, except for Sacred, are available as well using 'conda' package manager.

## Running the tests

The Unit-Test files are:

* [test_caratheodory_set.py](UnitTests/test_caratheodory_set.py) - Tests the Caratheodory booster method.
* [test_coreset_methods.py](UnitTests/test_coreset_methods.py) - Tests the method which produces the coreset 
for a given matrix.

Running any of these tests can be performed by:
```
<python_path> -m unittest <test_file_path>
```
## Acknowledgments
Credits for the original algorithms, paper and results of [[1]](#1) belong to its respectful authors: Alaa Maalouf, 
Ibrahim Jubran and Dr. Dan Feldman. The [following repo](https://github.com/ibramjub/Fast-and-Accurate-Least-Mean-Squares-Solvers) 
contains the original code of these researchers which produced the original results.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## References
<a id="1">[1]</a> [Fast and Accurate Least-Mean-Squares Solvers. 
Maalouf, Jubran and Feldman (NIPS 2019)](https://papers.nips.cc/paper/9040-fast-and-accurate-least-mean-squares-solvers.pdf).
