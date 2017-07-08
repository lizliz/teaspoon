Description
==============



This package provides wrappers for using various persistent homology and other Topological Data Analysis software packages through python.

The code is an compilation of work done by [Elizabeth Munch](http://www.elizabethmunch.com/math) along with her students and collaborators.  People who have contributed to teaspoon include:


- [Firas Khasawneh](http://www.firaskhasawneh.com)
- Brian Bollen



Requirements
==============

Please note that this code is still pre-alpha, so many things are not fully up and running yet.

*TODO: Insert instructions on bug lists and feature requests*

In order to use all wrappers in teaspoon.TDA, the following need to be installed so they can be run command line. Note: the teaspoon installation will not install or check for their install.  

- [Ripser](https://github.com/Ripser/ripser). Code by Ulrich Bauer for computing persistent homology of a point cloud or distance matrix.
- [Perseus](http://people.maths.ox.ac.uk/nanda/perseus/index.html). Code by Vidit Nanda for computing persistent homology of point clouds, cubical complexes, and distance matrices.
- [Hera](https://bitbucket.org/grey_narn/hera). Code by Michael Kerber, Dmitriy Morozov, and Arnur Nigmetov for computing bottleneck and Wasserstein distances.



Installation
==============

### Warning:
This code is still pre-alpha.  In particular, using the pip install seems to be finicky at best.  *TODO: Need to find someone who knows more about pip install to fix this up.* 
** If you are having installation issues, please make a note of what you've done, including copying error message outputs, as a comment into the [installation issue on gitlab](https://gitlab.msu.edu/TSAwithTDA/teaspoon/issues/1) so we can start figuring out what is up with this system.**



### Installing using pip:

As this code is still pre-alpha, your best bet is to cd into the folder containing teaspoon (should have setup.py there) and run

```{python}
    pip install -e .
```

This is the developmental installion for pip.  When things get more stable, we can remove the -e part.

### Installing using python:

According to the internet, the pip version of install appears to be better. However, if you don't use pip, another option is to cd into the teaspoon directory and run:

```{bash} 
    python setup.py develop
```
    
Again, this is the developer version of the installation.  Eventually, we will want to be doing 

```{bash} 
    python setup.py install
```
    