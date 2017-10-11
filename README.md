Description
==============



This package provides wrappers for using various persistent homology and other Topological Data Analysis software packages through python.


The code is an compilation of work done by [Elizabeth Munch](http://www.elizabethmunch.com/math) along with her students and collaborators.  People who have contributed to teaspoon include:


- [Firas Khasawneh](http://www.firaskhasawneh.com)
- Brian Bollen


Locations
==============


- **Code**: [http://gitlab.msu.edu/TSAwithTDA/teaspoon](https://gitlab.msu.edu/TSAwithTDA/teaspoon).

- **Documentation**: [http://elizabethmunch.com/math/teaspoon/index.html](http://elizabethmunch.com/math/teaspoon/index.html)


Requirements
==============

Please note that this code is still pre-alpha, so many things are not fully up and running yet.

Bugs reports and feature reqests can be posted on the [gitlab issues](https://gitlab.msu.edu/TSAwithTDA/teaspoon/issues) page.

In order to use all wrappers in teaspoon.TDA, the following need to be installed so they can be run command line. 

- [Ripser](https://github.com/Ripser/ripser). Code by Ulrich Bauer for computing persistent homology of a point cloud or distance matrix.
- [Perseus](http://people.maths.ox.ac.uk/nanda/perseus/index.html). Code by Vidit Nanda for computing persistent homology of point clouds, cubical complexes, and distance matrices. 
- [Hera](https://bitbucket.org/grey_narn/hera). Code by Michael Kerber, Dmitriy Morozov, and Arnur Nigmetov for computing bottleneck and Wasserstein distances.

<span style="color:red">Warning</span>: the teaspoon installation will not install nor check for the install of these packages. In particular, perseus should be installed so that typing `perseus ` runs it, even though some operating systems end up with an executable with a different name.


Installation
==============

### <span style="color:red">Warning</span>: 
This code is still pre-alpha.  In particular, using the pip install seems to be finicky at best.  
If you are having installation issues, please make a note of what you've done, including copying error message outputs, as a comment into the [installation issue on gitlab](https://gitlab.msu.edu/TSAwithTDA/teaspoon/issues/1) so we can start figuring out what is up with this system.



### Installing using pip:

As this code is still pre-alpha, your best bet is to cd into the folder containing teaspoon (should have setup.py there) and run

```{bash}
pip install .
```

If you want the developmental installation for pip run
```{bash}
pip install -e .
```
When things get more stable, we can remove the -e part.
If you already have teaspoon installed but want to update to a newer version, run
```{bash}
pip install -U .
```

### Installing using python:

According to the internet, the pip version of install appears to be better. However, if you don't use pip, another option is to cd into the teaspoon directory and run:

```{bash}
python setup.py develop
```

Again, this is the developer version of the installation.  Eventually, we will want to be doing 

```{bash}
python setup.py install
```

Documentation
=============

Documentation is done using [doxygen](http://www.doxygen.org).  Documentation can be found in the [doc folder](https://gitlab.msu.edu/TSAwithTDA/teaspoon/doc/html/index.html).  Further info can be found in the [contributing](https://gitlab.msu.edu/TSAwithTDA/teaspoon/blob/master/CONTRIBUTING.md) page.


Contributing
=============

See the [contributing](https://gitlab.msu.edu/TSAwithTDA/teaspoon/blob/master/CONTRIBUTING.md) page for more information on workflows.