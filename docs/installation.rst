Installation
------------

Installation and requirements
A working ``Python 3.5`` installation and the following libraries are required: 
``matplotlib``, ``numpy``, ``sklearn``, ``scipy``, `ot` and ``networkx``.
Having all dependencies available, ``novoSpaRc`` can be employed by cloning the 
repository, modifying the template ``reconstruct_tissue.py`` accordingly
and running it to perform the spatial reconstruction.

The code is partially based on adjustments of the `POT (Python Optimal Transport) <library https://github.com/rflamary/POT>`_.


``novoSpaRc`` requires a working ``Python 3.4`` installation.


Anaconda
~~~~~~~

To install ``novoSpaRc`` throuh Anaconda run::

    conda install -c conda-forge pot

Pull novoSpaRc from `PyPI <https://pypi.org/project/novosparc>`__ (consider
using ``pip3`` to access Python 3)::

    pip install novoSpaRc


PyPI only
~~~~~~~~

If you prefer to exlcusively use PyPI run::

    pip install novoSpaRc


Trouble shooting
~~~~~~~~~~~~~~~

If you do not have sudo rights (you get a ``Permission denied`` error)::

    pip install --user novosparc

If installation through ``pip`` fails try installing the ``pot`` library
first::

    pip install cython
    pip install pot

and then ``novoSpaRc``::

    pip install novosparc 
