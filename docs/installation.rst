Installation
------------

A working ``Python 3.5`` installation and the following libraries are required: 
``matplotlib``, ``numpy``, ``sklearn``, ``scipy``, ``ot`` and ``networkx``.

The code is partially based on adjustments of the `POT (Python Optimal Transport) <https://github.com/rflamary/POT>`_ library.


``novoSpaRc`` requires a working ``Python 3.4`` installation.


PyPI
~~~~

To install ``novoSpaRc`` try::

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
