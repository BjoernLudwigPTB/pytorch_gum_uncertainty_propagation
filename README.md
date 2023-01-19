# GUM-compliant_neural-network_uncertainty-propagation

[![pipeline status](https://gitlab1.ptb.de/ludwig10_masters_thesis/gum-compliant_neural-network_uncertainty-propagation/badges/main/pipeline.svg)](https://gitlab1.ptb.de/ludwig10_masters_thesis/gum-compliant_neural-network_uncertainty-propagation/-/commits/main)
[![coverage report](https://gitlab1.ptb.de/ludwig10_masters_thesis/gum-compliant_neural-network_uncertainty-propagation/badges/main/coverage.svg)](https://gitlab1.ptb.de/ludwig10_masters_thesis/gum-compliant_neural-network_uncertainty-propagation/-/commits/main)
[![Latest Release](https://gitlab1.ptb.de/ludwig10_masters_thesis/gum-compliant_neural-network_uncertainty-propagation/-/badges/release.svg)](https://gitlab1.ptb.de/ludwig10_masters_thesis/gum-compliant_neural-network_uncertainty-propagation/-/releases)

This is the code written in conjunction with the first part of my Master's thesis on 
GUM-compliant neural network robustness verification. The code was written for 
_Python 3.10_.

The final submission date is 23. January 2023. Until then, this code base will be 
subject to constant change.

## Getting started

The [INSTALL guide](INSTALL.md) assists in installing the required packages.

## Documentation

To locally build the HTML or pdf documentation first the required dependencies need 
to be installed into your virtual environment (check the [INSTALL guide](INSTALL.md) 
first and upon completion execute the following):

```shell
(venv) $ python -m piptools sync docs-requirements.txt
(venv) $ sphinx-build docs/ docs/_build
sphinx-build docs/ docs/_build
Running Sphinx v5.3.0
loading pickled environment... done
[...]
The HTML pages are in docs/_build.
```

After that the documentation can be viewed by opening the file
_docs/\_build/index.html_ in any browser.

## Roadmap

- host documentation on ReadTheDocs, now that we are on GitHub
- introduce binder for ease of access to the notebooks
- replace ZeMA module by the external package and remove it here

## Disclaimer

This software is developed under the sole responsibility of [Björn
Ludwig](https://github.com/BjoernLudwigPTB) (the author in the following). The 
software is made available "as is" free of cost. The author assumes no 
responsibility whatsoever for its use by other parties, and makes no guarantees, 
expressed or implied, about its quality, reliability, safety, suitability or any 
other characteristic. In no event will the author be liable for any direct, indirect or 
consequential damage arising in connection with the use of this software.

## License

pytorch_gum_uncertainty_propagation is distributed under the [MIT
license](https://github.com/BjoernLudwigPTB/pytorch_gum_uncertainty_propagation/blob/main/LICENSE).