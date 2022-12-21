# Changelog

<!--next-version-placeholder-->

## v0.11.0 (2022-12-21)
### Feature
* **uncertain_quadlu_propagate:** Introduce profiler into example ([`6659e70`](https://gitlab1.ptb.de/ludwig10_masters_thesis/gum-compliant_neural-network_uncertainty-propagation/-/commit/6659e7067e4788c2851a06ebed0e4c21004cbe70))
* **uncertain_quadlu:** Introduce profiler into forward ([`7406b67`](https://gitlab1.ptb.de/ludwig10_masters_thesis/gum-compliant_neural-network_uncertainty-propagation/-/commit/7406b67dd61d1a24e6e5883dcc78eab4bd44dddc))
* **uncertain_linear:** Introduce profiler into forward ([`a502cde`](https://gitlab1.ptb.de/ludwig10_masters_thesis/gum-compliant_neural-network_uncertainty-propagation/-/commit/a502cdec874afe45800a0e46ab155022eb77dc5f))

### Fix
* **assemble_pipeline:** Enable profiler that has been deactivated accidentally ([`40ae9ed`](https://gitlab1.ptb.de/ludwig10_masters_thesis/gum-compliant_neural-network_uncertainty-propagation/-/commit/40ae9ed9b756affde4d6d35f0c8fca7ab9caca44))

## v0.10.0 (2022-12-20)
### Feature
* **uncertain_quadlu_propagate:** Introduce actual propagation for QuadLUMLP ([`d72d9bd`](https://gitlab1.ptb.de/ludwig10_masters_thesis/gum-compliant_neural-network_uncertainty-propagation/-/commit/d72d9bdbddd4e9b6595ab586a3a8ed09b6576be8))
* **zema_dataset:** Introduce dataset extraction function ([`b207e87`](https://gitlab1.ptb.de/ludwig10_masters_thesis/gum-compliant_neural-network_uncertainty-propagation/-/commit/b207e87c40937ee5d5f733afa21da4f2f5bce82a))

### Fix
* **docs:** Fix accidentally removing ZeMA dataset content and really introduce QuadLUMLP propagation ([`ae20ecf`](https://gitlab1.ptb.de/ludwig10_masters_thesis/gum-compliant_neural-network_uncertainty-propagation/-/commit/ae20ecf32e7bb7bdd194ca315db1aad4e175a3cc))

### Documentation
* **uncertain_quadlu_propagate:** Introduce QuadLUMLP propagation into docs ([`db145b5`](https://gitlab1.ptb.de/ludwig10_masters_thesis/gum-compliant_neural-network_uncertainty-propagation/-/commit/db145b5cf55dc46473ba3a694faa7f812c06a4cc))
* **zema_dataset:** Introduce ZeMA dataset extraction function into docs ([`9934dea`](https://gitlab1.ptb.de/ludwig10_masters_thesis/gum-compliant_neural-network_uncertainty-propagation/-/commit/9934dea417833a94a07a72a476e436184386ca18))

### Performance
* **_is_positive_semi_definite:** Improve implementation to catch more edge cases ([`9591265`](https://gitlab1.ptb.de/ludwig10_masters_thesis/gum-compliant_neural-network_uncertainty-propagation/-/commit/9591265ef3b4ed9f5328ecf60f216b840a6005b8))

## v0.9.0 (2022-12-17)
### Feature
* **read_dataset:** Introduce full pipeline to extract ZeMA hdf5 into numpy array ([`4a275d0`](https://gitlab1.ptb.de/ludwig10_masters_thesis/gum-compliant_neural-network_uncertainty-propagation/-/commit/4a275d0bbb30de38db45b08ce0849b46f51cb171))

## v0.8.0 (2022-12-17)
### Feature
* **UncertainQuadLUMLP:** Introduce first implementation of an MLP equipped with UncertainQuadLU ([`d17d333`](https://gitlab1.ptb.de/ludwig10_masters_thesis/gum-compliant_neural-network_uncertainty-propagation/-/commit/d17d333a1021613b7db28b383c1cbb6e4044667f))
* **ValuesUncertainties:** Introduce new namedtuple datatype for values with associated uncertainty ([`b68e7fc`](https://gitlab1.ptb.de/ludwig10_masters_thesis/gum-compliant_neural-network_uncertainty-propagation/-/commit/b68e7fc36d9a1cabee6f99ea7ef9a283461b688c))

### Documentation
* **uncertainties:** Introduce uncertainties module into docs ([`2cdf789`](https://gitlab1.ptb.de/ludwig10_masters_thesis/gum-compliant_neural-network_uncertainty-propagation/-/commit/2cdf78922f23a9db7a45799a047d336569392859))

## v0.7.0 (2022-12-15)
### Feature
* **UncertainLinear:** Introduce Linear layer with uncertainty propagation ([`11ebf1e`](https://gitlab1.ptb.de/ludwig10_masters_thesis/gum-compliant_neural-network_uncertainty-propagation/-/commit/11ebf1e554e199493efe6856a243067769ddf685))

### Documentation
* **UncertainQuadLU:** Introduce _alpha's docstring ([`1b4c11a`](https://gitlab1.ptb.de/ludwig10_masters_thesis/gum-compliant_neural-network_uncertainty-propagation/-/commit/1b4c11a79c16b922a9b328fb72d17f70bdd0720f))

## v0.6.0 (2022-12-14)
### Feature
* **UncertainQuadLU:** Finalize implementation of UncertainQuadLU ([`76092a5`](https://gitlab1.ptb.de/ludwig10_masters_thesis/gum-compliant_neural-network_uncertainty-propagation/-/commit/76092a511600366b73539d62ee4b37b2ca96cf78))

### Fix
* **UncertainQuadLU:** Fix computation of propagated uncertainties ([`83c1b28`](https://gitlab1.ptb.de/ludwig10_masters_thesis/gum-compliant_neural-network_uncertainty-propagation/-/commit/83c1b28fdcd3aab2efb0ec60b56a120c9af8c8c1))

## v0.5.0 (2022-12-11)
### Feature
* **UncertainQuadLU:** Make UncertainQuadLU inplace-able ([`1f43385`](https://gitlab1.ptb.de/ludwig10_masters_thesis/gum-compliant_neural-network_uncertainty-propagation/-/commit/1f433853aef0f5445e0537a09f8527c4e2c55c86))

## v0.4.0 (2022-12-11)
### Feature
* **QuadLUMLP:** Introduce multi-layer perceptron with QuadLU activation ([`1122ad7`](https://gitlab1.ptb.de/ludwig10_masters_thesis/gum-compliant_neural-network_uncertainty-propagation/-/commit/1122ad764ea9132563ad8afb9ee5ce2d2369a07c))
* **quadlu:** Make quadlu inplace-able ([`92e0a73`](https://gitlab1.ptb.de/ludwig10_masters_thesis/gum-compliant_neural-network_uncertainty-propagation/-/commit/92e0a73ad5573501b3fa230f9bb76e719425e3f7))

### Documentation
* **README:** Introduce badges ([`ac26493`](https://gitlab1.ptb.de/ludwig10_masters_thesis/gum-compliant_neural-network_uncertainty-propagation/-/commit/ac264937c40fd25ad09342cc71b2846ad53dd05a))

## v0.3.0 (2022-12-10)
### Feature
* **quadlu:** Introduce functional version of QuadLU ([`b91dd3c`](https://gitlab1.ptb.de/ludwig10_masters_thesis/gum-compliant_neural-network_uncertainty-propagation/-/commit/b91dd3c06d8a0b999375611f52541bad35cfe9c6))

### Documentation
* **functionals:** Introduce functional version of QuadLU into docs ([`8ece184`](https://gitlab1.ptb.de/ludwig10_masters_thesis/gum-compliant_neural-network_uncertainty-propagation/-/commit/8ece184ea67179c3416dae87184aaa3e6f3d9b08))
* **Examples:** Introduce read_dataset into docs ([`49f752c`](https://gitlab1.ptb.de/ludwig10_masters_thesis/gum-compliant_neural-network_uncertainty-propagation/-/commit/49f752cd451f49f2b97127157c8c195b8539801f))
* **README:** Extend README with Documentation and Getting Started sections ([`f08fc19`](https://gitlab1.ptb.de/ludwig10_masters_thesis/gum-compliant_neural-network_uncertainty-propagation/-/commit/f08fc19ea39a3b48620dab1f74f645ff6f39fba0))

## v0.2.0 (2022-12-09)
### Feature
* **uncertainties:** Introduce module to process uncertainties with first function to construct cov ([`02e2859`](https://gitlab1.ptb.de/ludwig10_masters_thesis/gum-compliant_neural-network_uncertainty-propagation/-/commit/02e2859055c7f8be4a4d30c7edf1cf49b4b958b8))
* **QuadLU:** Introduce forward method to QuadLU and thus finish implementation ([`2f749ff`](https://gitlab1.ptb.de/ludwig10_masters_thesis/gum-compliant_neural-network_uncertainty-propagation/-/commit/2f749ff8e3e94788ad3e1773a3f2541348ff796a))

### Documentation
* **README:** Update roadmap ([`47460da`](https://gitlab1.ptb.de/ludwig10_masters_thesis/gum-compliant_neural-network_uncertainty-propagation/-/commit/47460dafc61d2a04e8070f9120836a6095dc0dff))
* **Sphinx:** Introduce sphinx docs ([`8d5dbe4`](https://gitlab1.ptb.de/ludwig10_masters_thesis/gum-compliant_neural-network_uncertainty-propagation/-/commit/8d5dbe43384dbc8e36cc705c26faa29a31b5070b))
* **INSTALL:** Introduce installation instructions ([`3daa8fb`](https://gitlab1.ptb.de/ludwig10_masters_thesis/gum-compliant_neural-network_uncertainty-propagation/-/commit/3daa8fba15d53d1cac07843ae179bf42d003c485))

## v0.1.0 (2022-12-03)
### Feature
* **mnist:** Introduce mnist example notebook ([`446c112`](https://github.com/ludwig10_masters_thesis/gum-compliant_neural-network_uncertainty-propagation/commit/446c112d7a01d5d5a394cf66c64a5c2ec914f5f2))

**[See all commits in this version](https://github.com/ludwig10_masters_thesis/gum-compliant_neural-network_uncertainty-propagation/compare/v0.0.0...v0.1.0)**
