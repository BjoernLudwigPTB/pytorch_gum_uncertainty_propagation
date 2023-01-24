# Changelog

<!--next-version-placeholder-->

## v0.18.0 (2023-01-24)
### Feature
* **plotting notebook:** Introduce notebook containing plots of the activation function ([`87876aa`](https://github.com/BjoernLudwigPTB/pytorch_gum_uncertainty_propagation/commit/87876aa1df50461f4527d371d001eb977a2fbe45))

### Documentation
* **INSTALL:** Improve install command for examples' dependencies ([`c39b141`](https://github.com/BjoernLudwigPTB/pytorch_gum_uncertainty_propagation/commit/c39b141868b1c381e3f008b26e1e29b76d082aef))

**[See all commits in this version](https://github.com/BjoernLudwigPTB/pytorch_gum_uncertainty_propagation/compare/v0.17.1...v0.18.0)**

## v0.17.1 (2023-01-21)
### Fix
* **zema_dataset:** Adapt to most recent zema_emc_annotated version v0.7.0 ([`7a7df38`](https://github.com/BjoernLudwigPTB/pytorch_gum_uncertainty_propagation/commit/7a7df3813908de4ccd9c4f8a7599d37480d9d61d))

### Documentation
* **examples:** Improve examples docstrings ([`e3331a2`](https://github.com/BjoernLudwigPTB/pytorch_gum_uncertainty_propagation/commit/e3331a2ccf5a15da067541ca6d0195198885848b))

**[See all commits in this version](https://github.com/BjoernLudwigPTB/pytorch_gum_uncertainty_propagation/compare/v0.17.0...v0.17.1)**

## v0.17.0 (2023-01-20)
### Feature
* **coverage:** Introduce CodeCov coverage report ([`12bdaab`](https://github.com/BjoernLudwigPTB/pytorch_gum_uncertainty_propagation/commit/12bdaabae7e4161674737b1254ebb852bde49999))
* **DOI:** Introduce DOI into CITATION and README ([`879da99`](https://github.com/BjoernLudwigPTB/pytorch_gum_uncertainty_propagation/commit/879da9907295d19257bb3245a158ebbff0b8534c))
* **uncertainties:** Turn is_symmetric and is_positive_semi_definite into public functions ([`00afcda`](https://github.com/BjoernLudwigPTB/pytorch_gum_uncertainty_propagation/commit/00afcda53a769c834f6067cce01beebfb96b5211))

**[See all commits in this version](https://github.com/BjoernLudwigPTB/pytorch_gum_uncertainty_propagation/compare/v0.16.0...v0.17.0)**

## v0.16.0 (2023-01-20)
### Feature
* **coverage:** Introduce coverage report upload into CI pipeline ([`84e9c97`](https://github.com/BjoernLudwigPTB/pytorch_gum_uncertainty_propagation/commit/84e9c9775eb2a542bdd340ca51011f5e2a0113c7))
* **ReadTheDocs:** Introduce settings and everything else to enable ReadTheDocs ([`a987651`](https://github.com/BjoernLudwigPTB/pytorch_gum_uncertainty_propagation/commit/a987651a3a41c8a1a1e97029dc974877936c7c05))
* **CITATION.cff:** Introduce Citation File Format-file and corresponding settings ([`fba4a38`](https://github.com/BjoernLudwigPTB/pytorch_gum_uncertainty_propagation/commit/fba4a38a305a97e52293e3f128d2164425907d97))

### Documentation
* **README:** Introduce link to ReadTheDocs and badge about documentation status ([`c884c76`](https://github.com/BjoernLudwigPTB/pytorch_gum_uncertainty_propagation/commit/c884c763305f06e5d0b857c9620ed930372c72ea))
* **README:** Introduce link to ReadTheDocs ([`70614d3`](https://github.com/BjoernLudwigPTB/pytorch_gum_uncertainty_propagation/commit/70614d3dede0f8ee3ec85c9aa3fe0f3f9265453c))
* **examples:** Introduce examples into docs ([`3f84e86`](https://github.com/BjoernLudwigPTB/pytorch_gum_uncertainty_propagation/commit/3f84e86a846b2be651cef0e87ba25c97569e798d))
* **INSTALL:** Introduce optional dependencies section ([`36b5327`](https://github.com/BjoernLudwigPTB/pytorch_gum_uncertainty_propagation/commit/36b53272f804812bfb6a899a4b739b19aa3ffa13))
* **README:** Introduce Disclaimer and License sections ([`ffe1111`](https://github.com/BjoernLudwigPTB/pytorch_gum_uncertainty_propagation/commit/ffe1111510a992157080c915ba9f2908efd40403))

**[See all commits in this version](https://github.com/BjoernLudwigPTB/pytorch_gum_uncertainty_propagation/compare/v0.15.0...v0.16.0)**

## v0.15.0 (2023-01-17)
### Feature
* **propagate:** Introduce current version of zema_emc_annotated and thus more flexibility ([`adcc4ec`](https://gitlab1.ptb.de/ludwig10_masters_thesis/gum-compliant_neural-network_uncertainty-propagation/-/commit/adcc4ec388424ac386758b4cfb4a22cacb81d193))

### Documentation
* **examples:** Adapt to new location of zema_dataset ([`e160b7c`](https://gitlab1.ptb.de/ludwig10_masters_thesis/gum-compliant_neural-network_uncertainty-propagation/-/commit/e160b7c85d558c22d6914a4074add1b401a7144e))
* **README:** Update roadmap ([`7b8606e`](https://gitlab1.ptb.de/ludwig10_masters_thesis/gum-compliant_neural-network_uncertainty-propagation/-/commit/7b8606ee70e1373b33ba320ac82da45763206797))
* **modules:** Introduce reference to Master's thesis into docstrings and docs ([`7ab7ad4`](https://gitlab1.ptb.de/ludwig10_masters_thesis/gum-compliant_neural-network_uncertainty-propagation/-/commit/7ab7ad473bfff6cf6aef94163eb06826a9490f5c))

## v0.14.0 (2022-12-21)
### Feature
* **zema_dataset:** Introduce local caching of extracted data ([`5614b91`](https://gitlab1.ptb.de/ludwig10_masters_thesis/gum-compliant_neural-network_uncertainty-propagation/-/commit/5614b915c67091fd7da5877d511b5f0f454b585c))

### Documentation
* **propagate:** Replace old module uncertain_quadlu_propagate after refactoring by propagate ([`d41ae32`](https://gitlab1.ptb.de/ludwig10_masters_thesis/gum-compliant_neural-network_uncertainty-propagation/-/commit/d41ae3201f3bb806663ebfa3f1b1e0a1d454cab4))

## v0.13.0 (2022-12-21)
### Feature
* **GUMSigmoidMLP:** Introduce new mult-layer perceptron exclusively equipped with GUMSigmoid ([`ea7916a`](https://gitlab1.ptb.de/ludwig10_masters_thesis/gum-compliant_neural-network_uncertainty-propagation/-/commit/ea7916a7993a5dc86fa07b3182be6b49935f94d1))
* **GUMSigmoid:** Introduce GUMSigmoid activation function ([`5ebd198`](https://gitlab1.ptb.de/ludwig10_masters_thesis/gum-compliant_neural-network_uncertainty-propagation/-/commit/5ebd1982bee6e5d3bd6e2e8dc4a72ad2a8428742))

### Documentation
* **cov_matrix_from_std_uncertainties:** Fix docstrings and insert reference ([`fdae1e3`](https://gitlab1.ptb.de/ludwig10_masters_thesis/gum-compliant_neural-network_uncertainty-propagation/-/commit/fdae1e3969f63c0cd6252fb9213af758c780a0ac))
* **GUMSoftplus:** Fix some formulae ([`e897a5f`](https://gitlab1.ptb.de/ludwig10_masters_thesis/gum-compliant_neural-network_uncertainty-propagation/-/commit/e897a5f1db4a36d8f4ae12b31f5db042d3180ee1))

## v0.12.0 (2022-12-21)
### Feature
* **GUMSoftplusMLP:** Introduce beta and threshold parameters ([`3ee9314`](https://gitlab1.ptb.de/ludwig10_masters_thesis/gum-compliant_neural-network_uncertainty-propagation/-/commit/3ee9314d6864dc2b162095bfdcfb5377752ef793))
* **MLP:** Introduce args and kwargs for setting parameters of the activation module ([`530c47d`](https://gitlab1.ptb.de/ludwig10_masters_thesis/gum-compliant_neural-network_uncertainty-propagation/-/commit/530c47d74a42c8ddfb2917daca878b486bf51688))
* **modules:** Introduce GUMSoftplus activation and GUMSoftplusMLP ([`0815e66`](https://gitlab1.ptb.de/ludwig10_masters_thesis/gum-compliant_neural-network_uncertainty-propagation/-/commit/0815e6610ee11cd3e95afdfbe1564099356f350c))
* **MLP:** Introduce generic MLP class ([`d942935`](https://gitlab1.ptb.de/ludwig10_masters_thesis/gum-compliant_neural-network_uncertainty-propagation/-/commit/d942935f3e067d5a4c8adc9023459aeb0fbef341))

### Fix
* **GUMQuadLU:** Fix propagation of uncertainties ([`5909770`](https://gitlab1.ptb.de/ludwig10_masters_thesis/gum-compliant_neural-network_uncertainty-propagation/-/commit/5909770bc383f4fc0203182babc0eab840c59db1))

### Documentation
* **GUMSoftplus:** Introduce beta and threshold into docstring ([`2a6d999`](https://gitlab1.ptb.de/ludwig10_masters_thesis/gum-compliant_neural-network_uncertainty-propagation/-/commit/2a6d9994a51a1e163e80d6ecadeb719f45c000d1))
* **GUMQuadLU:** Mention alphas default value ([`8492974`](https://gitlab1.ptb.de/ludwig10_masters_thesis/gum-compliant_neural-network_uncertainty-propagation/-/commit/8492974bb5ec46017e912aa687ca6aec6bbf30cf))

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
