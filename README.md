## Build and install using nbdev:

Rebuild is required after each modification of library notebooks in `/src` folder:

`rm -rf lib-pkg/hlai/* && nbdev_build_lib`

Install once using `pip install -e .`. Install per each env if you are using conda's pip.
