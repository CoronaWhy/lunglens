## Build and install using nbdev:

Rebuild is required after each modification of library notebooks in `/src` folder:

`rm -rf lib-pkg/hlai/* && nbdev_build_lib`

Install once using `pip install -e .`. Install per each env if you are using conda's pip.

## How to setup the development environment:

### Use new conda env

`conda env create --file environment.yml --force -v`

#### Quick env update:

`conda env update --file environment.yml --prune`