## How to setup the development environment:

### Use new conda env

`conda env create --file environment.yml --force -v`

#### Quick env update:

`conda env update --file environment.yml --prune`

## Build and install lunglens library using nbdev:

### Rebuild
Rebuild is required after each modification of library notebooks in `/lib-src` folder:

`nbdev_build_lib` 

if you have issues with imprting lunglens modules after your modifications you can try following command:

`rm -rf lib-pkg/lunglens/* && nbdev_build_lib`

### Install
Install once using `pip install -e .` from your conda environment

