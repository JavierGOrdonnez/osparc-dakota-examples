Dakota framework examples by the IT'IS Foundation
==========================================

This repository contains the example use cases for 
the [Dakota](https://github.com/snl-dakota/dakota) 
python module.

It uses a dakota python wheel built by the IT'IS Foundation:
[IT'IS-Dakota wheel](https://github.com/ITISFoundation/itis-dakota)

Running the example
-------------------

For now the code is only tested on Ubuntu(Linux) and WSL2(Windows). To run the code:
```
cd examples
make simple
```

If you are running on WSL2, you might have to install venv first: 
```
sudo apt install python3.10-venv
```
(with 3.10 replaced with the python version you are using)

Other Make targets are:
simple, simple_restart, simple_batch, adasampling, moga

Authors: wvangeit@it'is

Copyright (c) 2024 IT'IS Foundation, Zurich, Switzerland
