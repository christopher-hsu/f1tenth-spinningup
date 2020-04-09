**Status:** Maintenance (expect bug fixes and minor updates)

Welcome to Spinning Up in Deep RL! 
==================================

This is an educational resource produced by OpenAI that makes it easier to learn about deep reinforcement learning (deep RL).

For the unfamiliar: [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning) (RL) is a machine learning approach for teaching agents how to solve tasks by trial and error. Deep RL refers to the combination of RL with [deep learning](http://ufldl.stanford.edu/tutorial/).

This module contains a variety of helpful resources, including:

- a short [introduction](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) to RL terminology, kinds of algorithms, and basic theory,
- an [essay](https://spinningup.openai.com/en/latest/spinningup/spinningup.html) about how to grow into an RL research role,
- a [curated list](https://spinningup.openai.com/en/latest/spinningup/keypapers.html) of important papers organized by topic,
- a well-documented [code repo](https://github.com/openai/spinningup) of short, standalone implementations of key algorithms,
- and a few [exercises](https://spinningup.openai.com/en/latest/spinningup/exercises.html) to serve as warm-ups.

Get started at [spinningup.openai.com](https://spinningup.openai.com)!



## Installation for f1tenth_gym(Native)
The environment officially supports Python3, Python2 might also work. You'll need several dependencies to run this environment:

### Eigen and protobuf dependencies:

```bash
$ sudo apt-get install -y libzmq3-dev build-essential autoconf libtool libeigen3-dev
$ sudo cp -r /usr/include/eigen3/Eigen /usr/include
```

### Protobuf:

```bash
$ cd f1tenth_gym
$ git clone https://github.com/google/protobuf.git
$ cd protobuf
$ ./autogen.sh
$ ./configure
$ make -j4
$ sudo make install
$ ldconfig
$ make clean
```

### Python packages:

```bash
$ pip3 install --user numpy scipy numba zmq pyzmq Pillow gym protobuf pyyaml msgpack==0.6.2
```

### To install the simulation environment natively, clone this repo (it is already installed).

```bash
$ git clone https://github.com/f1tenth/f1tenth_gym
```

### Then install the env via the following steps:
```bash
$ cd f1tenth_gym
$ mkdir build
$ cd build
$ cmake ..
$ make
$ cp sim_requests_pb2.py ../gym/
$ cd ..
$ pip3 install --user -e gym/
```






Citing Spinning Up
------------------

If you reference or use Spinning Up in your research, please cite:

```
@article{SpinningUp2018,
    author = {Achiam, Joshua},
    title = {{Spinning Up in Deep Reinforcement Learning}},
    year = {2018}
}
```
