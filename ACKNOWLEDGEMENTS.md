# Acknowledgements

## Open Source Software

pyQuil makes use open source software. Without it, this software could
not have been made. For a list of third-party software included with
this distribution, see the `NOTICE` file.

Below is a list of software used by this distrubition that we
acknowledge.

* pyQuil (this software), licensed under the Apache 2.0 License.
* [Python](https://www.python.org/)
* [NumPy](http://www.numpy.org/)
* [Requests](http://docs.python-requests.org/en/master/)

## Pre-Release pyQuil History and Credits

Initially named `pyqpl`, work on pyQuil started in June of 2016 at Rigetti Computing as a way to
make Quil easier to write and generate. The project grew out of an initial proof-of-concept by
Robert Smith (@tarballs-are-good), and was immediately and substantially grown into fuller project
by him, Will Zeng (@willzeng), and Spike Curtis (@spikecurtis).

The mathematical functionality of pyQuil started with an algebra module for manipulating Pauli
operators, which was contributed by Nick Rubin (@ncrubin). With this, he authored the first
larger-scale non-trivial algorithms using pyQuil, such as VQE and QAOA. The algorithms module has
since been released as a separate project called [Grove](https://github.com/rigetticomputing/grove).

After about 300 commits, the git history was removed for release.

We give special thanks to Anthony Polloreno (@ampolloreno), Peter Karalekas (@pkaralekas), Nikolas
Tezak (@ntezak), and Chris Osborn (@cbosborn) for their contributions, use, and rigorous testing of
pyQuil prior to its release.