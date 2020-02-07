Acknowledgements
================

Pre-Release History
-------------------

Initially named `pyqpl`, work on pyQuil started in June of 2016 at Rigetti Computing as a way to
make Quil easier to write and generate. The project grew out of an initial proof-of-concept by
[Robert Smith][rss], and was immediately and substantially grown into fuller project by him,
[Will Zeng][wjz], and [Spike Curtis][spike].

The mathematical functionality of pyQuil started with an algebra module for manipulating Pauli
operators, which was contributed by [Nick Rubin][ncr]. With this, he authored the first
larger-scale non-trivial algorithms using pyQuil, such as VQE and QAOA. The algorithms module has
since been released as a separate project called [Grove][grove].

After about 300 commits, the git history was removed for release.

We give special thanks to [Anthony Polloreno][amp], [Peter Karalekas][pjk], [Nikolas Tezak][nikt],
and [Chris Osborn][cbo] for their contributions, use, and rigorous testing of pyQuil prior to its
initial public release.

Pre-2.0 (QCS) History
---------------------

As part of the launch of [Quantum Cloud Services][qcs], there was an internally-maintained branch
of pyQuil for prototyping the new quantum programming interface optimized to support the variational
execution model. Once QCS was launched, and pyQuil 2.0 released alongside it, all the work on this
branch was squashed into a [single commit][qcs-commit]. However, this commit was the culmination of
months of work by many people, including [Matthew Harrigan][mph], [Eric Peterson][ecp],
[Nikolas Tezak][nikt], [Lauren Capelluto][lcaps], [Peter Karalekas][pjk], and [Robert Smith][rss].

Maintainers
-----------

Over the course of its history, pyQuil has had a few different maintainers, each of who were in
charge of pull-request review and the release process for an extended period of time. In reverse
chronological order, they are [Peter Karalekas][pjk], [Matthew Harrigan][mph],
[Steven Heidel][steveo], and [Will Zeng][wjz].

[amp]: https://github.com/ampolloreno
[cbo]: https://github.com/cbosborn
[ecp]: https://github.com/ecpeterson
[grove]: https://github.com/rigetti/grove
[lcaps]: https://github.com/lcapelluto
[mph]: https://github.com/mpharrigan
[ncr]: https://github.com/ncrubin
[nikt]: https://github.com/ntezak
[pjk]: https://github.com/karalekas
[qcs]: https://scirate.com/arxiv/2001.04449
[qcs-commit]: https://github.com/rigetti/pyquil/commit/a952c3df9afaf40b076734a9a37f93d09d3b399f
[rss]: https://github.com/stylewarning
[spike]: https://github.com/spikecurtis
[steveo]: https://github.com/stevenheidel
[wjz]: https://github.com/willzeng
