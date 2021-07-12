Contributing to pyQuil
======================

Welcome to pyQuil, and thanks for wanting to be a contributor! 🎉

This guide is to help walk you through how to open issues and pull requests for the pyQuil
project, as well as share some general how-tos for development, testing, and maintenance.

If all you want to do is ask a question, you should do so in our
[Rigetti Forest Slack Workspace][slack-invite] rather than opening an issue. Otherwise,
read on to learn more!

This project and everyone participating in it are governed by pyQuil's
[Code of Conduct](.github/CODE_OF_CONDUCT.md). In contributing, you are expected
to uphold this code. Please report unacceptable behavior by contacting support@rigetti.com.

Table of Contents
-----------------

[Ways to Contribute](#ways-to-contribute)

- [Reporting a Bug](#reporting-a-bug)

- [Requesting an Enhancement](#requesting-an-enhancement)

- [Choosing an Issue to Address](#choosing-an-issue-to-address)

- [Making a Pull Request](#making-a-pull-request)

- [Adding a Tutorial Notebook](#adding-a-tutorial-notebook)

[Developer How-Tos](#developer-how-tos)

- [Install Dependencies](#install-dependencies)

- [Style Guidelines](#style-guidelines)

- [Running the Tests](#running-the-tests)

- [Building the Docs](#building-the-docs)

- [Working with the Parser](#working-with-the-parser)

- [Using the Docker Image](#using-the-docker-image)

[Tips for Maintainers](#tips-for-maintainers)

- [Merging a Pull Request](#merging-a-pull-request)

- [Managing the CI Pipelines](#managing-the-ci-pipelines)

- [Release Process](#release-process)

- [Issue and PR Labels](#issue-and-pr-labels)

Ways to Contribute
------------------

### Reporting a Bug

If you've encountered an error or unexpected behavior when using pyQuil, please file a
[bug report](https://github.com/rigetti/pyquil/issues/new?assignees=&labels=bug+%3Abug%3A&template=BUG_REPORT.md&title=).
Make sure to fill out the sections that allow us to reproduce the issue and understand
the context of your development environment. We welcome the opportunity to improve pyQuil,
so don't be shy if you think you've found a problem!

### Requesting an Enhancement

If you have an idea for a new addition to pyQuil, please let us know by creating a
[feature request](https://github.com/rigetti/pyquil/issues/new?assignees=&labels=enhancement+%3Asparkles%3A&template=FEATURE_REQUEST.md&title=).
The more information you can provide, the easier it will be for the pyQuil developers
to implement! A clear description of the problem being addressed, a potential solution,
and any alternatives you've considered are all great things to include.

### Choosing an Issue to Address

Rather than opening an issue, if you'd like to work on one that currently exists, we
have some issue labels that make it easy to figure out where to start. The
[good first issue][good-first-issue-label] label references issues that we think a
newcomer wouldn't have too much trouble taking on. In addition, the
[help wanted][help-wanted-label] label is for issues that the team would like to
see completed, but that we don't currently have the bandwidth for.

### Making a Pull Request

Once you've selected an issue to tackle, 
[forked the repository](https://github.com/rigetti/pyquil/fork), and made your changes,
the next step is to [open a pull request](https://github.com/rigetti/pyquil/compare)!
We've made opening one easy by providing a [Pull Request Template](.github/PULL_REQUEST_TEMPLATE.md)
that includes a checklist of things to complete before asking for code review. We look
forward to reviewing your work! 🙂

### Adding a Tutorial Notebook

You may have noticed that the `examples` directory has been removed from pyQuil, and a
"launch binder" badge was added to the README. We decided to move all the example notebooks
into a separate repository, [rigetti/forest-tutorials][forest-tutorials], so that they could
be run on [Binder][mybinder], which provides a web-based setup-free execution environment
for [Jupyter][jupyter] notebooks. We're always looking for new tutorials to help people
learn about quantum programming, so if you'd like to contribute one, make a pull request
to that repository directly!

Developer How-Tos
-----------------

### Install Dependencies

Before running any of the below commands, you'll need to install [Poetry][poetry] and run the
following from the top-level directory of this repo:

```bash
poetry install
```

[poetry]: https://python-poetry.org

### Style Guidelines

We use [Black](https://black.readthedocs.io/en/stable/index.html) and `flake8` to automatically
lint the code and enforce style requirements as part of the CI pipeline. You can run these style
tests yourself locally by running `make check-style` (to check for violations of the `flake8` rules)
and `make check-format` (to see if `black` would reformat the code) in the top-level directory of
the repository. If you aren't presented with any errors, then that means your code is good enough
for the linter (`flake8`) and formatter (`black`). If `make check-format` fails, it will present
you with a diff, which you can resolve by running `make format`. Black is very opinionated, but
saves a lot of time by removing the need for style nitpicks in PR review. We only deviate from its
default behavior in one category: we choose to use a line length of 100 rather than the Black
default of 88 (this is configured in the [`pyproject.toml`](pyproject.toml) file). As for `flake8`,
we ignore a couple of its rules (all for good reasons), and the specific configuration can be
found in the [`.flake8`](.flake8) file.

In addition to linting and formatting, we use type hints for all parameters and return values,
following the [PEP 484 syntax][pep-484]. This is enforced as part of the CI via the command
`make check-types`, which uses the popular static typechecker [mypy](http://mypy-lang.org/).
For more information on the specific configuration of `mypy` that we use for typechecking, please
refer to the [`mypy.ini`](mypy.ini) file. Also, because we use the `typing` module, types (e.g.
`type` and `rtype` entries) should be omitted when writing (useful) [Sphinx-style][sphinx]
docstrings for classes, methods, and functions.

All of these style-related tests can be performed locally with a single command, by running the
following:

```bash
make check-all
```

### Running the Tests

We use `pytest` to run the pyQuil unit tests. These are run automatically on Python 3.7 and
3.8 as part of the CI pipeline, but you can run them yourself locally as well. Many of the
tests depend on having running QVM and quilc servers. To start them, run each of the following
in a separate terminal window:

```bash
docker run --rm -it -p 5555:5555 rigetti/quilc -R -P
```

```bash
docker run --rm -it -p 5000:5000 rigetti/qvm -S
```

> **Note:** The above commands require [Docker][docker], but you can also download the QVM and
> quilc as part of the [Forest SDK][forest], and run them directly with `qvm -S` and
> `quilc -R -P`, respectively.

Once the QVM and quilc servers are running, you can run all the unit/integration tests with:

```bash
make test
```

To skip [slow tests](#slow-tests), you may run:

```bash
make test-fast
```

You can run end-to-end tests with:

```bash
make e2e TEST_QUANTUM_PROCESSOR=<quantum processor ID>
```

Or you may run all tests (unit/integration/e2e) with:

```bash
make test-all TEST_QUANTUM_PROCESSOR=<quantum processor ID>
```

> **Note:** for `TEST_QUANTUM_PROCESSOR`, supply a value similar to what you would supply to
> `get_qc()`. End-to-end tests are most useful against a real QPU, but they can also be run
> against a QVM (e.g. `make e2e TEST_QUANTUM_PROCESSOR=2q-qvm`).

#### Slow Tests

Some tests (particularly those related to operator estimation and readout symmetrization)
require a nontrivial amount of computation. For this reason, they have been marked
as slow and are not run unless `pytest` is given the `--runslow` option, which is defined
in the [`conftest.py`](test/unit/conftest.py) file.

For a full, up-to-date list of these slow tests, you may invoke (from the top-level directory):

```bash
grep -A 1 -r pytest.mark.slow  test/unit/
```

#### Seeded Tests

When making considerable changes to `operator_estimation.py`, we recommend that you set the
`pytest` option `--use-seed` (as defined in [`conftest.py`](test/unit/conftest.py)) to `False`
to make sure you have not broken anything. Thus, the command is:

```bash
pytest --use-seed=False <path/to/test-file-or-dir>
```

#### Code Coverage

In addition to testing the source code for correctness, we use `pytest` and the `pytest-cov`
plugin to calculate code coverage (via the `make test` command).

The coverage report omits the autogenerated parser code, the `external` module, and all of
the test code (as is specified in the [`.coveragerc`](.coveragerc) configuration file).

#### Summary

All of the above `pytest` variations can be mixed and matched according to what you're
trying to accomplish. For example, if you want to carefully test the operator estimation
code, run all of the slow tests, and also calculate code coverage, you could run:

```bash
pytest --cov=pyquil --use-seed=False --runslow <path/to/test-file-or-dir>
```

### Building the Docs

The [pyQuil docs](https://pyquil.readthedocs.io) build automatically as part of the CI pipeline.
However, you can also build them locally to make sure that everything renders correctly. We use
[Sphinx](http://www.sphinx-doc.org/en/master/) to build the documentation, and
then host it on [Read the Docs](https://readthedocs.org/) (RTD).

Before you can build the docs locally, you must make sure to install `pandoc` via your favorite
OS-level package manager (e.g. `brew`, `apt`, `yum`) in order to convert the [Changelog](CHANGELOG.md)
into reStructuredText (RST). Once you have done this, run the following from the top-level directory
of this repo:

```bash
make docs
```

If the build is successful, then you can navigate to the newly-created `docs/build`
directory and open the `index.html` file in your browser (`open index.html` works on macOS,
for example). You can then click around the docs just as if they were hosted on RTD, and
verify that everything looks right!

### Working with the Parser

The parser is implemented with Lark. See the [parser README](pyquil/_parser/README.md).

### Using the Docker Image

Rather than having a user go through the effort of setting up their local Forest
environment (a Python virtual environment with pyQuil installed, along with quilc
and qvm servers running), the Forest [Docker](https://www.docker.com/) image gives a
convenient way to quickly get started with quantum programming. This is not a wholesale
replacement for locally installing the Forest SDK, as Docker containers are ephemeral
filesystems, and therefore are not the best solution when the data they produce need
to be persisted.

The [`rigetti/forest`][docker-forest] Docker image is built
and pushed to DockerHub automatically as part of the CI pipeline. Developers can also
build the image locally by running `make docker` from the top-level directory. This
creates an image tagged by a shortened version of the current git commit hash (run
`docker images` to see all local images). To then start a container from this image, run:

```bash
docker run -it rigetti/forest:<commit hash>
```

Where `<commit hash>` is replaced by the actual git commit hash. This will drop you into an
`ipython` REPL with pyQuil installed and `quilc` / `qvm` servers running in the background.
Exiting the REPL (via `C-d`) will additionally shut down the Docker container and return
you to the shell that ran the image. Docker images typically only have one running process,
but we leverage an [`entrypoint.sh`](entrypoint.sh) script to initialize the Forest SDK
runtime when the container starts up.

The image is defined by its [Dockerfile](Dockerfile), along with a [`.dockerignore`](.dockerignore)
to indicate which files to omit when building the image. It is additionally important to
note that this image depends on a collection of parent images, pinned to specific versions.
This pinning ensures reproducibility, but requires that these versions be updated manually
as necessary. The section of the Dockerfile that would need to be edited looks something like
this:

```dockerfile
ARG quilc_version=1.20.0
ARG qvm_version=1.17.1
ARG python_version=3.7
```

Once a version has been changed, committed, and pushed, the CI will then use that new
version in all builds going forward.

Tips for Maintainers
--------------------

### Merging a Pull Request

When merging PRs, we have a couple of guidelines:

1. Double-check that the PR author has completed everything in the PR checklist that
   is applicable to the changes.

2. Always use the "squash and merge" option so that every PR corresponds to one commit.
   This keeps the git history clean and encourages many small (quickly reviewable) PRs
   rather than behemoth ones with lots of commits.

3. When pressing the merge button, each commit message will be turned into a bullet point
   below the title of the issue. Make sure to truncate the PR title to ~50 characters
   (unless completely impossible) so it fits on one line in the commit history, and delete
   any spurious bullet points that add no meaningful content. Also make sure the final commit
   message is prefixed with an appropriate Semantic Release prefix, so that it appears
   correctly in release notes.

### Managing the CI Pipelines

The CI/CD pipelines that underpin pyQuil are critical for supporting the job of its maintainer.
They validate formatting, style, correctness, and good code practice, and also build and
distribute the repository via [PyPI][pypi] and [DockerHub][docker-forest], all with minimal human
intervention. These pipelines almost always work as expected, but every now and then something goes
wrong, and it requires a deeper dive.

We use a collection of services for CI/CD -- [GitLab CI][gitlab-ci] and [GitHub Actions][gha] (GHA).

The configuration for GitLab CI is contained in the [`.gitlab-ci.yml`](.gitlab-ci.yml), and the
GHA configuration is in the [`.github/workflows`](.github/workflows) directory. GHA is responsible
for running checks and tests for PRs, while GitLab is responsible for additional tasks that require
access to resources that are not available publicly. This includes publishing docs, publishing to PyPI,
publishing Docker images, and running end-to-end tests on real QPUs.

### Release Process

Release is now automated through the use of Semantic Release. All development must occur on the `rc`
branch. For each push to this branch, a release candidate will be published to [GitHub][gh-releases],
[PyPI][pypi], and [DockerHub][docker-forest] (with image tags `rc` and `<version>`). When a final
release is desired, the `rc` branch should be merged to `master`. This will trigger a non-RC release
to [GitHub][gh-releases], [PyPI][pypi], and [DockerHub][docker-forest] (with image tags `latest` and
`<version>`).

After a final release is produced by Semantic Release, `master` will need to be `merged` back into `rc`
(due to additional `master` commits created by Semantic Release). This is necessary to re-synchronize
`rc` with `master` so that further development can continue on `rc`.

> ⚠️ **WARNING**:
> 
> Never rebase (or force-push to) the `rc` or `master` branches. Merge only, as Semantic Release
> relies heavily on tags and notes attached to previous release commits. See [this help page][sr-fp] for more info.

[sr-fp]: https://semantic-release.gitbook.io/semantic-release/support/troubleshooting#release-not-found-release-branch-after-git-push-force

> **Note:** While Semantic Release does a good job of creating release notes for the GitHub release,
> it may be necessary to hand-modify them after they're posted to simplify/clarify them. For instance,
> if a feature or fix was introduced on `rc` over a series of commits/PRs (not necessarily adjacent),
> it will result in multiple bullets in the release notes. It may be necessary to merge these bullets
> into a single bullet (maintaining references to all commits/PRs) so that the release notes make sense.

### Issue and PR Labels

We use a collection of labels to add metadata to the issues and pull requests in
the pyQuil project.

| Label | Description |
| --- | --- |
| [`bug 🐛`][bug-label] | An issue that needs fixing. |
| [`devops 🚀`][devops-label] | An issue related to CI/CD. |
| [`discussion 🤔`][discussion-label] | For design discussions. |
| [`documentation 📝`][documentation-label] | An issue for improving docs. |
| [`enhancement ✨`][enhancement-label] | A request for a new feature. |
| [`good first issue 👶`][good-first-issue-label] | A place to get started. |
| [`help wanted 👋`][help-wanted-label] | Looking for takers. |
| [`quality 🎨`][quality-label] | Improve code quality. |
| [`refactor 🔨`][refactor-label] | Rework existing functionality. |
| [`work in progress 🚧`][wip-label] | This PR is not ready to be merged. |

[bug-label]: https://github.com/rigetti/pyquil/labels/bug%20%3Abug%3A
[devops-label]: https://github.com/rigetti/pyquil/labels/devops%20%3Arocket%3A
[discussion-label]: https://github.com/rigetti/pyquil/labels/discussion%20%3Athinking%3A
[documentation-label]: https://github.com/rigetti/pyquil/labels/documentation%20%3Amemo%3A
[enhancement-label]: https://github.com/rigetti/pyquil/labels/enhancement%20%3Asparkles%3A
[good-first-issue-label]: https://github.com/rigetti/pyquil/labels/good%20first%20issue%20%3Ababy%3A
[help-wanted-label]: https://github.com/rigetti/pyquil/labels/help%20wanted%20%3Awave%3A
[quality-label]: https://github.com/rigetti/pyquil/labels/quality%20%3Aart%3A
[refactor-label]: https://github.com/rigetti/pyquil/labels/refactor%20%3Ahammer%3A
[wip-label]: https://github.com/rigetti/pyquil/labels/work%20in%20progress%20%3Aconstruction%3A

[slack-invite]: https://join.slack.com/t/rigetti-forest/shared_invite/enQtNTUyNTE1ODg3MzE2LWQwNzBlMjZlMmNlN2M5MzQyZDlmOGViODQ5ODI0NWMwNmYzODY4YTc2ZjdjOTNmNzhiYTk2YjVhNTE2NTRkODY

[docker]: https://www.docker.com/products/docker-desktop
[docker-forest]: https://hub.docker.com/r/rigetti/forest
[forest]: https://qcs.rigetti.com/sdk-downloads
[forest-tutorials]: https://github.com/rigetti/forest-tutorials
[gha]: https://github.com/rigetti/pyquil/actions
[gh-releases]: https://github.com/rigetti/pyquil/releases
[gitlab-ci]: https://gitlab.com/rigetti/forest/pyquil/pipelines
[jupyter]: https://jupyter.org/
[mybinder]: https://mybinder.org
[pep-484]: https://www.python.org/dev/peps/pep-0484/
[pypi]: https://pypi.org/project/pyquil/
[sphinx]: https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html
