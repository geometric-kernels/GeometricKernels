# Contribution guidelines

### Who are we?

*Maintainers* (in alphabetical order: Viacheslav Borovitskiy, Vincent Dutordoir, Peter Mostowsky) steer the project, keep the community thriving, and manage contributions.

*Contributors* (you?) submit issues, make pull requests, answer questions on Slack, and more.

Community is important to us, and we want everyone to feel welcome and be able to contribute to their fullest. Our [code of conduct](CODE_OF_CONDUCT.md) gives an overview of what that means.

Here is the list of original contributors to the project (in alphabetical order): Viacheslav Borovitskiy, Vincent Dutordoir, Michael Hutchinson, Noemie Jaquier,  Peter Mostowsky, Aditya Ravuri, Alexander Terenin.

### Reporting a bug

Finding and fixing bugs helps us provide robust functionality to all users. You can either submit a bug report or, if you know how to fix the bug yourself, you can submit a bug fix. We gladly welcome either, but a fix is likely to be released sooner, simply because others may not have time to quickly implement a fix themselves. If you're interested in implementing it, but would like help in doing so, you can send [the maintainers](#who-are-we) an email or open an [issue](https://github.com/geometric-kernels/GeometricKernels/issues/new).

We use GitHub issues for bug reports. You can use the [issue template](https://github.com/geometric-kernels/GeometricKernels/issues/new) to start writing yours. Once you've submitted it, the maintainers will take a look as soon as possible, ideally within the week, and get back to you about how to proceed. If it's a small easy fix, they may implement it then and there. For fixes that are more involved, they will discuss with you about how urgent the fix is, with the aim of providing some timeline of when you can expect to see it.

If you'd like to submit a bug fix, [open a pull request](https://github.com/geometric-kernels/GeometricKernels/compare). We recommend you discuss your changes with the community before you begin working on them (e.g. via issues), so that questions and suggestions can be made early on.

### Requesting a feature

GeometricKernels is built on features added and improved by the community. You can submit a feature request either as an issue or, if you can implement the change yourself, as a pull request. We gladly welcome either, but a pull request is likely to be released sooner, simply because others may not have time to quickly implement it themselves.

We use GitHub issues for feature requests. You can use the [issue template](https://github.com/geometric-kernels/GeometricKernels/issues/new) to start writing yours. Once you've submitted it, the maintainers will take a look as soon as possible, ideally within the week, and get back to you about how to proceed. If it's a small easy feature that is backwards compatible, they may implement it then and there. For features that are more involved, they will discuss with you about a timeline for implementing it. Features that are not backwards compatible are likely to take longer to reach a release. It may become apparent during discussions that a feature doesn't lie within the scope of GeometricKernels, in which case we will discuss alternative options with you, such as adding it as a notebook or an external extension to GeometricKernels.

If you'd like to submit a pull request, [open a pull request](https://github.com/geometric-kernels/GeometricKernels/compare). We recommend you discuss your changes with the community before you begin working on them (e.g. via issues), so that questions and suggestions can be made early on.

### Pull request guidelines

- Limit the pull request to the smallest useful feature or enhancement, or the smallest change required to fix a bug. This makes it easier for reviewers to understand why each change was made, and makes reviews quicker.
- Where appropriate, include [documentation](#documentation), [type hints](#type-checking), and [tests](#tests). See those sections for more details.
- Pull requests that modify or extend the code should include appropriate tests, or be covered by already existing tests. In particular:
  - New features should include a demonstration of how to use the new API, and should include sufficient tests to give confidence that the feature works as expected.
  - Bug fixes should include tests to verify that the updated code works as expected and defend against future regressions.
  - When refactoring code, verify that existing tests are adequate.
- In commit messages, be descriptive but to the point. Comments such as "further fixes" obscure the more useful information.

### Documentation

GeometricKernels has two primary sources of documentation: the notebooks and the API reference.

For the API reference, we document Python code inline, using [reST markup](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html). See [here](docs/README.md) for details on the documentation build. All parts of the public API need docstrings (indeed anything without docstrings won't appear in the built documentation). Similarly, don't add docstrings to private functionality, else it will appear in the documentation website. Use code comments sparingly, as they incur a maintenance cost and tend to drift out of sync with the corresponding code.

### Quality checks

We use [make](https://www.gnu.org/software/make/manual/make.html#toc-Overview-of-make) to run our commands. This guide assumes you have this installed.

#### Type and format checking

We use [type hints](https://docs.python.org/3/library/typing.html) for documentation and static type checking with [mypy](http://mypy-lang.org). We format all Python code, other than the notebooks, with [black](https://black.readthedocs.io/en/stable/), [flake8](https://flake8.pycqa.org/en/latest/), and [isort](https://pycqa.github.io/isort/). You may need to run these before pushing changes, with (in the repository root)

Run the type and format checkers (black, flake8 and mypy) with
```bash
$ make lint
```

The formatter (isort and black) can be run with
```bash
$ make format
```

#### Tests

We write and run tests with [pytest](https://pytest.org).

Run tests with
```bash
$ make test
```

#### Continuous integration

[GitHub actions](https://github.com/geometric-kernels/GeometricKernels/blob/main/.github/workflows/quality-checks.yaml) will automatically run the quality checks against pull requests to the develop branch. The GitHub repository is set up such that these need to pass in order to merge.


# License

[Apache License 2.0](LICENSE)
