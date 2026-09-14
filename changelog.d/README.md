# Changelog fragments

One file per pull request that changes something a user would notice.  At release time
`towncrier` collects them into `CHANGELOG.md` and empties this directory.

Name the file `<pr-number>.<type>.md`, where the type is one of:

| type       | goes under          |
|------------|---------------------|
| `breaking` | Breaking Changes    |
| `feature`  | Features            |
| `fix`      | Fixes               |
| `doc`      | Documentation       |
| `misc`     | Other Changes       |

The contents are one or two sentences in the present tense, written for someone reading
the release notes rather than the diff:

```
$ cat changelog.d/1869.feature.md
A gate angle may now be any Quil arithmetic expression over declared memory, so the
output of `quilc` simulates without rewriting.
```

`make changelog-fragment` creates one for you, or just write the file by hand.  A pull
request that changes no behaviour -- a refactor, a test, a CI tweak -- needs no fragment.
