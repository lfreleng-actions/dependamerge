<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: 2026 The Linux Foundation
-->

# Bot Policy

Dependamerge exists to land the steady stream of version bumps that
dependency and hook-update bots raise. Other GitHub Apps raise pull
requests too, and some carry code changes that need a human reviewer, so
being a bot is not enough to be in scope. The bot policy decides which
Apps' pull requests a run acts on.

## Defaults

| App | In scope by default |
| --- | --- |
| `dependabot[bot]` | Yes |
| `renovate[bot]` | Yes |
| `pre-commit-ci[bot]` | Yes |
| Any other App | No |

A policy can exclude a built-in App, and include any other one. Pull
requests from an App outside the policy appear in the run's summary with
the reason, rather than disappearing:

```text
🚫 Bot PRs out of scope: 16
   lf-releng-code-monkey-bot[bot] ×16 (not included by any policy)
   Include an App with --include-bots <name>, or in a policy file
```

A source pull request from such an App stops the run with exit code `8`
and the same hint. Being outside the policy is not something an
`--override` can waive: include the App instead.

Who counts as a bot follows from the account type GitHub reports, never
from a name. A person whose login contains "bot", or a User account
called `renovate`, is a person.

## Naming Apps

Each entry names one App, case-insensitively, in any of these forms:

- `lf-releng-code-monkey-bot`
- `lf-releng-code-monkey-bot[bot]`
- `app/lf-releng-code-monkey-bot`

`*` in an inclusion list includes every App. It restores the behaviour of
releases before this policy existed, so use it knowingly.

Dependamerge looks each listed App up once per run. An exclusion naming
no App stops the run with exit code `2` before anything merges, since a
misspelt exclusion would leave its App in scope. An inclusion naming no
App produces a warning.

## Sources

Most specific first:

| Level | Source |
| --- | --- |
| Run | `--include-bots` and `--exclude-bots` on `merge` and `close` |
| Repository | `.github/dependamerge/config.json` in the repository |
| Repository | The repository's `DEPENDAMERGE_BOTS` Actions variable |
| Owner | `.github/dependamerge/config.json` in the owner's `.github` repo |
| Owner | The organisation's `DEPENDAMERGE_BOTS` Actions variable |
| User | `config.json` in `$XDG_CONFIG_HOME/dependamerge` |
| Built-in | The defaults above |

`$XDG_CONFIG_HOME` defaults to `~/.config` on every platform.

For each App, the most specific source that names it decides. So an
owner can include its own bot everywhere while one repository excludes
it, and `--include-bots renovate` brings Renovate back into scope for one
run whatever the files say. Your own file never overrules a repository
or owner; the run's flags can, and the run says when they do:

```text
ℹ️ This run's flags overrule my-org/.github for renovate[bot]
```

### Run flags

Both flags take comma- or space-separated names and may repeat:

```bash
# Also merge the organisation's own update bot, but never Renovate
dependamerge merge https://github.com/my-org \
  --include-bots my-org-update-bot --exclude-bots renovate
```

### User file

A JSON object with either or both lists:

```json
{
  "include_bots": ["my-org-update-bot"],
  "exclude_bots": ["renovate"]
}
```

A list may also be a single string of comma- or space-separated names.
Any other key is an error: a misspelt `exlude_bots` must stop the run,
not leave a bot in scope. So is a file that is not valid JSON, is larger
than 16 KiB, or names one App in both lists. An invalid file stops the
run with exit code `2` before dependamerge fetches anything.

### Repository and owner files

The same document, committed as `.github/dependamerge/config.json`:

- in a repository, for its own pull requests;
- in the owner's `.github` repository, for every repository the
  organisation or user owns.

Dependamerge reads each file from the repository's default branch, never
from a pull request, so a pull request cannot change the policy that
applies to it.

The files cost no requests of their own. The owner's file arrives with
the query that tells an organisation from a user, which replaces one
that fetched the first page of repositories twice, and a repository's
with the first page of its open pull requests. A run against one pull
request reads the owner's file and its repository's in one request,
before it judges the pull request's author.

An invalid owner file stops the run with exit code `2` before anything
merges. An invalid repository file takes none of that repository's pull
requests into scope, people's included; the run reports it and carries
on with the other repositories:

```text
⚠️ Skipped 3 PR(s) in my-org/widget: its bot policy file is invalid
```

Dependamerge looks up the Apps a file names as it does for the flags. An
unknown exclusion makes a repository's file invalid, and stops the run
in the owner's file.

### Actions variables

A repository or organisation can also hold its policy in an Actions
variable named `DEPENDAMERGE_BOTS`, containing the same JSON document as
a file. At one level, the file and the variable combine: each list is the
union of both. An App that one includes and the other excludes makes the
level's policy invalid, with the same outcome as an invalid file.

GraphQL does not expose variables, so each costs a REST request.
Dependamerge reads the organisation's variable once, and a repository's
when that repository holds an App's pull request, since nothing else
there depends on it. Files cost no requests, so prefer them at scale.

Reading variables needs more than reading code:

| Variable | Classic token | Fine-grained token |
| --- | --- | --- |
| Repository | `repo` scope | "Variables" repository permission, read |
| Organisation | `admin:org` scope | "Variables" organisation permission, read |

A token can lack one and hold the other. When GitHub refuses a kind of
variable, dependamerge stops reading that kind and says so once. The
refused variables' inclusions do not apply, and their exclusions cannot
hold, so keep exclusions that must hold in files. Any other failure to
read a variable stops the run, since the variable might hold an
exclusion:

```text
⚠️ This token cannot read the organisation's DEPENDAMERGE_BOTS Actions
variables: their inclusions do not apply and their exclusions cannot
hold. Keep exclusions that must hold in policy files.
```
