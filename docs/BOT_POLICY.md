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
| User | `config.json` in `$XDG_CONFIG_HOME/dependamerge` |
| Built-in | The defaults above |

`$XDG_CONFIG_HOME` defaults to `~/.config` on every platform.

For each App, the most specific source that names it decides. So
`--include-bots renovate` brings Renovate back into scope for one run
even when your own file excludes it.

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
