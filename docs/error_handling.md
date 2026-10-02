<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: 2025 The Linux Foundation
-->

# Error Handling in Dependamerge

This document describes the centralized error handling system implemented in
dependamerge to provide consistent, user-friendly error reporting.

## Overview

The error handling system provides:

- **Standardized exit codes** for different error categories
- **Consistent error messages** with rich formatting
- **Smart error detection** for common failure scenarios
- **Automatic error conversion** from exceptions to structured formats
- **Graceful cleanup** of resources like progress trackers

## Exit Codes

Dependamerge uses semantic exit codes for different failure types:

| Code | Name                | Description                                     |
| ---- | ------------------- | ----------------------------------------------- |
| 0    | SUCCESS             | Operation completed                             |
| 1    | GENERAL_ERROR       | General operational failure                     |
| 2    | CONFIGURATION_ERROR | Configuration validation failed                 |
| 3    | GITHUB_API_ERROR    | GitHub API access failed                        |
| 4    | NETWORK_ERROR       | Network connectivity issues                     |
| 5    | REPOSITORY_ERROR    | Git repository operation failed                 |
| 6    | PR_STATE_ERROR      | Pull request in invalid state                   |
| 7    | MERGE_ERROR         | Pull requests or Gerrit changes left unmerged   |
| 8    | VALIDATION_ERROR    | Input validation failed                         |

### When a merge run exits 7

A `merge` run that actually merges (`--no-confirm`, or a confirmed
interactive run) exits `7` when any pull request or Gerrit change it
attempted ends **failed** or **blocked**: those need a human. It prints
the full summary first, so the exit code never replaces the detail.

Other outcomes do not count as failures. An **unsettled** PR merges on
a re-run, **auto-merge pending** completes on GitHub's side, and
**skipped** or **closed** PRs leave nothing to follow up. A preview or
`--dry-run` predicts outcomes rather than producing them, so it exits
`0` whatever it predicts.

## Error Messages

All error messages follow a consistent format:

- Start with ❌ emoji for visual clarity
- Use active voice and clear language
- Provide actionable guidance when possible
- Include relevant details for debugging

Example:

```text
❌ GitHub API access failed; ensure GITHUB_TOKEN has required permissions
Details: Resource not accessible by integration
```

## Usage Examples

### Basic Error Handling

```python
from dependamerge.error_codes import exit_with_error, ExitCode

# Exit with a specific error code
exit_with_error(
    ExitCode.CONFIGURATION_ERROR,
    message="❌ Invalid configuration provided",
    details="Missing required field 'token'",
)
```

### GitHub API Errors

```python
from dependamerge.error_codes import exit_for_github_api_error

# Handle GitHub API permission errors
try:
    api_call()
except Exception as e:
    exit_for_github_api_error(
        details="Failed to access repository",
        exception=e
    )
```

### Pull Request State Errors

```python
from dependamerge.error_codes import exit_for_pr_state_error

# Handle invalid PR states
if pr.state != "open":
    exit_for_pr_state_error(
        pr_number=123,
        pr_state="closed",
        details="Cannot merge closed pull request"
    )
```

### Custom Error Handling

```python
from dependamerge.error_codes import DependamergeError, ExitCode

# Create custom structured errors
error = DependamergeError(
    exit_code=ExitCode.MERGE_ERROR,
    message="❌ Merge conflict detected",
    details="Manual resolution required",
    original_exception=original_error
)

# Display error and exit
error.display_and_exit()
```

## Error Detection

The system automatically detects common error patterns:

### GitHub API Permission Errors

```python
from dependamerge.error_codes import is_github_api_permission_error

if is_github_api_permission_error(exception):
    # Handle permission-related errors
    pass
```

Detects patterns like:

- "Resource not accessible by integration"
- "Bad credentials"
- "401 Unauthorized"
- "403 Forbidden"

### Network Errors

```python
from dependamerge.error_codes import is_network_error

if is_network_error(exception):
    # Handle network connectivity issues
    pass
```

Detects patterns like:

- "Connection refused"
- "Network is unreachable"
- "Connection timed out"
- "DNS resolution failed"

### Rate Limit Errors

```python
from dependamerge.error_codes import is_rate_limit_error

if is_rate_limit_error(exception):
    # Handle rate limiting
    pass
```

Detects patterns like:

- "API rate limit exceeded"
- "Secondary rate limit"
- "Excessive requests"

## Error Conversion

The system converts existing exceptions to structured errors:

```python
from dependamerge.error_codes import (
    convert_git_error,
    convert_github_api_error,
    convert_network_error
)

# Convert Git errors
try:
    git_operation()
except GitError as e:
    structured_error = convert_git_error(e)
    structured_error.display_and_exit()

# Convert GitHub API errors
try:
    api_call()
except Exception as e:
    structured_error = convert_github_api_error(e)
    structured_error.display_and_exit()
```

## CLI Integration

The CLI main functions use comprehensive error handling:

```python
try:
    main_operation()
except DependamergeError as exc:
    # Structured errors handle display and exit
    exc.display_and_exit()
except (GitError, RateLimitError, GraphQLError) as exc:
    # Convert known errors to structured format
    converted_error = convert_appropriate_error(exc)
    converted_error.display_and_exit()
except Exception as e:
    # Categorize and handle unexpected errors
    if is_github_api_permission_error(e):
        exit_for_github_api_error(exception=e)
    else:
        exit_with_error(ExitCode.GENERAL_ERROR, exception=e)
```

## Progress Tracker Cleanup

All error paths ensure proper cleanup of progress trackers:

```python
try:
    operation_with_progress()
except Exception as e:
    if progress_tracker:
        progress_tracker.stop()
    handle_error(e)
```

## Best Practices

1. **Use specific exit codes** instead of generic GENERAL_ERROR when possible
2. **Provide helpful details** that guide users toward solutions
3. **Log technical details** while showing user-friendly messages
4. **Clean up resources** like progress trackers in all error paths
5. **Convert exceptions** to structured errors for consistency
6. **Test error paths** to ensure they work as expected

## Migration Guide

To update existing error handling:

1. Replace `typer.Exit(1)` with appropriate `exit_for_*` functions
2. Replace generic `print` statements with structured error messages
3. Add error detection and conversion for common exception types
4. Ensure progress tracker cleanup in all error paths
5. Use consistent error message formatting

Example migration:

```python
# Before
if not token:
    print("Error: GitHub token required")
    raise typer.Exit(1)

# After
if not token:
    exit_for_configuration_error(
        message="❌ GitHub token required",
        details="Provide --token or set GITHUB_TOKEN environment variable"
    )
```

## Testing

Test error handling with mock objects to avoid actual program termination:

```python
from unittest.mock import patch

with patch("sys.exit") as mock_exit:
    try:
        exit_with_error(ExitCode.VALIDATION_ERROR, "Test error")
    except SystemExit:
        pass

    # Verify correct exit code
    mock_exit.assert_called_with(8)
```

This centralized approach ensures consistent error handling across the entire
dependamerge codebase while providing clear guidance to users when operations
fail.
