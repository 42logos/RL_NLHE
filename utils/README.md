# Utils Directory

This directory contains utility scripts and functions for the NLHE project.

## Scripts

### cleanup_branches.py

A Python script to delete local and remote git branches that haven't been merged into master, while preserving specified protected branches.

**Usage:**
```bash
# Delete all unmerged local branches except protected ones
python utils/cleanup_branches.py

# Delete both local and remote unmerged branches
python utils/cleanup_branches.py --delete-remote

# Preview what would be deleted without actually deleting
python utils/cleanup_branches.py --delete-remote --dry-run

# Force delete branches even if they have unmerged changes
python utils/cleanup_branches.py --delete-remote --force

# Use custom protected branches
python utils/cleanup_branches.py --protected-branches feature-x develop staging --delete-remote
```

**Features:**
- Automatically protects `master`, `main`, and the current branch
- Supports custom protected branch names
- Can delete both local and remote branches
- Dry-run mode to preview changes
- Force delete option for unmerged branches
- Interactive confirmation before deletion
- Comprehensive logging and error handling

**Options:**
- `--protected-branches`: Space-separated list of branch names to protect from deletion (default: master feat/doc_up feat/rust)
- `--dry-run`: Show what would be deleted without actually deleting
- `--force`: Force delete branches even if they have unmerged changes
- `--delete-remote`: Also delete remote branches from origin
- `--help`: Show help message

**Safety Features:**
- Always protects `master`, `main`, and current branch
- Interactive confirmation before deletion (unless in dry-run mode)
- Graceful handling of non-existent branches
- Clear logging of all operations
- Separate handling of local and remote branches
