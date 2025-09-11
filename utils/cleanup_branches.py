#!/usr/bin/env python3
"""
Git Branch Cleanup Utility

This script deletes local and remote branches that haven't been merged into master,
while preserving specified protected branches.

Usage:
    python cleanup_branches.py [--protected-branches A B C] [--dry-run] [--force] [--delete-remote]

Options:
    --protected-branches: Space-separated list of branch names to protect from deletion
    --dry-run: Show what would be deleted without actually deleting
    --force: Force delete branches even if they have unmerged changes
    --delete-remote: Also delete remote branches from origin
    --help: Show this help message
"""

import subprocess
import sys
import argparse
from typing import List, Set
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class GitBranchCleaner:
    def __init__(self, protected_branches: List[str], dry_run: bool = False, force: bool = False, delete_remote: bool = False):
        self.protected_branches = set(protected_branches)
        self.dry_run = dry_run
        self.force = force
        self.delete_remote = delete_remote
        
        # Always protect master and main branches
        self.protected_branches.update(['master', 'main', 'HEAD'])
        
    def run_git_command(self, command: List[str]) -> str:
        """Run a git command and return the output."""
        try:
            result = subprocess.run(
                command, 
                capture_output=True, 
                text=True, 
                check=True,
                cwd='.'
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            logger.error(f"Git command failed: {' '.join(command)}")
            logger.error(f"Error: {e.stderr}")
            raise
    
    def get_current_branch(self) -> str:
        """Get the currently checked out branch."""
        try:
            return self.run_git_command(['git', 'branch', '--show-current'])
        except subprocess.CalledProcessError:
            logger.error("Failed to get current branch")
            sys.exit(1)
    
    def get_all_local_branches(self) -> List[str]:
        """Get all local branch names."""
        try:
            output = self.run_git_command(['git', 'branch', '--format=%(refname:short)'])
            branches = [branch.strip() for branch in output.split('\n') if branch.strip()]
            return branches
        except subprocess.CalledProcessError:
            logger.error("Failed to get local branches")
            sys.exit(1)
    
    def get_all_remote_branches(self) -> List[str]:
        """Get all remote branch names from origin."""
        try:
            output = self.run_git_command(['git', 'branch', '-r', '--format=%(refname:short)'])
            remote_branches = []
            for branch in output.split('\n'):
                branch = branch.strip()
                if branch and branch.startswith('origin/') and not branch.endswith('/HEAD'):
                    # Remove 'origin/' prefix to get just the branch name
                    branch_name = branch[7:]  # Remove 'origin/' prefix
                    remote_branches.append(branch_name)
            return remote_branches
        except subprocess.CalledProcessError:
            logger.error("Failed to get remote branches")
            return []
    
    def get_merged_branches(self) -> Set[str]:
        """Get branches that have been merged into master."""
        try:
            # First try with master
            try:
                output = self.run_git_command(['git', 'branch', '--merged', 'master', '--format=%(refname:short)'])
            except subprocess.CalledProcessError:
                # If master doesn't exist, try with main
                output = self.run_git_command(['git', 'branch', '--merged', 'main', '--format=%(refname:short)'])
            
            merged = set(branch.strip() for branch in output.split('\n') if branch.strip())
            return merged
        except subprocess.CalledProcessError:
            logger.error("Failed to get merged branches")
            sys.exit(1)
    
    def get_merged_remote_branches(self) -> Set[str]:
        """Get remote branches that have been merged into master."""
        try:
            # First try with origin/master
            try:
                output = self.run_git_command(['git', 'branch', '-r', '--merged', 'origin/master', '--format=%(refname:short)'])
            except subprocess.CalledProcessError:
                # If origin/master doesn't exist, try with origin/main
                try:
                    output = self.run_git_command(['git', 'branch', '-r', '--merged', 'origin/main', '--format=%(refname:short)'])
                except subprocess.CalledProcessError:
                    logger.warning("Neither origin/master nor origin/main found, skipping remote branch merge check")
                    return set()
            
            merged_remote = set()
            for branch in output.split('\n'):
                branch = branch.strip()
                if branch and branch.startswith('origin/') and not branch.endswith('/HEAD'):
                    # Remove 'origin/' prefix to get just the branch name
                    branch_name = branch[7:]
                    merged_remote.add(branch_name)
            return merged_remote
        except subprocess.CalledProcessError:
            logger.error("Failed to get merged remote branches")
            return set()
    
    def get_unmerged_branches(self) -> List[str]:
        """Get branches that haven't been merged into master."""
        all_branches = set(self.get_all_local_branches())
        merged_branches = self.get_merged_branches()
        current_branch = self.get_current_branch()
        
        # Remove current branch from consideration
        if current_branch:
            self.protected_branches.add(current_branch)
            logger.info(f"Protecting current branch: {current_branch}")
        
        unmerged = all_branches - merged_branches - self.protected_branches
        return sorted(list(unmerged))
    
    def get_unmerged_remote_branches(self) -> List[str]:
        """Get remote branches that haven't been merged into master."""
        if not self.delete_remote:
            return []
            
        all_remote_branches = set(self.get_all_remote_branches())
        merged_remote_branches = self.get_merged_remote_branches()
        current_branch = self.get_current_branch()
        
        # Add current branch to protected branches for remote as well
        if current_branch:
            self.protected_branches.add(current_branch)
        
        unmerged_remote = all_remote_branches - merged_remote_branches - self.protected_branches
        return sorted(list(unmerged_remote))
    
    def delete_branch(self, branch_name: str) -> bool:
        """Delete a single local branch."""
        delete_flag = '-D' if self.force else '-d'
        
        if self.dry_run:
            logger.info(f"[DRY RUN] Would delete local branch: {branch_name}")
            return True
        
        try:
            self.run_git_command(['git', 'branch', delete_flag, branch_name])
            logger.info(f"Deleted local branch: {branch_name}")
            return True
        except subprocess.CalledProcessError as e:
            if "not fully merged" in e.stderr:
                logger.warning(f"Branch '{branch_name}' is not fully merged. Use --force to delete anyway.")
            else:
                logger.error(f"Failed to delete local branch '{branch_name}': {e.stderr}")
            return False
    
    def delete_remote_branch(self, branch_name: str) -> bool:
        """Delete a single remote branch."""
        if self.dry_run:
            logger.info(f"[DRY RUN] Would delete remote branch: origin/{branch_name}")
            return True
        
        try:
            self.run_git_command(['git', 'push', 'origin', '--delete', branch_name])
            logger.info(f"Deleted remote branch: origin/{branch_name}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to delete remote branch 'origin/{branch_name}': {e.stderr}")
            return False
    
    def cleanup_branches(self) -> None:
        """Main cleanup function."""
        logger.info("Starting branch cleanup...")
        logger.info(f"Protected branches: {', '.join(sorted(self.protected_branches))}")
        
        if self.dry_run:
            logger.info("Running in DRY RUN mode - no branches will be deleted")
        
        # Get unmerged local branches
        unmerged_branches = self.get_unmerged_branches()
        
        # Get unmerged remote branches if requested
        unmerged_remote_branches = self.get_unmerged_remote_branches() if self.delete_remote else []
        
        if not unmerged_branches and not unmerged_remote_branches:
            logger.info("No unmerged branches found to delete.")
            return
        
        # Display what will be deleted
        if unmerged_branches:
            logger.info(f"Found {len(unmerged_branches)} unmerged local branches to delete:")
            for branch in unmerged_branches:
                logger.info(f"  - {branch}")
        
        if unmerged_remote_branches:
            logger.info(f"Found {len(unmerged_remote_branches)} unmerged remote branches to delete:")
            for branch in unmerged_remote_branches:
                logger.info(f"  - origin/{branch}")
        
        if not self.dry_run:
            total_branches = len(unmerged_branches) + len(unmerged_remote_branches)
            confirm = input(f"\nAre you sure you want to delete these {total_branches} branches? (y/N): ")
            if confirm.lower() not in ['y', 'yes']:
                logger.info("Operation cancelled.")
                return
        
        deleted_count = 0
        failed_count = 0
        
        # Delete local branches
        for branch in unmerged_branches:
            if self.delete_branch(branch):
                deleted_count += 1
            else:
                failed_count += 1
        
        # Delete remote branches
        for branch in unmerged_remote_branches:
            if self.delete_remote_branch(branch):
                deleted_count += 1
            else:
                failed_count += 1
        
        if self.dry_run:
            logger.info(f"[DRY RUN] Would have deleted {deleted_count} branches")
        else:
            logger.info(f"Successfully deleted {deleted_count} branches")
            if failed_count > 0:
                logger.warning(f"Failed to delete {failed_count} branches")


def main():
    parser = argparse.ArgumentParser(
        description="Delete local and remote git branches that haven't been merged into master",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Delete all unmerged local branches except protected ones
  python cleanup_branches.py
  
  # Delete both local and remote unmerged branches
  python cleanup_branches.py --delete-remote
  
  # Preview what would be deleted without actually deleting
  python cleanup_branches.py --delete-remote --dry-run
  
  # Force delete branches even if they have unmerged changes
  python cleanup_branches.py --delete-remote --force
        """
    )
    
    parser.add_argument(
        '--protected-branches',
        nargs='*',
        default=['master', 'feat/doc_up', 'feat/rust'],
        help='Branch names to protect from deletion (default: master feat/doc_up feat/rust)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be deleted without actually deleting'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force delete branches even if they have unmerged changes'
    )
    
    parser.add_argument(
        '--delete-remote',
        action='store_true',
        help='Also delete remote branches from origin'
    )
    
    args = parser.parse_args()
    
    # Validate that we're in a git repository
    try:
        subprocess.run(['git', 'rev-parse', '--git-dir'], 
                      capture_output=True, check=True)
    except subprocess.CalledProcessError:
        logger.error("This script must be run from within a git repository")
        sys.exit(1)
    
    # Create the cleaner and run cleanup
    cleaner = GitBranchCleaner(
        protected_branches=args.protected_branches,
        dry_run=args.dry_run,
        force=args.force,
        delete_remote=args.delete_remote
    )
    
    try:
        cleaner.cleanup_branches()
    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
