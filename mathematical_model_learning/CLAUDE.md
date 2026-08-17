# Code Modification Safety Rules

- Before any edit touching more than one file, create a git checkpoint:
  ```
  git add -A && git commit -m 'checkpoint: [brief description]' && git tag checkpoint-$(date +%s)
  ```
- After completing edits, verify by running the project's build and test commands.
- If the build fails or tests break, immediately `git reset --hard` to the last checkpoint and try a different approach.
- After 3 consecutive successful checkpoints, squash them into a single meaningful commit with a detailed message.
- NEVER modify global configuration files outside the project repository. If a fix requires global changes, stop and ask for explicit permission.
