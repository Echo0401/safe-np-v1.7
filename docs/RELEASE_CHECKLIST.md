# Public release checklist

This folder is reviewable locally. It is not yet a final authorized public release.

- [ ] Confirm code holders (including institutional/coauthor claims), select the final LICENSE **only after authorization**, and resolve third-party source attribution.
- [ ] Fill the paper's approved title, authors, citation/DOI and final repository URL. No citation metadata is invented here.
- [ ] Review WESAD reference provenance and keep it separate from v1.7 Safe NP results.
- [ ] Decide whether each group-level result table may be made public.
- [ ] Confirm the public RELAX data license/version and WESAD provider terms.
- [ ] Decide how authorized reviewers may obtain restricted inputs; do not add them to this source repository.
- [x] Author-reported: clean Windows Python 3.10 virtual environment, dependencies installed, `pip check` clear, six offline checks passed, and one minimal real RELAX training run completed (29 October 2026). See `docs/LOCAL_VALIDATION_20261029.md`. This does **not** cover complete 65-fold retraining.
- [ ] Inspect source/config numeric settings and compare any newly authorized input against the frozen input hash.
- [ ] Rerun `scripts/scan_release.py` after every edit or added asset. A static scan cannot prove full anonymity or ownership.
- [ ] Keep raw data, individual predictions, exact longitudinal time keys, model weights, personal documents and all recovered chats outside Git.
- [ ] Review every file to be committed (`git status`, `git diff --cached --name-only`, `git ls-files`); `.gitignore` does not remove an already tracked private file. Do not commit `data/restricted/`, `runs/`, `.idea/`, original individual predictions, or the private audit files.
- [ ] Keep the local `private_review_do_not_publish` directory and history-recovery archives outside the repository.

GitHub warns above 50 MiB and blocks regular Git files above 100 MiB; this candidate checks the lower threshold. See [GitHub's file-size documentation](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github). Large assets require a separate approved distribution plan, not automatic upload or a blanket move to LFS.

No remote repository, push, data upload, public release or credential change was performed in this task.
