## Summary

<!-- 1-3 sentences: what this PR changes and why. -->

<details>
<summary>Details</summary>

<!-- Optional: longer walkthrough, commit breakdown, benchmark tables, or command output. -->

</details>

## Test plan

<!-- How was this tested? Check all that apply. -->

- [ ] C++ tests pass (`ctest --test-dir build --output-on-failure`)
- [ ] Python tests pass (`uv run pytest tests/python/ -v`)
- [ ] Doc examples pass (`uv run pytest --codeblocks docs/ README.md -v`)
- [ ] Pre-commit checks pass (`uv run --frozen --only-group dev pre-commit run --all-files --show-diff-on-failure`)

## Contributor agreement

- [ ] I confirm this contribution is my original work (or I have the right to
  submit it) and I license it under this project's
  [Apache-2.0 license](../LICENSE).
- [ ] If this contribution was AI-assisted, I have reviewed the generated code
  and included an `Assisted-by:` git trailer in the commit message.
