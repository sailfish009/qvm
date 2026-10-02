# Publication checklist — file-upload workflow

Version0.0.3 is a local reliability-update candidate for the existing public repository. No upload was performed by
this environment.

## Before upload

- [x] Apache-2.0 attribution and package author identify `sailfish009`.
- [x] No credentials or private keys are present.
- [x] Public examples do not require sibling research repositories.
- [x] All independently reported numerical, branching, lowering and immutability regressions have tests.
- [x] Standard references, semantic sweeps and property counterexamples reproduce locally.
- [x] Source wheel and clean upload bundle build locally.
- [ ] Review the 0.0.3 changelog and migration notes.
- [ ] Upload through the GitHub web interface.
- [ ] Confirm the hosted GitHub Actions workflow passes.

## Files to upload

Extract `artifacts/qvm_v0.0003_github_upload.zip` and upload its **contents**, not only the ZIP and not the
containing directory. `README.md` must appear at repository root. The equivalent extracted tree is
`artifacts/github_upload/`.

The clean bundle excludes local logs, editable-install paths, caches, wheels and generated scientific artifacts.
The wheel under `artifacts/dist/` can be attached to GitHub Release `v0.0.3` rather than committed to source.

## GitHub web interface

1. Open the existing repository and choose **Add file → Upload files**.
2. Drag the extracted bundle contents into the upload area, including `.github/` and `.gitignore`.
3. Commit with a message such as `Release 0.0.3 reliability corrections`.
4. Confirm all Actions jobs pass.
5. Create release/tag `v0.0.3` and optionally attach the wheel.

Do not upload access tokens, editable-install records or machine-specific environment files.
