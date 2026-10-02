# Publication checklist — file-upload workflow

The source is prepared locally but has not been uploaded.

## Before upload

- [x] License text identifies `Copyright 2026 sailfish009`.
- [x] Package author metadata identifies `sailfish009`.
- [x] No credentials or private keys are present.
- [x] Public examples do not depend on `/research` or sibling repositories.
- [x] Standard reference vectors include source hashes and a portable manifest.
- [x] Tests, source reproduction, wheel build and isolated import pass locally.
- [ ] Confirm the final GitHub repository name and description.
- [ ] Confirm Apache-2.0 remains the intended license.
- [ ] Run the GitHub Actions workflow after upload.

## Files to upload

Use the contents of `artifacts/github_upload/` or extract `artifacts/qvm_v0.0002_github_upload.zip`. Upload the
**contents**, not the containing directory and not only the ZIP file, so `README.md` appears at the repository
root.

The clean upload bundle excludes local logs, editable-install paths, caches and the locally built wheel. A wheel
can be attached later to a GitHub Release.

## GitHub web interface

1. Create an empty repository without auto-generating a README or license.
2. Choose **Add file → Upload files**.
3. Drag the extracted bundle contents into the upload area.
4. Use a commit message such as `Initial public release: qvm_v0.0002`.
5. After upload, open the **Actions** tab and confirm all tests pass.
6. Optionally create release/tag `v0.0.2` and attach the wheel from `artifacts/dist/`.

Do not upload access tokens or local environment files.
