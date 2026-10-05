# GitHub update — web file upload

Release candidate: **0.0.6**, seven quantum-equation layers without attention/QKV requirements.
No GitHub upload, hosted CI run, tag creation, or package-registry publication is performed by these scripts.

## Local verification

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python examples/inspect_overlap_geometry.py
python examples/train_overlap_embedding.py
python examples/equations_overview.py
python examples/reproduce_standard_semantics.py
python examples/reproduce_reliability_cases.py
python examples/assumption_sweep.py
python examples/discover_counterexamples.py
python -m pip wheel . --no-deps --wheel-dir artifacts/dist
python tools/build_upload.py
```

Review `CHANGELOG.md`, `MIGRATION.md`, `RESULTS.md` and `OVERLAP_API.md`. Check that Apache-2.0 attribution and package
metadata are appropriate for your repository. The clean source ZIP excludes local logs, generated experiment data,
editable-install records, wheels, caches, and machine-specific environment files.

## Files

- Upload source tree: `artifacts/github_upload/`
- Identical ZIP: `artifacts/qvm_v0.0006_github_upload.zip`
- Optional Release attachment: `artifacts/dist/qvm_counterfactual-0.0.6-py3-none-any.whl`
- `SOURCE_MANIFEST.json` in the upload tree records file hashes (excluding itself).

## GitHub 웹 업로드 순서

1. 기존 GitHub 저장소에서 **Add file → Upload files**를 선택합니다.
2. ZIP을 먼저 압축 해제하고 **내부 파일과 폴더**를 업로드합니다. ZIP 파일 자체만 올리지 마세요.
3. `README.md`, `pyproject.toml`, `qvm/`, `tests/`가 저장소 최상위에 놓이는지 확인합니다.
4. 숨김 항목인 `.github/`와 `.gitignore`도 포함합니다. 파일 선택기에서 보이지 않으면 숨김 파일 표시를 켭니다.
5. 변경 내용을 검토하고 `Release 0.0.6: quantum-equation layers` 같은 메시지로 commit합니다.
6. Actions에서 Python 3.11–3.14 작업이 통과하는지 확인합니다. 로컬 통과와 hosted CI 통과는 다릅니다.
7. 통과 후 `v0.0.6` tag/release를 만들고 wheel을 필요에 따라 첨부합니다.

웹 업로드는 기존 파일을 갱신하지만, 새 묶음에 없는 과거 파일을 자동으로 삭제하지는 않습니다.
기존 저장소의 연구 기록은 보존하고, 오래된 생성물이나 중복 파일의 정리는 별도로 검토하세요.
토큰·인증 정보·개인 환경 파일을 업로드하지 마세요.

## Owner verification (not yet performed by the agent)

- [ ] Review source changes and migration notes.
- [ ] Upload through the GitHub web interface.
- [ ] Confirm all hosted Actions jobs pass.
- [ ] Create release/tag `v0.0.6` and attach the wheel if desired.
