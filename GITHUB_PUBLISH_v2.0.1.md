# Publish MOD_Mask v2.0.1

Repository: `Gandome/rural_mask_grid_city_crop`

Version 2.0.0 and `legacy-v1` remain preserved. Publish v2.0.1 as a new commit/tag;
do not move or delete the existing v2.0.0 tag.

## 1. Clone or update the repository

```bash
git clone https://github.com/Gandome/rural_mask_grid_city_crop.git
cd rural_mask_grid_city_crop
git checkout main
git pull origin main
```

## 2. Copy the v2.0.1 release contents

From the parent directory containing both folders:

```bash
rsync -av --delete \
  --exclude '.git/' \
  --exclude '.pytest_cache/' \
  --exclude '__pycache__/' \
  rural_mask_grid_city_crop-v2.0.1-public/ \
  rural_mask_grid_city_crop/
```

## 3. Test and commit

```bash
cd rural_mask_grid_city_crop
python -m pytest -v
git status
git add -A
git commit -m "Release MOD_Mask v2.0.1"
git push origin main
```

## 4. Tag v2.0.1

```bash
git tag -a v2.0.1 -m "MOD_Mask v2.0.1"
git push origin v2.0.1
```

## 5. Create the GitHub Release

```bash
gh release create v2.0.1 \
  ../rural_mask_grid_city_crop-v2.0.1-public.zip \
  --repo Gandome/rural_mask_grid_city_crop \
  --title "MOD_Mask v2.0.1" \
  --notes-file RELEASE_NOTES_v2.0.1.md \
  --verify-tag \
  --latest
```

## 6. Verify

```bash
gh release view v2.0.1 \
  --repo Gandome/rural_mask_grid_city_crop

git log --oneline --decorate --graph --all -10
```

Expected version relationships:

- `legacy-v1` / `pre-v2.0.0-20260829`: previous public implementation
- `v2.0.0`: immutable original v2 release
- `main` / `v2.0.1`: current patch release
