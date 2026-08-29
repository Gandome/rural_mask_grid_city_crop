# Publish MOD_Mask v2.0.0 as the latest GitHub release

Repository: `Gandome/rural_mask_grid_city_crop`

The current repository state should be preserved before replacing `main`.

## 1. Clone the current repository

```bash
git clone https://github.com/Gandome/rural_mask_grid_city_crop.git
cd rural_mask_grid_city_crop
```

## 2. Preserve the pre-v2 state

```bash
git tag -a pre-v2 -m "Repository state before MOD_Mask v2.0.0"
git push origin pre-v2
```

## 3. Copy the v2.0.0 release contents into the clone

From the parent directory, assuming the release folder is named
`rural_mask_grid_city_crop-v2.0.0-public`:

```bash
rsync -av --delete \
  --exclude '.git/' \
  rural_mask_grid_city_crop-v2.0.0-public/ \
  rural_mask_grid_city_crop/
```

Then:

```bash
cd rural_mask_grid_city_crop
git status
git add -A
git commit -m "Release MOD_Mask v2.0.0"
git push origin main
```

## 4. Tag the release

```bash
git tag -a v2.0.0 -m "MOD_Mask v2.0.0"
git push origin v2.0.0
```

## 5. Publish the GitHub Release

With GitHub CLI installed and authenticated:

```bash
gh release create v2.0.0 \
  ../rural_mask_grid_city_crop-v2.0.0-public.zip \
  --repo Gandome/rural_mask_grid_city_crop \
  --title "MOD_Mask v2.0.0" \
  --notes-file RELEASE_NOTES_v2.0.0.md \
  --verify-tag \
  --latest
```

## 6. Verify

```bash
gh release view v2.0.0 \
  --repo Gandome/rural_mask_grid_city_crop \
  --json tagName,name,isDraft,isPrerelease,url
```

Also verify the `tests` workflow under the repository's **Actions** tab.
