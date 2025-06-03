# Test Data Directory

## ⚠️ IMPORTANT: RAW Files Policy

**DO NOT COMMIT RAW FILES TO GIT**

This directory is for test data, but RAW image files (ARW, CR2, NEF, DNG, etc.) must NEVER be committed to the git repository because:

1. They exceed GitHub's recommended file size limit (50MB)
2. They will bloat the repository size
3. They are binary files that don't benefit from version control

## How to Use Test Data

If you need RAW files for testing:

1. Place them in this directory locally
2. They will be automatically ignored by git (see .gitignore)
3. The files will be available for local testing but won't be tracked

## Supported RAW Formats (all gitignored)

- Sony: *.ARW
- Canon: *.CR2
- Nikon: *.NEF
- Adobe: *.DNG
- Fujifilm: *.RAF
- Olympus: *.ORF
- Panasonic: *.RW2
- Pentax: *.PEF
- Samsung: *.SRW
- Sigma: *.X3F

All extensions are ignored in both uppercase and lowercase.