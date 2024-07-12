# How to issue a new zizou release

## Pypi
Install the build system
```
python3 -m pip install --upgrade build
```
Install `twine`:
```
python3 -m pip install --upgrade twine
```
Make sure the version number is incremented in the `pyproject.toml` file.

Run the build to create two new files under `dist`. One is the wheel and the other the packaged source code:
```
python3 -m build
```
Upload to pypi:
```
python3 -m twine upload dist/*
```
When prompted for the username enter `__token__` and paste in your pypi token as the password.

## Documentation
Install the [mkdocs](https://www.mkdocs.org/) package and the [mkdocstrings](https://mkdocstrings.github.io/) plugin:
```
python3 -m pip install --upgrade mkdocs "mkdocstrings[python]"
```

To view the documentation locally run:
```
mkdocs serve
```
To generate documentation run:
```
mkdocs build
```
from the project root directory. This creates a directory called `site` containing all the necessary files to host the documentation.
Please don't add `site` to version control. If it is the first time you built the documentation, run the following:

```
mv site ../tsc-tools-website
cd ../tsc-tools-website
git init
git add .
git commit -m "update documentation"
git branch -m main
git add origin git@github.com:tsc-tools/zizou.github.io.git
git push -u --force origin main
```
[Github pages](https://pages.github.com/) will then publish the documentation under https://tsc-tools.github.io.