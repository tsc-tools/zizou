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
When prompted for the username enter `__token__` and paste in your pypi token as the password. To avoid having to enter your token everytime you can setup a `.pypirc` file as described [here](https://packaging.python.org/en/latest/specifications/pypirc/).

## Documentation
Install the [mkdocs](https://www.mkdocs.org/) package as well as the [mkdocstrings](https://mkdocstrings.github.io/) 
and the [mkdocs-jupyter](https://github.com/danielfrg/mkdocs-jupyter) plugins:
```
python3 -m pip install --upgrade mkdocs "mkdocstrings[python]" mkdocs-jupyter
```

To view the documentation locally on port 8000 run:
```
mkdocs serve -a 0.0.0.0:8000
```
To publish the documentation to [Github pages](https://pages.github.com/) run:

```
mkdocs gh-deploy -r origin
```
where origin is the remote name that you have given to the github repo. This will then publish the documentation under https://tsc-tools.github.io/zizou.