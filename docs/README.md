# Compiling Documentation

## Installing Depdenencies

If you are using `uv`, first install docs building depedenencies **while in the `docs/` dirrectory**.
```bash
uv add -r requirements.txt
```
If you are using `pip`, you can run
```bash
pip install -r requirements.txt
```

## Running Doctests

Execute doctests by running the following command **while in the `docs/` dirrectory**.
```bash
make doctest
```

## Compiling Documentation

If all tests are passed, run the following command **while in the `docs/` dirrectory**.
```bash
make html
```

To view the compiled documentation open the file `_build/html/index.html`.
