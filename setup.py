from setuptools import Extension, find_packages, setup


def build_graph_extension():
    extension = Extension("matris.graph.cygraph", ["matris/graph/cygraph.pyx"])

    try:
        from Cython.Build import cythonize
    except ImportError:
        extension.sources = ["matris/graph/cygraph.c"]
        return [extension]

    return cythonize(
        [extension],
        compiler_directives={"language_level": 3},
    )


setup(
    packages=find_packages(include=["matris", "matris.*"]),
    ext_modules=build_graph_extension(),
)
