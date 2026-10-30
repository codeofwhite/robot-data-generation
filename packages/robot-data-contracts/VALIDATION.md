# Validation

The unchanged protocol source passed 27 CPU tests. Independently installed wheels
also wrote/read 15 Panda/Isaac/PhysX Episodes in the bounded public validation:
5/5 nominal and 7/10 XY-retargeted task successes. All failures were retained.
Release preparation additionally installed both packages from one checkout into a
fresh Python 3.14.6 venv with separately resolved dependencies; all 49 CPU tests
passed (27 contracts + 22 Generation), and pip check passed. Full public simulation
evidence accompanies the enclosing Generation repository, not this standalone
protocol source distribution.
The extraction changes packaging/documentation only; Episode v1 is unchanged.
