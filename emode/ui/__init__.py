"""The browser interface of an EMode session: the app tab, its server and its figures.

``figure.py`` turns EMode's plot data into a renderer-neutral ``FigureSpec``,
``renderers/`` draws one with Plotly (the tab) or matplotlib (a static image),
``server.py`` is the session's local web server, ``views.py`` the session's
plots and ``session.py`` what ``EMode.plot()`` does. Importing this package
imports neither matplotlib nor a browser.
"""
