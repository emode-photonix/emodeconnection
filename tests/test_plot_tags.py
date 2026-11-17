"""``plot(tags=)``: a script's plot tagged like a view saved from the tab (GUI-M4d, emode#1704).

Eric, 2026-11-17: every entry point that saves a view takes ``tags=``, a list
of strings (or one string separated by commas), trimmed and without blanks or
repeats as the tab's sheets keep them. The tags travel in the ``SavedView``
posted to EMode, so the simulation file keeps them, and the sidebar groups the
script's plot by its first tag like any saved view.

Replacing a saved view by name keeps its tags when the call gives no
``tags=`` (None), and replaces them when it does (``tags=[]`` clears them).
"""

from __future__ import annotations

from typing import Any, List

import pytest

from emode.plotting import SavedView
from emode.types import ArgumentError

from plot_fixtures import farfield
from test_plot_inventory import _round_trip
from test_saved_views import _recipe, _sidebar
from test_saved_views import em as _saved_em  # noqa: F401  (fixture)
from test_saved_views import file_views  # noqa: F401  (fixture)
from test_workbench import desktop  # noqa: F401  (autouse fixture)
from test_workbench import em as _workbench_em  # noqa: F401  (fixture)


@pytest.fixture
def em(_saved_em):  # noqa: F811
    return _saved_em


def _posted(em: Any) -> List[SavedView]:
    return [c["view"] for c in em.client.calls("EM_save_view")]


def _tags_of(em: Any, title: str) -> List[str]:
    return next(v["tags"] for v in _sidebar(em) if v["title"] == title)


def test_a_tagged_plot_posts_its_tags_and_the_sidebar_lists_them(em):
    em.plot(plane="x-y", component="Ey", name="TE", tags=["modes", "C band"])
    assert _posted(em)[-1].name == "TE" and _posted(em)[-1].tags == ["modes", "C band"]
    assert _tags_of(em, "TE") == ["modes", "C band"]


def test_tags_given_as_text_are_split_trimmed_and_deduplicated(em):
    em.plot(plane="x-y", component="Ey", tags=" modes, C band ,,modes ")
    assert _posted(em)[-1].tags == ["modes", "C band"]
    em.plot(plane="z-x", component="Ey", tags=["a", " a ", "", "b"])
    assert _posted(em)[-1].tags == ["a", "b"]


def test_tags_that_are_not_strings_are_refused(em):
    with pytest.raises(ArgumentError, match="tags"):
        em.plot(plane="x-y", tags=5)
    assert _posted(em) == []


def test_every_entry_point_takes_tags(em):
    em.plot_S_matrix(tags=["ports"])
    em.material_explorer(material="Si", tags=["index"])
    assert [v.tags for v in _posted(em)] == [["ports"], ["index"]]


def test_replacing_by_name_keeps_the_tags_unless_the_call_gives_them(em):
    em.plot(plane="x-y", component="Ey", name="TE", tags=["modes"])
    em.plot(plane="z-x", component="Ey", name="TE")
    assert _posted(em)[-1].tags == ["modes"]
    em.plot(plane="x-y", component="Ex", name="TE", tags=["draft"])
    assert _posted(em)[-1].tags == ["draft"]
    em.plot(plane="x-y", component="Ex", name="TE", tags=[])
    assert _posted(em)[-1].tags == []
    assert [v["title"] for v in _sidebar(em)].count("TE") == 1


def test_the_same_plot_again_with_new_tags_posts_them(em):
    em.plot(plane="x-y", component="Ey", tags=["a"])
    posts = len(_posted(em))
    em.plot(plane="x-y", component="Ey")  # the same recipe, no tags given: nothing to post
    assert len(_posted(em)) == posts
    em.plot(plane="x-y", component="Ey", tags=["b"])
    assert len(_posted(em)) == posts + 1 and _posted(em)[-1].tags == ["b"]
    assert len(_sidebar(em)) == 1


def test_a_files_tagged_view_replaced_by_a_tagless_call_keeps_its_tags(em, file_views):
    file_views.views = [_recipe("TE", tags=["modes"])]
    em.plot(name="TE", plane="z-x")
    assert _posted(em)[-1].tags == ["modes"]


def test_a_client_held_view_takes_tags_and_is_not_posted(em):
    farfield().show(em=em, name="far", tags=["beams"])
    assert _tags_of(em, "far") == ["beams"]
    assert _posted(em) == []


def test_the_tags_cross_the_wire_in_the_saved_view(em):
    em.plot(plane="x-y", component="Ey", name="TE", tags=["modes"])
    assert _round_trip(_posted(em)[-1]).tags == ["modes"]
