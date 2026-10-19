"""Tests for Components filetree layout, headers, renames, and System section."""
from pathlib import Path
import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from frontend.config_tree import ConfigTreeWidget, find_config_root


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_components_panel_tree_headers_and_renames(qapp):
    """Verify that Components mode displays category headers, renames, and a System section at the end."""
    tree_widget = ConfigTreeWidget(mode="components")
    tree = tree_widget.tree

    # Collect top-level item texts and UserRole data
    top_items = []
    for i in range(tree.topLevelItemCount()):
        item = tree.topLevelItem(i)
        data = item.data(0, Qt.ItemDataRole.UserRole) or {}
        top_items.append((item.text(0), data, item))

    top_names = [name for name, _, _ in top_items]

    # Verify expected headers in exact order
    expected_headers = [
        "Algorithms",
        "Environments",
        "Models",
        "Data Loaders",
        "Evaluators",
        "Figures",
        "System",
    ]
    assert top_names == expected_headers, f"Expected top headers {expected_headers}, got {top_names}"

    # Verify all top-level items are styled as headers (is_group=True, no icon)
    for name, data, item in top_items:
        assert data.get("is_group") is True, f"Header '{name}' should have is_group=True"
        assert item.icon(0).isNull(), f"Header '{name}' should have no icon, but found one"
        # Font should be bold
        font = item.font(0)
        assert font.bold(), f"Header '{name}' font should be bold"

    # Verify System node children
    system_item = top_items[-1][2]
    system_child_names = []
    for i in range(system_item.childCount()):
        child = system_item.child(i)
        data = child.data(0, Qt.ItemDataRole.UserRole) or {}
        system_child_names.append((child.text(0), data, child))

    child_labels = [name for name, _, _ in system_child_names]
    assert "Global Defaults (config)" in child_labels
    assert "paradigms" in child_labels
    assert "site" in child_labels
    assert "hydra" in child_labels
    assert "workflow" in child_labels

    # Verify that system config groups have is_group=False and have non-null icons
    for name in ("paradigms", "site", "hydra", "workflow"):
        item_tuple = next((t for t in system_child_names if t[0] == name), None)
        assert item_tuple is not None, f"Group '{name}' missing under System"
        _, data, item = item_tuple
        assert data.get("is_group") is False, f"System group '{name}' should have is_group=False"
        assert not item.icon(0).isNull(), f"System group '{name}' should have an icon"

    # Verify file navigation works through headers and system groups
    selected = tree_widget.select_file("agent/dqn.yaml")
    if selected:
        curr = tree.currentItem()
        assert curr is not None
        assert "dqn" in curr.text(0)
        assert curr.parent().text(0) == "Algorithms"

    selected_sys = tree_widget.select_file("site/local.yaml")
    if selected_sys:
        curr = tree.currentItem()
        assert curr is not None
        assert "local" in curr.text(0)
        assert curr.parent().text(0) == "site"
        assert curr.parent().parent().text(0) == "System"

    selected_global = tree_widget.select_file("config.yaml")
    if selected_global:
        curr = tree.currentItem()
        assert curr is not None
        assert "Global Defaults (config)" in curr.text(0)
        assert curr.parent().text(0) == "System"


def test_components_panel_namespaced_subfolders(qapp):
    """Verify that Figures, Data Loaders, and Evaluators organize into namespaced subfolders."""
    config_root = find_config_root()
    test_subdirs = [
        config_root / "plot" / "mock_ns",
        config_root / "data" / "mock_ns",
        config_root / "eval" / "mock_ns",
    ]
    created_files = []
    try:
        for sdir in test_subdirs:
            sdir.mkdir(parents=True, exist_ok=True)
            dummy_file = sdir / "test_item.yaml"
            dummy_file.write_text("test_key: 123\n", encoding="utf-8")
            created_files.append(dummy_file)

        tree_widget = ConfigTreeWidget(mode="components")
        tree = tree_widget.tree

        headers = {}
        for i in range(tree.topLevelItemCount()):
            item = tree.topLevelItem(i)
            headers[item.text(0)] = item

        # Check Figures subfolders
        figures_item = headers["Figures"]
        figures_children = [figures_item.child(j).text(0) for j in range(figures_item.childCount())]
        assert "mock_ns" in figures_children, f"Expected 'mock_ns' in Figures, got {figures_children}"
        if (config_root / "plot" / "thetalib").exists():
            assert "thetalib" in figures_children

        # Check Data Loaders subfolders
        data_item = headers["Data Loaders"]
        data_children = [data_item.child(j).text(0) for j in range(data_item.childCount())]
        assert "mock_ns" in data_children, f"Expected 'mock_ns' in Data Loaders, got {data_children}"

        # Check Evaluators subfolders
        eval_item = headers["Evaluators"]
        eval_children = [eval_item.child(j).text(0) for j in range(eval_item.childCount())]
        assert "mock_ns" in eval_children, f"Expected 'mock_ns' in Evaluators, got {eval_children}"

        # Verify selecting namespaced files
        selected = tree_widget.select_file("plot/mock_ns/test_item.yaml")
        assert selected is True
        curr = tree.currentItem()
        assert curr is not None
        assert curr.text(0) == "test_item"
        assert curr.parent().text(0) == "mock_ns"
        assert curr.parent().parent().text(0) == "Figures"
    finally:
        for f in created_files:
            if f.exists():
                f.unlink()
        for sdir in test_subdirs:
            if sdir.exists() and not any(sdir.iterdir()):
                sdir.rmdir()

