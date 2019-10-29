from bs4 import BeautifulSoup, Comment
import re


def apply_html_transformations(config, html, mwtest_object):
    tfxs = config["transformations"]["html"] or []
    soup = parse(html)
    for tfx in tfxs:
        assert "name" in tfx, "Transformation must have a name"

        tfx_f = globals()[tfx["name"]] if tfx["name"] in globals() else None
        assert tfx_f is not None, "Transformation not found: " + tfx["name"]
        if "args" in tfx:
            soup = tfx_f(config, soup, **tfx["args"])
        else:
            soup = tfx_f(config, soup)

    soup = fix_root(config, soup, mwtest_object)
    return re.sub(r"\n+", "\n", str(soup))


def fix_root(config, soup, mwtext_object):
    soup.name = "text"
    soup.attrs = {
        "id": "AUTOGUM_" + config["family"] + "_" + mwtext_object.file_safe_url,
        "revid": mwtext_object.rev_id,
        "sourceURL": mwtext_object.url,
        "type": config["family"],
        "title": mwtext_object.title,
    }
    return soup


def parse(html):
    return BeautifulSoup(html, features="html.parser")


def new_root(config, soup, css_selector=None):
    """Find an HTML element and discard ALL content above it; i.e., make it the new document root"""
    return soup.select(css_selector)[0]


def discard_empty_elements(config, soup):
    """Remove HTML elements that have no/whitespace-only content"""
    for tag in soup.find_all():
        if len(tag.get_text(strip=True)) == 0:
            tag.extract()
    return soup


def discard_comments(config, soup):
    for tag in soup(text=lambda text: isinstance(text, Comment)):
        tag.extract()
    return soup


def discard_attributes_by_name(config, soup, name_regexes=[]):
    """Discard all attributes on all elements whose names match a pattern."""
    for pattern in name_regexes:
        if not pattern[0] == "^":
            pattern = "^" + pattern
        if not pattern[-1] == "$":
            pattern += "$"

        for tag in soup.find_all():
            if hasattr(tag, "attrs"):
                tag.attrs = {
                    k: v for k, v in tag.attrs.items() if not re.search(pattern, k)
                }

    return soup


def discard_elements(config, soup, css_selectors=[]):
    def depth(node):
        if node is None:
            return 0
        return 1 + depth(node.parent)

    for selector in css_selectors:
        for tag in sorted(soup.select(selector), reverse=True, key=depth):
            tag.extract()

    return soup


def excise_elements(config, soup, css_selectors=[]):
    """
    For each css selector, get rid of them while preserving their children.
    E.g., if the css selector is "span":

        <p>Exact science based on <b><span><em><span>Cubics</span></em></span></b>,
        not on <span>theories</span>. Wisdom is Cubic testing of knowledge.</p>

    becomes

        <p>Exact science based on <b><em>Cubics</em></b>,
        not on theories. Wisdom is Cubic testing of knowledge.</p>

    inspired by: https://stackoverflow.com/questions/1765848/remove-a-tag-using-beautifulsoup-but-keep-its-contents
    """

    def depth(node):
        if node is None:
            return 0
        return 1 + depth(node.parent)

    for selector in css_selectors:
        for tag in sorted(soup.select(selector), reverse=True, key=depth):
            if getattr(tag, "parent", None):
                while len(tag.contents) > 0:
                    c = tag.contents[0]
                    tag.insert_before(c)
                tag.extract()

    return soup


def substitute_tags(config, soup, substitutions):
    for substitution in substitutions:
        src_name = substitution["src_tag"]
        new_name = substitution["new_tag"]
        new_attrs = (
            substitution["new_tag_attrs"] if "new_tag_attrs" in substitution else None
        )
        attr_map = substitution["attr_map"] if "attr_map" in substitution else None

        for tag in soup.find_all():
            if tag.name == src_name:
                tag.name = new_name
                if attr_map and hasattr(tag, "attrs"):
                    for old_attr_name, new_attr_name in attr_map.items():
                        v = tag.attrs[old_attr_name]
                        # special handling for URLs: attempt to resolve if it's relative
                        if (
                            new_name == "ref"
                            and new_attr_name == "target"
                            and v.startswith("./")
                        ):
                            v = "https://" + config["url"] + "/wiki" + v[1:]
                        tag.attrs[new_attr_name] = v
                        del tag.attrs[old_attr_name]
                if new_attrs:
                    if not hasattr(tag, "attrs"):
                        tag.attrs = new_attrs
                    else:
                        tag.attrs.update(new_attrs)

    return soup


def excise_unless_whitelisted(config, soup, whitelist=[]):
    def depth(node):
        if node is None:
            return 0
        return 1 + depth(node.parent)

    for tag in sorted(soup.find_all(), reverse=True, key=depth):
        if getattr(tag, "parent", None) and tag.name not in whitelist:
            while len(tag.contents) > 0:
                c = tag.contents[0]
                tag.insert_before(c)
            tag.extract()

    return soup
