import bs4
from bs4 import BeautifulSoup, Comment
import re


def get_children_for_partition(gum_tei):
    """Given the output of apply_html_transformations, return the direct children
    of the root element so we can partition them."""
    soup = parse(gum_tei)
    return soup.contents[0].contents


def rewrap_partition(children):
    text = BeautifulSoup(f"<text></text>", "html.parser")
    for child in children:
        text.contents[0].append(child)
    return str(text)


def apply_html_transformations(config, html, mwtext_object):
    tfxs = config["transformations"]["html"] or []
    html = unescape_html_elements(html)
    soup = parse(html)
    for tfx in tfxs:
        assert "name" in tfx, "Transformation must have a name"

        tfx_f = globals()[tfx["name"]] if tfx["name"] in globals() else None
        assert tfx_f is not None, "Transformation not found: " + tfx["name"]
        if "args" in tfx:
            soup = tfx_f(config, soup, **tfx["args"])
        else:
            soup = tfx_f(config, soup)

    soup = fix_root(config, soup, mwtext_object)
    soup = insert_document_title(soup, mwtext_object)
    return re.sub(r"\n+", "\n", str(soup))


def unescape_html_elements(html):
    html = re.sub(r"_-=(.*?)=-_", r"<\1>", html)
    html = re.sub(r"_-=/(.*?)=-_", r"<\1>", html)
    html = re.sub(r"_-=quot=-_", r"&quot;", html)
    return html


def insert_document_title(soup, mwtext_object):
    title_head = BeautifulSoup(f"<head>{mwtext_object.title}</head>", "html.parser")
    soup.insert(0, title_head)
    return soup


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


def discard_empty_elements(config, soup, exempt=[]):
    """Remove HTML elements that have no/whitespace-only content"""
    for tag in soup.find_all():
        if len(tag.get_text(strip=True)) == 0 and tag.name not in exempt:
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


def drop_empty_headings(config, soup, depth=0):
    headings = ["h1", "h2", "h3", "h4", "h5", "h6"]

    def last_non_whitespace_child(children, i):
        # start by assuming the heading "spans" the rest of the document
        section_end = len(children)

        # go through every child occuring later in the document
        for j in range(i + 1, len(children)):

            # if we encounter another heading that is "bigger" than the one we're considering
            # (i.e., one where the number following the "h" is less than or equal to this one's)
            # then that is where the section ends
            if children[j].name in headings and int(children[j].name[1]) <= int(
                children[i].name[1]
            ):
                section_end = j
                break

        return all(str(c).isspace() for c in children[i + 1 : section_end])

    children = list(soup.children)
    for i, child in enumerate(children):
        if child.name in headings:
            if last_non_whitespace_child(children, i):
                child.extract()
            next_child = children[i + 1]
            if next_child.name in headings and next_child.name == child.name:
                child.extract()

    if depth < 5:
        return drop_empty_headings(config, soup, depth + 1)

    return soup


def trim_trailing_headings(config, soup):
    """get rid of any <head>s, starting from the end of the document"""

    children = soup.contents
    while len(children) > 0 and children[-1].name == "head":
        children[-1].extract()
        children = soup.contents
    return soup


def remove_nested_tags(config, soup, tag_names=[]):
    def depth(node):
        if node is None:
            return 0
        return 1 + depth(node.parent)

    for tag_name in tag_names:
        for element in sorted(soup.find_all(), reverse=True, key=depth):
            if not hasattr(element, "contents") or len(element.contents) == 0:
                continue

            child_element = element.contents[0]
            if (
                len(element.contents) == 1
                and hasattr(child_element, "name")
                and child_element.name == tag_name
                and hasattr(element, "name")
                and element.name == tag_name
            ):
                element.unwrap()
    return soup
