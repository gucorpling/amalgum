from bs4 import BeautifulSoup


def apply_html_transformations(config, html):
    tfxs = config["transformations"]["html"] or []
    soup = parse(html)
    for tfx in tfxs:
        assert "name" in tfx, "Transformation must have a name"

        tfx_f = globals()[tfx["name"]] if tfx["name"] in globals() else None
        assert tfx_f is not None, "Transformation not found: " + tfx["name"]
        if "args" in tfx:
            soup = tfx_f(soup, **tfx["args"])
        else:
            soup = tfx_f(soup)
    return str(soup)


def parse(html):
    return BeautifulSoup(html, features="html.parser")


def new_root(soup, css_selector=None):
    return soup.select(css_selector)[0]
