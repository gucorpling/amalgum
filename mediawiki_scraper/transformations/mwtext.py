import mwparserfromhell
import re


def _cleanup(mwtext):
    mwtext = mwtext.replace("<noinclude>", "")
    mwtext = mwtext.replace("</noinclude>", "")
    return mwtext


def apply_mwtext_transformations(config, mwtext):
    tfxs = config["transformations"]["mwtext"] or []
    wikicode = mwparserfromhell.parse(_cleanup(mwtext))
    for tfx in tfxs:
        assert "name" in tfx, "Transformation must have a name"

        tfx_f = globals()[tfx["name"]] if tfx["name"] in globals() else None
        assert tfx_f is not None, "Transformation not found: " + tfx["name"]
        if "args" in tfx:
            wikicode = tfx_f(config, wikicode, **tfx["args"])
        else:
            wikicode = tfx_f(config, wikicode)
    return str(wikicode)


def drop_headings(config, wikicode, titles=None):
    out_nodes = []
    skip = False
    titles = [x.lower().strip() for x in titles]
    for n in wikicode._nodes:
        if (
            isinstance(n, mwparserfromhell.nodes.heading.Heading)
            and n.title.lower().strip() in titles
        ):
            skip = True
        elif isinstance(n, mwparserfromhell.nodes.heading.Heading):
            skip = False
        if skip:
            continue
        out_nodes.append(n)
    wikicode._nodes = out_nodes
    return wikicode


def drop_templates(config, wikicode, re_patterns=[]):
    out_nodes = []
    for n in wikicode._nodes:
        name = str(n).replace(" ", "_").replace("{", "").replace("}", "")
        if isinstance(n, mwparserfromhell.nodes.template.Template) and any(
            re.match(pattern, name) for pattern in re_patterns
        ):
            continue
        out_nodes.append(n)
    wikicode._nodes = out_nodes
    return wikicode


def transform_image_wikilinks(config, wikicode):
    out_nodes = []
    for n in wikicode._nodes:
        name = str(n)
        if isinstance(n, mwparserfromhell.nodes.wikilink.Wikilink) and name.startswith(
            "[[Image:"
        ):
            title_str = str(n.title).replace("Image:", "").replace('"', "_=_quot_=_")
            tag = mwparserfromhell.nodes.text.Text(
                f' _-=figure rend="{title_str}"=-__-=/figure=-_'
            )
            out_nodes.append(tag)
        else:
            out_nodes.append(n)
    wikicode._nodes = out_nodes
    return wikicode


def transform_wikihow_video_templates(config, wikicode):
    out_nodes = []
    for n in wikicode._nodes:
        if (
            isinstance(n, mwparserfromhell.nodes.template.Template)
            and n.name == "whvid"
        ):
            title_str = str(n.params[0])
            tag = mwparserfromhell.nodes.text.Text(
                f' _-=figure rend="{title_str}"=-__-=/figure=-_'
            )
            out_nodes.append(tag)
        else:
            out_nodes.append(n)
    wikicode._nodes = out_nodes
    return wikicode
