import os

import peewee as pw
import datetime


PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
db = pw.SqliteDatabase(PARENT_DIR + os.sep + "texts.db")


class BaseModel(pw.Model):
    class Meta:
        database = db


class MWText(BaseModel):
    mediawiki_link = pw.CharField(primary_key=True)
    url = pw.CharField(unique=True)
    rev_id = pw.IntegerField()
    text = pw.CharField()
    title = pw.CharField()
    file_safe_url = pw.CharField()
    scraped_at = pw.DateTimeField(default=datetime.datetime.now)


def add_text(mediawiki_link, url, rev_id, text, title, file_safe_url):
    return MWText.create(
        mediawiki_link = mediawiki_link,
        url=url,
        rev_id=rev_id,
        text=text,
        title=title,
        file_safe_url=file_safe_url
    )


def mwtext_exists(mediawiki_link):
    return MWText.select().where(MWText.mediawiki_link == mediawiki_link).count() > 0

def get_mwtext(mediawiki_link):
    return MWText.get(MWText.mediawiki_link == mediawiki_link)

def mwtext_exists_by_url(url):
    return MWText.select().where(MWText.url == url).count() > 0

def get_mwtext_by_url(url):
    return MWText.get(MWText.url == url)

def initialize():
    db.connect()
    db.create_tables([MWText], safe=True)


initialize()
