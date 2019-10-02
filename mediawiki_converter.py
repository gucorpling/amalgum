doc = """{{Infobox baseball biography
|name=Doc Gessler
|image=Docgessler.jpg
|position=[[Right fielder]]
|birth_date={{Birth date|1880|12|23}}
|birth_place=[[Greensburg, Pennsylvania]]
|death_date={{death date and age|1924|12|24|1880|12|23}}
|death_place=[[Greensburg, Pennsylvania]]
|bats=Left
|throws=Right
|debutleague = MLB
|debutdate=April 23
|debutyear=1903
|debutteam=Detroit Tigers
|finalleague = MLB
|finaldate=October 7
|finalyear=1911
|finalteam=Washington Senators
|statleague = MLB
|stat1label=[[Batting average (baseball)|Batting average]]
|stat1value=.280
|stat2label=[[Home run]]s
|stat2value=14
|stat3label=[[Run (baseball)|Runs scored]]
|stat3value=363
|teams=
'''As player'''
* [[Detroit Tigers]] ({{mlby|1903}})
* [[Brooklyn Superbas]] ({{mlby|1903}}–{{mlby|1906}})
* [[Chicago Cubs]] ({{mlby|1906}})
* [[Boston Red Sox]] ({{mlby|1908}}–{{mlby|1909}})
* [[Washington Senators (1901–60)|Washington Senators]] ({{mlby|1909}}–{{mlby|1911}})
'''As manager'''
* [[Pittsburgh Rebels|Pittsburgh Stogies]] ({{Baseball year|1914}})
|highlights=
*Led the [[American League]] in [[on-base percentage]] in 1908
*Led the American League in [[hit by pitch]]es in 1910
}}
'''Henry Homer "Doc" Gessler''' (December 23, 1880 – December 25, 1924) was a [[Major League Baseball]] player born in [[Greensburg, Pennsylvania]], who began his eight-season career, at the age of 22, with the [[Detroit Tigers]] in {{Baseball year|1903}}. He played mainly as a [[right fielder]] in a career that totaled 880 [[games played]], 2969 [[at bat]]s, 831 [[Hit (baseball)|hit]]s, 363 [[runs batted in|RBI]]s and 14 [[home run]]s. Doc died in Greensburg at the age of 44, and is interred in Saint Bernard Cemetery in [[Indiana, Pennsylvania]].<ref name="retrosheet">{{cite web| title = Doc Gessler's Stats | work = retrosheet.org | url=http://www.retrosheet.org/boxesetc/G/Pgessd101.htm | accessdate = 2008-02-11 }}</ref>

==College years==

Before his baseball career, he attended [[Ohio University]], [[Washington & Jefferson College]],<ref name="reference">{{cite web| title = Doc Gessler's Stats | work = baseball-reference.com | url=https://www.baseball-reference.com/g/gessldo01.shtml | accessdate = 2008-02-11 }}</ref> and became a physician, graduating from [[Johns Hopkins Medical School]]. He was one of three [[Physician|doctors]] in the [[1906 World Series]] (with [[Doc White]] and [[Frank Owen (baseball)|Frank Owen]]).<ref name="library">{{cite web | title = Doc Gessler Biography | work = baseballlibrary.com | url = http://www.baseballlibrary.com/ballplayers/player.php?name=Doc_Gessler_1880 | accessdate = 2008-02-11 | url-status = dead | archiveurl = https://web.archive.org/web/20071013125522/http://www.baseballlibrary.com/ballplayers/player.php?name=Doc_Gessler_1880 | archivedate = 2007-10-13 }}</ref>

==Career==

After his short stay with Detroit, he then moved on to the [[Brooklyn Superbas]] in an unknown transaction. For Brooklyn, he became a good hitter, [[batting average (baseball)|batting]] .290 in both of his full seasons with them. After a slow start in {{Baseball year|1906}}, he was traded to the [[Chicago Cubs]] in exchange for Hub Knolls on April 28.<ref name="retrosheet"/>

He didn't play in the Majors for the {{Baseball year|1907}} season, but reappeared for the {{Baseball year|1908}} [[Boston Red Sox]] and batted .308, hit 14 [[triple (baseball)|triples]], and led the [[American League]] in [[on-base percentage]].<ref name="retrosheet"/> The following season, [[manager (baseball)|manager]] [[Fred Lake]] announced that Doc would be team's Captain for the {{Baseball year|1909}} season.<ref name="captain">{{cite news| title = Gessler To Be Captain of The Red Sox | work = New York Times, 01-19-1909 | url=https://www.nytimes.com/1909/01/19/archives/gessler-to-be-captain-of-red-sox.html | accessdate = 2008-02-11 | date=January 19, 1909}}</ref> This situation did not last the season, as he was traded to the [[Washington Senators (1901–60)|Washington Senators]] on September 9, 1909 in exchange for Charlie Smith.<ref name="retrosheet"/> He played three seasons for the Senators and retired after the {{Baseball year|1911}} season.<ref name="retrosheet"/>

In eight seasons, Gessler posted a .280 [[batting average (baseball)|batting average]]  with 370 [[run (baseball)|runs]], 127 [[double (baseball)|doubles]], 50 [[Triple (baseball)|triples]], 14 [[home runs]], 142 [[stolen bases]], 333 [[bases on balls]], .370 [[on-base percentage]] and .370 [[slugging percentage]]. He finished his career with a .959 [[fielding percentage]] playing at right field and first base.<ref name="retrosheet" />

==Managerial stint==

Doc became the manager of the [[Pittsburgh Rebels|Pittsburgh Stogies]] of the upstart [[Federal League]] in {{Baseball year|1914}}, but after 11 games, and a 3 win 8 loss record, was replaced by [[Rebel Oakes]].<ref name="retrosheet"/> The team soon adopted the nickname '''Rebels''' after their new manager, who remained their manager through the 1914 season, and the entire {{Baseball year|1915}} season.

==References==
{{reflist}}

==External links==
{{Baseballstats |mlb=114726|espn= |br= g/gessldo01|fangraphs=1004642 |cube=11841 |brm=gessle001har}}
{{Boston Red Sox team captains}}

{{DEFAULTSORT:Gessler, Doc}}
[[Category:1880 births]]
[[Category:1924 deaths]]
[[Category:Baseball players from Pennsylvania]]
[[Category:Major League Baseball right fielders]]
[[Category:Detroit Tigers players]]
[[Category:Brooklyn Superbas players]]
[[Category:Chicago Cubs players]]
[[Category:Boston Red Sox players]]
[[Category:Washington Senators (1901–60) players]]
[[Category:Ohio Bobcats baseball players]]
[[Category:People from Greensburg, Pennsylvania]]
[[Category:Washington & Jefferson Presidents baseball players]]
[[Category:Newark Sailors players]]
[[Category:Columbus Senators players]]
[[Category:Kansas City Blues (baseball) players]]
[[Category:Johns Hopkins University alumni]]
"""

import mwparserfromhell
import subprocess

# text -> tree of parsed nodes
wikicode = mwparserfromhell.parse(doc)

for n in wikicode._nodes:
    print(type(n), n)

# we can perform arbitrary operations on the parsed nodes, e.g. discard everything after the ==References== header:
out_nodes = []
for n in wikicode._nodes:
    if isinstance(n, mwparserfromhell.nodes.heading.Heading) and n.title == "References":
        break
    out_nodes.append(n)
wikicode._nodes = out_nodes

# printing automatically produces a plain mediawiki markup string
print(wikicode)

# use Parsoid to produce HTML
parser_subprocess = subprocess.Popen(
    'parsoid/bin/parse.js',
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE
)
parser_subprocess.stdin.write(str(wikicode).encode('utf-8'))
parser_subprocess.stdin.close()
html = parser_subprocess.stdout.read().decode('utf-8')
print(html)
parser_subprocess.wait()



