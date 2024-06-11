import lintdb as ldb
from datasets import load_dataset

from tqdm import tqdm
import typer
import random
import time
import os
import pathlib
import csv
import shutil
import math
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import re


# Source: https://en.wikipedia.org/wiki/Google
sample_doc = """google llc (/ˈɡuːɡəl/ (listen)) is an american multinational technology company focusing on online advertising, search engine technology, cloud computing, computer software, quantum computing, e-commerce, artificial intelligence, and consumer electronics.
it has been referred to as "the most powerful company in the world" and one of the world's most valuable brands due to its market dominance, data collection, and technological advantages in the area of artificial intelligence.
its parent company alphabet is considered one of the big five american information technology companies, alongside amazon, apple, meta, and microsoft.
google was founded on september 4, 1998, by computer scientists larry page and sergey brin while they were phd students at stanford university in california.
together they own about 14% of its publicly listed shares and control 56% of its stockholder voting power through super-voting stock.
the company went public via an initial public offering (ipo) in 2004.
in 2015, google was reorganized as a wholly owned subsidiary of alphabet inc. google is alphabet's largest subsidiary and is a holding company for alphabet's internet properties and interests.
sundar pichai was appointed ceo of google on october 24, 2015, replacing larry page, who became the ceo of alphabet.
on december 3, 2019, pichai also became the ceo of alphabet.
the company has since rapidly grown to offer a multitude of products and services beyond google search, many of which hold dominant market positions.
these products address a wide range of use cases, including email (gmail), navigation (waze & maps), cloud computing (cloud), web browsing (chrome), video sharing (youtube), productivity (workspace), operating systems (android), cloud storage (drive), language translation (translate), photo storage (photos), video calling (meet), smart home (nest), smartphones (pixel), wearable technology (pixel watch & fitbit), music streaming (youtube music), video on demand (youtube tv), artificial intelligence (google assistant), machine learning apis (tensorflow), ai chips (tpu), and more.
discontinued google products include gaming (stadia), glass, google+, reader, play music, nexus, hangouts, and inbox by gmail.
google's other ventures outside of internet services and consumer electronics include quantum computing (sycamore), self-driving cars (waymo, formerly the google self-driving car project), smart cities (sidewalk labs), and transformer models (google brain).
google and youtube are the two most visited websites worldwide followed by facebook and twitter.
google is also the largest search engine, mapping and navigation application, email provider, office suite, video sharing platform, photo and cloud storage provider, mobile operating system, web browser, ml framework, and ai virtual assistant provider in the world as measured by market share.
on the list of most valuable brands, google is ranked second by forbes and fourth by interbrand.
it has received significant criticism involving issues such as privacy concerns, tax avoidance, censorship, search neutrality, antitrust and abuse of its monopoly position.
google began in january 1996 as a research project by larry page and sergey brin when they were both phd students at stanford university in california.
the project initially involved an unofficial "third founder", scott hassan, the original lead programmer who wrote much of the code for the original google search engine, but he left before google was officially founded as a company; hassan went on to pursue a career in robotics and founded the company willow garage in 2006.
while conventional search engines ranked results by counting how many times the search terms appeared on the page, they theorized about a better system that analyzed the relationships among websites.
they called this algorithm pagerank; it determined a website's relevance by the number of pages, and the importance of those pages that linked back to the original site.
page told his ideas to hassan, who began writing the code to implement page's ideas.
page and brin originally nicknamed the new search engine "backrub", because the system checked backlinks to estimate the importance of a site.
hassan as well as alan steremberg were cited by page and brin as being critical to the development of google.
rajeev motwani and terry winograd later co-authored with page and brin the first paper about the project, describing pagerank and the initial prototype of the google search engine, published in 1998.
héctor garcía-molina and jeff ullman were also cited as contributors to the project.
pagerank was influenced by a similar page-ranking and site-scoring algorithm earlier used for rankdex, developed by robin li in 1996, with larry page's pagerank patent including a citation to li's earlier rankdex patent; li later went on to create the chinese search engine baidu.
eventually, they changed the name to google; the name of the search engine was a misspelling of the word googol, a very large number written 10100 (1 followed by 100 zeros), picked to signify that the search engine was intended to provide large quantities of information.
google was initially funded by an august 1998 investment of $100,000 from andy bechtolsheim, co-founder of sun microsystems.
this initial investment served as a motivation to incorporate the company to be able to use the funds.
page and brin initially approached david cheriton for advice because he had a nearby office in stanford, and they knew he had startup experience, having recently sold the company he co-founded, granite systems, to cisco for $220 million.
david arranged a meeting with page and brin and his granite co-founder andy bechtolsheim.
the meeting was set for 8 am at the front porch of david's home in palo alto and it had to be brief because andy had another meeting at cisco, where he now worked after the acquisition, at 9 am.
andy briefly tested a demo of the website, liked what he saw, and then went back to his car to grab the check.
david cheriton later also joined in with a $250,000 investment.
google received money from two other angel investors in 1998: amazon.com founder jeff bezos, and entrepreneur ram shriram.
page and brin had first approached shriram, who was a venture capitalist, for funding and counsel, and shriram invested $250,000 in google in february 1998.
shriram knew bezos because amazon had acquired junglee, at which shriram was the president.
it was shriram who told bezos about google.
bezos asked shriram to meet google's founders and they met 6 months after shriram had made his investment when bezos and his wife were in a vacation trip to the bay area.
google's initial funding round had already formally closed but bezos' status as ceo of amazon was enough to persuade page and brin to extend the round and accept his investment.
between these initial investors, friends, and family google raised around $1,000,000, which is what allowed them to open up their original shop in menlo park, california.
craig silverstein, a fellow phd student at stanford, was hired as the first employee.
after some additional, small investments through the end of 1998 to early 1999, a new $25 million round of funding was announced on june 7, 1999, with major investors including the venture capital firms kleiner perkins and sequoia capital.
both firms were initially reticent about investing jointly in google, as each wanted to retain a larger percentage of control over the company to themselves.
larry and sergey however insisted in taking investments from both.
both venture companies finally agreed to investing jointly $12.5 million each due to their belief in google's great potential and through mediation of earlier angel investors ron conway and ram shriram who had contacts in the venture companies.
in march 1999, the company moved its offices to palo alto, california, which is home to several prominent silicon valley technology start-ups.
the next year, google began selling advertisements associated with search keywords against page and brin's initial opposition toward an advertising-funded search engine.
to maintain an uncluttered page design, advertisements were solely text-based.
in june 2000, it was announced that google would become the default search engine provider for yahoo!, one of the most popular websites at the time, replacing inktomi.
in 2003, after outgrowing two other locations, the company leased an office complex from silicon graphics, at 1600 amphitheatre parkway in mountain view, california.
three years later, google bought the property from sgi for $319 million.
by that time, the name "google" had found its way into everyday language, causing the verb "google" to be added to the merriam-webster collegiate dictionary and the oxford english dictionary, denoted as: "to use the google search engine to obtain information on the internet".
the first use of the verb on television appeared in an october 2002 episode of buffy the vampire slayer.
additionally, in 2001 google's investors felt the need to have a strong internal management, and they agreed to hire eric schmidt as the chairman and ceo of google.
eric was proposed by john doerr from kleiner perkins.
he had been trying to find a ceo that sergey and larry would accept for several months, but they rejected several candidates because they wanted to retain control over the company.
michael moritz from sequoia capital at one point even menaced requesting google to immediately pay back sequoia's $12.5m investment if they did not fulfill their promise to hire a chief executive officer, which had been made verbally during investment negotiations.
eric wasn't initially enthusiastic about joining google either, as the company's full potential hadn't yet been widely recognized at the time, and as he was occupied with his responsibilities at novell where he was ceo.
as part of him joining, eric agreed to buy $1 million of google preferred stocks as a way to show his commitment and to provide funds google needed.
on august 19, 2004, google became a public company via an initial public offering.
at that time larry page, sergey brin, and eric schmidt agreed to work together at google for 20 years, until the year 2024.
the company offered 19,605,052 shares at a price of $85 per share.
shares were sold in an online auction format using a system built by morgan stanley and credit suisse, underwriters for the deal.
the sale of $1.67 billion gave google a market capitalization of more than $23 billion.
on november 13, 2006, google acquired youtube for $1.65 billion in google stock, on march 11, 2008, google acquired doubleclick for $3.1 billion, transferring to google valuable relationships that doubleclick had with web publishers and advertising agencies.
by 2011, google was handling approximately 3 billion searches per day.
to handle this workload, google built 11 data centers around the world with several thousand servers in each.
these data centers allowed google to handle the ever-changing workload more efficiently.
in may 2011, the number of monthly unique visitors to google surpassed one billion for the first time.
in may 2012, google acquired motorola mobility for $12.5 billion, in its largest acquisition to date.
this purchase was made in part to help google gain motorola's considerable patent portfolio on mobile phones and wireless technologies, to help protect google in its ongoing patent disputes with other companies, mainly apple and microsoft, and to allow it to continue to freely offer android.

"""

app = typer.Typer()

@app.command()
def eval():
    if os.path.exists("experiments/goog"):
        shutil.rmtree("experiments/goog")
    # config = ldb.Configuration()
    # config.num_subquantizers = 64
    # config.dim = 128
    # config.nbits = 4
    # config.index_type = ldb.IndexEncoding_XTR
    # index = ldb.IndexIVF(f"experiments/goog", config)
    index = ldb.IndexIVF(f"experiments/goog", 50, 128, 4, 10, 64, ldb.IndexEncoding_XTR)

    opts = ldb.CollectionOptions()
    opts.model_file = "/home/matt/deployql/LintDB/assets/xtr/encoder.onnx"
    opts.tokenizer_file = "/home/matt/deployql/LintDB/assets/xtr/spiece.model"

    collection = ldb.Collection(index, opts)

    chunks = sample_doc.split("\n")

    collection.train(chunks, 50, 10)

    for i, snip in enumerate(chunks):
        collection.add(0, i, snip, {'docid': f'{i}'})

    query = "Who founded google"

    opts = ldb.SearchOptions()
    opts.k_top_centroids = 100

    print(opts.k_top_centroids)

    results = collection.search(0, query, 3, opts)

    for result in results:
        print(f"docid: {result.metadata['docid']}, score: {result.score}")




if __name__ == "__main__":
    app()