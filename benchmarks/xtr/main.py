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
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')


# Source: https://en.wikipedia.org/wiki/Google
sample_doc = """Google LLC (/ˈɡuːɡəl/ (listen)) is an American multinational technology company focusing on online advertising, search engine technology, cloud computing, computer software, quantum computing, e-commerce, artificial intelligence,[9] and consumer electronics. It has been referred to as "the most powerful company in the world"[10] and one of the world's most valuable brands due to its market dominance, data collection, and technological advantages in the area of artificial intelligence.[11][12][13] Its parent company Alphabet is considered one of the Big Five American information technology companies, alongside Amazon, Apple, Meta, and Microsoft.
Google was founded on September 4, 1998, by computer scientists Larry Page and Sergey Brin while they were PhD students at Stanford University in California. Together they own about 14% of its publicly listed shares and control 56% of its stockholder voting power through super-voting stock. The company went public via an initial public offering (IPO) in 2004. In 2015, Google was reorganized as a wholly owned subsidiary of Alphabet Inc. Google is Alphabet's largest subsidiary and is a holding company for Alphabet's internet properties and interests. Sundar Pichai was appointed CEO of Google on October 24, 2015, replacing Larry Page, who became the CEO of Alphabet. On December 3, 2019, Pichai also became the CEO of Alphabet.[14]
The company has since rapidly grown to offer a multitude of products and services beyond Google Search, many of which hold dominant market positions. These products address a wide range of use cases, including email (Gmail), navigation (Waze & Maps), cloud computing (Cloud), web browsing (Chrome), video sharing (YouTube), productivity (Workspace), operating systems (Android), cloud storage (Drive), language translation (Translate), photo storage (Photos), video calling (Meet), smart home (Nest), smartphones (Pixel), wearable technology (Pixel Watch & Fitbit), music streaming (YouTube Music), video on demand (YouTube TV), artificial intelligence (Google Assistant), machine learning APIs (TensorFlow), AI chips (TPU), and more. Discontinued Google products include gaming (Stadia), Glass, Google+, Reader, Play Music, Nexus, Hangouts, and Inbox by Gmail.[15][16]
Google's other ventures outside of Internet services and consumer electronics include quantum computing (Sycamore), self-driving cars (Waymo, formerly the Google Self-Driving Car Project), smart cities (Sidewalk Labs), and transformer models (Google Brain).[17]
Google and YouTube are the two most visited websites worldwide followed by Facebook and Twitter. Google is also the largest search engine, mapping and navigation application, email provider, office suite, video sharing platform, photo and cloud storage provider, mobile operating system, web browser, ML framework, and AI virtual assistant provider in the world as measured by market share. On the list of most valuable brands, Google is ranked second by Forbes[18] and fourth by Interbrand.[19] It has received significant criticism involving issues such as privacy concerns, tax avoidance, censorship, search neutrality, antitrust and abuse of its monopoly position.
Google began in January 1996 as a research project by Larry Page and Sergey Brin when they were both PhD students at Stanford University in California.[20][21][22] The project initially involved an unofficial "third founder", Scott Hassan, the original lead programmer who wrote much of the code for the original Google Search engine, but he left before Google was officially founded as a company;[23][24] Hassan went on to pursue a career in robotics and founded the company Willow Garage in 2006.[25][26]
While conventional search engines ranked results by counting how many times the search terms appeared on the page, they theorized about a better system that analyzed the relationships among websites.[27] They called this algorithm PageRank; it determined a website's relevance by the number of pages, and the importance of those pages that linked back to the original site.[28][29] Page told his ideas to Hassan, who began writing the code to implement Page's ideas.[23]
Page and Brin originally nicknamed the new search engine "BackRub", because the system checked backlinks to estimate the importance of a site.[20][30][31] Hassan as well as Alan Steremberg were cited by Page and Brin as being critical to the development of Google. Rajeev Motwani and Terry Winograd later co-authored with Page and Brin the first paper about the project, describing PageRank and the initial prototype of the Google search engine, published in 1998. Héctor García-Molina and Jeff Ullman were also cited as contributors to the project.[32] PageRank was influenced by a similar page-ranking and site-scoring algorithm earlier used for RankDex, developed by Robin Li in 1996, with Larry Page's PageRank patent including a citation to Li's earlier RankDex patent; Li later went on to create the Chinese search engine Baidu.[33][34]
Eventually, they changed the name to Google; the name of the search engine was a misspelling of the word googol,[20][35][36] a very large number written 10100 (1 followed by 100 zeros), picked to signify that the search engine was intended to provide large quantities of information.[37]
Google was initially funded by an August 1998 investment of $100,000 from Andy Bechtolsheim,[20] co-founder of Sun Microsystems. This initial investment served as a motivation to incorporate the company to be able to use the funds.[39][40] Page and Brin initially approached David Cheriton for advice because he had a nearby office in Stanford, and they knew he had startup experience, having recently sold the company he co-founded, Granite Systems, to Cisco for $220 million. David arranged a meeting with Page and Brin and his Granite co-founder Andy Bechtolsheim. The meeting was set for 8 AM at the front porch of David's home in Palo Alto and it had to be brief because Andy had another meeting at Cisco, where he now worked after the acquisition, at 9 AM. Andy briefly tested a demo of the website, liked what he saw, and then went back to his car to grab the check. David Cheriton later also joined in with a $250,000 investment.[41][42]
Google received money from two other angel investors in 1998: Amazon.com founder Jeff Bezos, and entrepreneur Ram Shriram.[43] Page and Brin had first approached Shriram, who was a venture capitalist, for funding and counsel, and Shriram invested $250,000 in Google in February 1998. Shriram knew Bezos because Amazon had acquired Junglee, at which Shriram was the president. It was Shriram who told Bezos about Google. Bezos asked Shriram to meet Google's founders and they met 6 months after Shriram had made his investment when Bezos and his wife were in a vacation trip to the Bay Area. Google's initial funding round had already formally closed but Bezos' status as CEO of Amazon was enough to persuade Page and Brin to extend the round and accept his investment.[44][45]
Between these initial investors, friends, and family Google raised around $1,000,000, which is what allowed them to open up their original shop in Menlo Park, California.[46] Craig Silverstein, a fellow PhD student at Stanford, was hired as the first employee.[22][47][48]
After some additional, small investments through the end of 1998 to early 1999,[43] a new $25 million round of funding was announced on June 7, 1999,[49] with major investors including the venture capital firms Kleiner Perkins and Sequoia Capital.[40] Both firms were initially reticent about investing jointly in Google, as each wanted to retain a larger percentage of control over the company to themselves. Larry and Sergey however insisted in taking investments from both. Both venture companies finally agreed to investing jointly $12.5 million each due to their belief in Google's great potential and through mediation of earlier angel investors Ron Conway and Ram Shriram who had contacts in the venture companies.[50]
In March 1999, the company moved its offices to Palo Alto, California,[51] which is home to several prominent Silicon Valley technology start-ups.[52] The next year, Google began selling advertisements associated with search keywords against Page and Brin's initial opposition toward an advertising-funded search engine.[53][22] To maintain an uncluttered page design, advertisements were solely text-based.[54] In June 2000, it was announced that Google would become the default search engine provider for Yahoo!, one of the most popular websites at the time, replacing Inktomi.[55][56]
In 2003, after outgrowing two other locations, the company leased an office complex from Silicon Graphics, at 1600 Amphitheatre Parkway in Mountain View, California.[58] Three years later, Google bought the property from SGI for $319 million.[59] By that time, the name "Google" had found its way into everyday language, causing the verb "google" to be added to the Merriam-Webster Collegiate Dictionary and the Oxford English Dictionary, denoted as: "to use the Google search engine to obtain information on the Internet".[60][61] The first use of the verb on television appeared in an October 2002 episode of Buffy the Vampire Slayer.[62]
Additionally, in 2001 Google's investors felt the need to have a strong internal management, and they agreed to hire Eric Schmidt as the chairman and CEO of Google.[46] Eric was proposed by John Doerr from Kleiner Perkins. He had been trying to find a CEO that Sergey and Larry would accept for several months, but they rejected several candidates because they wanted to retain control over the company. Michael Moritz from Sequoia Capital at one point even menaced requesting Google to immediately pay back Sequoia's $12.5m investment if they did not fulfill their promise to hire a chief executive officer, which had been made verbally during investment negotiations. Eric wasn't initially enthusiastic about joining Google either, as the company's full potential hadn't yet been widely recognized at the time, and as he was occupied with his responsibilities at Novell where he was CEO. As part of him joining, Eric agreed to buy $1 million of Google preferred stocks as a way to show his commitment and to provide funds Google needed.[63]
On August 19, 2004, Google became a public company via an initial public offering. At that time Larry Page, Sergey Brin, and Eric Schmidt agreed to work together at Google for 20 years, until the year 2024.[64] The company offered 19,605,052 shares at a price of $85 per share.[65][66] Shares were sold in an online auction format using a system built by Morgan Stanley and Credit Suisse, underwriters for the deal.[67][68] The sale of $1.67 billion gave Google a market capitalization of more than $23 billion.[69]
On November 13, 2006, Google acquired YouTube for $1.65 billion in Google stock,[70][71][72][73] On March 11, 2008, Google acquired DoubleClick for $3.1 billion, transferring to Google valuable relationships that DoubleClick had with Web publishers and advertising agencies.[74][75]
By 2011, Google was handling approximately 3 billion searches per day. To handle this workload, Google built 11 data centers around the world with several thousand servers in each. These data centers allowed Google to handle the ever-changing workload more efficiently.[46]
In May 2011, the number of monthly unique visitors to Google surpassed one billion for the first time.[76][77]
In May 2012, Google acquired Motorola Mobility for $12.5 billion, in its largest acquisition to date.[78][79][80] This purchase was made in part to help Google gain Motorola's considerable patent portfolio on mobile phones and wireless technologies, to help protect Google in its ongoing patent disputes with other companies,[81] mainly Apple and Microsoft,[82] and to allow it to continue to freely offer Android.[83]
"""
sample_doc = re.sub(r'\[\d+\]', '', sample_doc)

# Single-sentence chunks.
chunks = [chunk.lower() for chunk in sent_tokenize(sample_doc)]

app = typer.Typer()

use_collection = False

@app.command()
def eval():
    if os.path.exists("experiments/goog"):
        shutil.rmtree("experiments/goog")
    config = ldb.Configuration()
    config.num_subquantizers = 64
    config.dim = 128
    config.nbits = 4
    config.quantizer_type = ldb.IndexEncoding_XTR
    index = ldb.IndexIVF(f"experiments/goog", config)
    # index = ldb.IndexIVF(f"experiments/goog", 50, 128, 4, 10, 64, ldb.IndexEncoding_XTR)


    if use_collection:
        opts = ldb.CollectionOptions()
        opts.model_file = "/home/matt/deployql/LintDB/assets/xtr/encoder.onnx"
        opts.tokenizer_file = "/home/matt/deployql/LintDB/assets/xtr/spiece.model"

        collection = ldb.Collection(index, opts)

        collection.train(chunks, 50, 10)

        for i, snip in enumerate(chunks):
            collection.add(0, i, snip, {'docid': f'{i}'})
    else:
        import tensorflow as tf
        import tensorflow_text as text

        # tokenizer = text.SentencePieceTokenizer("/home/matt/deployql/LintDB/assets/xtr/spiece.model", add_eos=True)
        model = tf.saved_model.load('/home/matt/deployql/LintDB/assets/xtr/tf')
        encoder = model.signatures['serving_default']

        training_embeddings = None
        for i, snip in enumerate(chunks):
            embeddings = encoder(tf.constant([snip.lower()]))
            batch_lengths = np.sum(embeddings["mask"].numpy(), axis=1)
            length = int(batch_lengths[0])
            embeds = embeddings['encodings'].cpu().numpy()[0, :length, :]
            if training_embeddings is None:
                training_embeddings = embeds
            else:
                training_embeddings = np.append(training_embeddings, embeds, axis=0)

        print("first training embedding")
        print(training_embeddings[0])
        index.train(training_embeddings, 50, 10)

        for i, snip in enumerate(chunks):
            embeddings = encoder(tf.constant([snip.lower()]))
            batch_lengths = np.sum(embeddings["mask"].numpy(), axis=1)
            length = int(batch_lengths[0])
            embeds = embeddings['encodings'].cpu().numpy()[0, :length, :]
            passage = {'embeddings': embeds, 'id': i, 'metadata': {'docid': f'{i}'}}
            index.add(0, [passage])

    query = "Who founded google"

    opts = ldb.SearchOptions()
    opts.k_top_centroids = 1
    opts.nearest_tokens_to_fetch = 100
    opts.n_probe = 1

    if use_collection:
        results = collection.search(0, query, 3, opts)
    else:
        embeddings = encoder(tf.constant([query]))
        batch_lengths = np.sum(embeddings["mask"].numpy(), axis=1)
        length = int(batch_lengths[0])
        embeds = embeddings['encodings'].cpu().numpy()[0, :length, :]
        results = index.search(0, embeds, 3, opts)

    for result in results:
        print(f"docid: {result.metadata['docid']}, score: {result.score}")




if __name__ == "__main__":
    app()