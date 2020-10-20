import arxivscraper
import pandas as pd
scraper = arxivscraper.Scraper(category='physics:astro-ph', date_from='2017-05-27',date_until='2017-06-07',filters={'categories':['astro-ph.GA']})
output = scraper.scrape()
cols = ('id', 'title', 'categories', 'abstract', 'doi', 'created', 'updated', 'authors')
Abstracts = pd.DataFrame(output,columns=cols)['abstract']
for i in range(len(Abstracts)):
    with open('trente.csv','a') as fd:
        fd.write("\""+Abstracts[i]+"\""+","+"astro-ph.GA")
df = pd.read_csv("trente.csv")
print(df["Abstract"][1])
