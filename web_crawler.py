from pprint import pprint
import bs4
import requests


def scrape_links(from_url, for_depth, all_links={}):

    if for_depth >= 1:
        print("Scraping leve {}".format(for_depth))

        if isinstance(from_url, str):
            still_to_scrape = ['still to scrape']
        elif isinstance(from_url, list):
            still_to_scrape = ['someg']

        # list urls to scrape_links
        edges_dict = scrape_links_of_pages(still_to_scrape)
        for_depth -= 1
        all_links.update(edges_dict)

        # recursive call
        scrape_links(values, for_depth, all_links=all_links)

    else:
        print("Done! Nothing more to crawl.")

    return all_links


skipped_that = set([])
seen_that = set([])


def scrape_links_of_page(url):
    scraped_links = set([])  # unique values only

    if url not in seen_that:
        seen_that.update([url])
        try:
            h = requests.head(url, verify=False)
            content_type = h.headers.get('content-type')
            if content_type and content_type.startswith('text/html'):
                r = requests.get(url)
                if r.status_code == 200:
                    soup = bs4.BeautifulSoup(r.text)
                    a_elem = soup.select('a')

                    for link in a_elem:
                        link_candidate = link.get('href')
                        if link_candidate and link_candidate.startswith('http'):
                            scraped_links.update([link_candidate])
                else:
                    print("skipped {} ... got {}".format(r.url, r.status_code))
                    skipped_that.update([r.url])
            else:
                print('Skipped {} ... since content-type {} '.format(url, content_type))
                skipped_that.update([r.url])
        except Exception as e:
            print(e)
            skipped_that.update([url])

    return {url: list(scraped_links)}


def scrape_links_of_pages(url_lst):
    """
    url_lst:
        list of URLS from which to scrape links (href)
    return:
        dict of outgoing links
    """
    all_out_links = {}
    for url in url_lst:
        links_per_page_dict = scrape_links_of_page(url)
        all_out_links.update(links_per_page_dict)
    return all_out_links


start_url = 'https://www.version2.dk/artikel/google-deepmind-vi-oeger-sikkerheden-mod-misbrug-sundhedsdata-1074452'

url_lst = scrape_links_of_page(start_url)
pprint(scrape_links_of_pages(url_lst[start_url]))
