import bs4
import requests


r = requests.get('https://github.com/UsernameDiana')
r.raise_for_status()
soup = bs4.BeautifulSoup(r.text)

print(soup.prettify()[:1500])