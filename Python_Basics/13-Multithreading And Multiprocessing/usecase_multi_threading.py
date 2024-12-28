'''
Real-world example: Multithreading for I/O-bound Tasks
Scenario: Web scraping
Web scraping often involves making numerous network requests to fetch web pages.
These tasks are I/O-bound because they spend a lot of time waiting for responses from servers.
Multithreading can significantly improve the performace by allowing multiple web pages to be fetched concurrently.

Let use below urls for this use case:
https://python.langchain.com/v0.2/docs/introduction/
https://python.langchain.com/v0.2/docs/concepts/
https://python.langchain.com/v0.2/docs/tutorials/
'''

import threading
import requests
from bs4 import BeautifulSoup

urls = ["https://python.langchain.com/docs/introduction/","https://python.langchain.com/docs/tutorials/","https://python.langchain.com/docs/concepts/"]

def fetch_contents(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    print(f'Fetched {len(soup.text)} characters from {url}')

threads = []

for url in urls:
    thread=threading.Thread(target=fetch_contents,args=(urls,))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()


print('All pages content fetched!!')

