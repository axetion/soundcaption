import getpass
import lxml.html
import os.path
import re
import requests
import sys
import shutil
import time

QUERY = 'https://freesound.org/search?f=duration:[0%20TO%2030]%20license:"Creative+Commons+0"&page='
rm_whitespace = re.compile("\s+")
get_stars = re.compile("\d+")


def login(user, pw):
    session = requests.session()
    csrf = session.get("https://freesound.org/home/login/", stream=True)
    csrf.raw.decode_content = True

    session.post("https://freesound.org/home/login/", headers={"Referer": "https://freesound.org/home/login/"}, data={
        "csrfmiddlewaretoken": lxml.html.parse(csrf.raw).xpath("//input[@name='csrfmiddlewaretoken']")[0].get("value"),
        "username": user,
        "password": pw
    })


def wrap_request(url, timeout, callback, retries=3):
    for _ in range(retries):
        try:
            request = session.get(url, stream=True, timeout=timeout)
            request.raw.decode_content = True
            return callback(request.raw)
        except Exception:
            time.sleep(1)

    raise Exception(url)


if __name__ == "__main__":
    user = input("Username: ")
    pw = getpass.getpass()

    login(user, pw)

    pagecount = 2
    page = 1
    num_sounds = 1

    while page < pagecount:
        pageno = str(page)
        print("Page " + pageno)

        dom = wrap_request(query + pageno, 30, lxml.html.parse)
        new_pagecount = dom.xpath("//li[@class='last-page']/a")

        if new_pagecount:
            pagecount = int(new_pagecount[0].text)

        for sound in dom.xpath("//div[@class='sample_player_small']"):
            links = sound.xpath(".//a[@class='title']")

            description_href = links[0].get("href")
            sound_id = description_href[:-1].rsplit("/", 1)[1]

            print("Retrieving " + sound_id + " (" + str(num_sounds) + ")")
            if os.path.exists(sound_id + ".txt"):
                print("Already have this sound")
                continue

            stars = sound.xpath(".//li[@class='current-rating']")[0].get("style")
            stars = int(get_stars.search(stars)[0])

            if stars > 0 and stars < 50:
                print("Rejected for having a poor rating")
                continue

            print("Stars: " + str(stars))

            description_dom = wrap_request("https://freesound.org" + description_href, 30, lxml.html.parse)
            description = description_dom.xpath("//div[@id='sound_description']")[0].text_content().strip()

            if len(rm_whitespace.sub("", description)) > 10:
                with open(sound_id + ".txt", "w") as description_file:
                    description_file.write(description)

                audio_href = description_dom.xpath("//a[@id='download_button']")[0].get("href")
                audio_extension = audio_href.rsplit(".", 1)[1]

                with open(sound_id + "." + audio_extension, "wb") as audio_file:
                    wrap_request("https://freesound.org" + audio_href, 800, lambda x: shutil.copyfileobj(x, audio_file))

                num_sounds += 1
                time.sleep(1)
            else:
                print("Rejected for being too short")

        page += 1
        time.sleep(1)
